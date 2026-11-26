//! Player actor: rodio lives on one dedicated thread; the UI only talks
//! Command in / Event out. Generation stamps let the reducer drop stale
//! events after a track switch.
//!
//! Spike-proven rules (see design charter appendix):
//! - every Decoder must know its byte length, or backward seeks fail;
//! - rodio is pinned to 0.22.x (0.20 panics opening M4A).

mod cache_stream;

use std::io::{Cursor, Read, Seek};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc as std_mpsc, Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use rodio::{ChannelCount, Decoder, Player, SampleRate, Source};
use stream_download::{Settings, StreamDownload, StreamHandle, StreamPhase};
use tokio::sync::{mpsc, oneshot};
use yesplaymusic_core::cache::{CacheLease, CacheMetadata};

use crate::spectrum::{SampleBuffer, SampleTap};
pub use cache_stream::CacheWritePlan;
use cache_stream::{CacheImportReader, CacheStreamProvider};

pub enum PlayerCommand {
    PlayCached {
        generation: u64,
        lease: CacheLease,
    },
    PlayUrl {
        generation: u64,
        url: String,
        cache: Option<CacheWritePlan>,
        unm_source: bool,
    },
    PlayBytes {
        generation: u64,
        bytes: Vec<u8>,
        unm_source: bool,
    },
    Play,
    TogglePause,
    SeekTo(Duration),
    SetVolume(f32),
    Stop,
}

#[derive(Debug)]
pub enum PlayerEvent {
    Started {
        generation: u64,
        total: Option<Duration>,
    },
    Position {
        generation: u64,
        position: Duration,
    },
    Paused {
        generation: u64,
        paused: bool,
    },
    Ended {
        generation: u64,
    },
    SeekFailed {
        generation: u64,
        message: String,
    },
    Failed {
        generation: u64,
        message: String,
        cached: Option<CacheMetadata>,
        unm_source: bool,
    },
}

#[derive(Clone)]
pub struct PlayerHandle {
    commands: std_mpsc::Sender<PlayerCommand>,
    wake: std_mpsc::Sender<()>,
    samples: Arc<SampleBuffer>,
}

impl PlayerHandle {
    pub fn send(&self, command: PlayerCommand) {
        if self.commands.send(command).is_ok() {
            let _ = self.wake.send(());
        }
    }

    pub fn samples(&self) -> &SampleBuffer {
        &self.samples
    }
}

pub fn spawn(
    runtime: tokio::runtime::Handle,
) -> (PlayerHandle, mpsc::UnboundedReceiver<PlayerEvent>) {
    spawn_with_backend(
        runtime,
        Box::new(SystemOutputBackend),
        DEVICE_POLL_INTERVAL,
        SWITCH_RETRY_DELAY,
    )
}

fn spawn_with_backend(
    runtime: tokio::runtime::Handle,
    backend: Box<dyn OutputBackend>,
    device_poll_interval: Duration,
    retry_delay: Duration,
) -> (PlayerHandle, mpsc::UnboundedReceiver<PlayerEvent>) {
    let samples = SampleBuffer::shared();
    let (command_tx, command_rx) = std_mpsc::channel();
    let (wake_tx, wake_rx) = std_mpsc::channel();
    let (work_tx, work_rx) = std_mpsc::channel();
    let (sink_error_tx, sink_error_rx) = std_mpsc::channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let actor_wake = wake_tx.clone();
    let output_wake = wake_tx.clone();
    let actor_samples = samples.clone();
    std::thread::Builder::new()
        .name("ypm-player".into())
        .spawn(move || {
            actor(ActorContext {
                runtime,
                commands: command_rx,
                wake: wake_rx,
                wake_tx: actor_wake,
                work_tx,
                work_rx,
                sink_errors: sink_error_rx,
                events: event_tx,
                output: OutputContext {
                    backend,
                    errors: sink_error_tx,
                    wake: output_wake,
                    retry_delay,
                },
                device_poll_interval,
                samples: actor_samples,
            })
        })
        .expect("spawn player thread");
    (
        PlayerHandle {
            commands: command_tx,
            wake: wake_tx,
            samples,
        },
        event_rx,
    )
}

/// Anything rodio can decode from: a file or a stream-download reader.
trait MediaSource: Read + Seek + Send + Sync {}
impl<T: Read + Seek + Send + Sync> MediaSource for T {}

/// Reader plus its byte length — carrying the length is what makes
/// backward seeks work, so the two travel together by construction.
struct Media {
    reader: Box<dyn MediaSource>,
    byte_len: Option<u64>,
    cached: Option<CacheMetadata>,
    cancel: Option<StreamCancel>,
    unm_source: bool,
}

struct UrlMedia {
    url: String,
    cache: Option<CacheWritePlan>,
    unm_source: bool,
}

type StreamCancel = Arc<dyn Fn() + Send + Sync>;
type AudioDecoder = Decoder<Box<dyn MediaSource>>;

struct Decoded {
    source: AudioDecoder,
    total: Option<Duration>,
    cancel: Option<StreamCancel>,
}

struct StartedPlayback {
    cancel: Option<StreamCancel>,
}

struct DecodeFailure {
    message: String,
    cached: Option<CacheMetadata>,
    unm_source: bool,
}

enum WorkResult {
    Opened(Result<Media, DecodeFailure>),
    Decoded(Result<Decoded, DecodeFailure>),
}

struct CompletedWork {
    request: u64,
    generation: u64,
    result: WorkResult,
}

enum PendingTask {
    Open(tokio::task::JoinHandle<()>),
    Decode { _task: std::thread::JoinHandle<()> },
}

struct PendingWork {
    request: u64,
    generation: u64,
    cancel: Option<StreamCancel>,
    task: PendingTask,
}

impl PendingWork {
    fn finish(mut self) {
        self.cancel = None;
    }
}

impl Drop for PendingWork {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel();
        }
        if let PendingTask::Open(task) = &self.task {
            task.abort();
        }
    }
}

// Word-synchronised lyrics need a responsive clock without busy-redrawing.
const TICK: Duration = Duration::from_millis(50);
/// Poll interval for detecting OS default output device switches (e.g.
/// Windows taskbar audio switcher) without noticeable latency or overhead.
const DEVICE_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// One brief retry when opening the replacement sink right after an OS
/// device switch (WASAPI endpoints can momentarily report busy).
const SWITCH_RETRY_DELAY: Duration = Duration::from_millis(150);

struct SharedOutput {
    queue: Mutex<rodio::queue::SourcesQueueOutput>,
    active_token: AtomicU64,
}

/// Bridges the single long-lived `rodio::Player` queue into whichever OS
/// `MixerDeviceSink` is currently attached. Flipping `active_token` cleanly
/// detaches an old sink so playback can pause on disconnect or hot-switch
/// devices without re-downloading or re-decoding the active track.
struct OutputBridge {
    shared: Arc<SharedOutput>,
    token: u64,
}

impl Iterator for OutputBridge {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.shared.active_token.load(Ordering::Acquire) != self.token {
            return None;
        }
        let mut queue = self
            .shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if self.shared.active_token.load(Ordering::Acquire) != self.token {
            return None;
        }
        queue.next()
    }
}

impl Source for OutputBridge {
    fn current_span_len(&self) -> Option<usize> {
        if self.shared.active_token.load(Ordering::Acquire) != self.token {
            return Some(0);
        }
        self.shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .current_span_len()
    }

    fn channels(&self) -> ChannelCount {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .channels()
    }

    fn sample_rate(&self) -> SampleRate {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceChange {
    Unchanged,
    Switched,
    Disconnected,
}

enum SinkHandle {
    Device {
        _sink: rodio::MixerDeviceSink,
        device_id: Option<rodio::cpal::DeviceId>,
    },
    #[cfg(test)]
    Silent { _bridge: OutputBridge },
    #[cfg(test)]
    Fake {
        device_id: u64,
        _bridge: OutputBridge,
    },
}

struct ActiveSink {
    token: u64,
    handle: SinkHandle,
}

trait OutputBackend: Send + 'static {
    fn open(
        &mut self,
        bridge: OutputBridge,
        token: u64,
        errors: std_mpsc::Sender<u64>,
        wake: std_mpsc::Sender<()>,
    ) -> anyhow::Result<SinkHandle>;

    fn poll_device_change(&mut self, sink: &SinkHandle) -> DeviceChange;
}

struct SystemOutputBackend;

impl OutputBackend for SystemOutputBackend {
    fn open(
        &mut self,
        bridge: OutputBridge,
        token: u64,
        errors: std_mpsc::Sender<u64>,
        wake: std_mpsc::Sender<()>,
    ) -> anyhow::Result<SinkHandle> {
        use rodio::cpal::traits::{DeviceTrait, HostTrait};

        let host = rodio::cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("no default audio output device"))?;
        let device_id = device.id().ok();
        let builder = rodio::DeviceSinkBuilder::from_device(device)
            .map_err(|error| anyhow::anyhow!("{error}"))?
            .with_error_callback(move |error| {
                tracing::warn!(?error, token, "audio output stream error");
                let _ = errors.send(token);
                let _ = wake.send(());
            });
        let mut sink = builder
            .open_sink_or_fallback()
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        // Avoid stderr drop chatter corrupting the TUI when replacing sinks.
        sink.log_on_drop(false);
        sink.mixer().add(bridge);
        Ok(SinkHandle::Device {
            _sink: sink,
            device_id,
        })
    }

    fn poll_device_change(&mut self, sink: &SinkHandle) -> DeviceChange {
        use rodio::cpal::traits::{DeviceTrait, HostTrait};

        let SinkHandle::Device {
            device_id: Some(current_id),
            ..
        } = sink
        else {
            return DeviceChange::Unchanged;
        };
        let host = rodio::cpal::default_host();
        let Some(default_device) = host.default_output_device() else {
            return DeviceChange::Disconnected;
        };
        let Ok(default_id) = default_device.id() else {
            return DeviceChange::Unchanged;
        };
        if &default_id == current_id {
            return DeviceChange::Unchanged;
        }
        // Distinguish manual OS output switching (previous device still active)
        // from physical/Bluetooth disconnect (previous device gone): only
        // hot-switch when the previous device is still present; a vanished
        // device auto-pauses instead of blasting laptop speakers.
        let still_present = host.output_devices().ok().is_some_and(|mut devices| {
            devices.any(|device| device.id().is_ok_and(|id| &id == current_id))
        });
        if still_present {
            DeviceChange::Switched
        } else {
            DeviceChange::Disconnected
        }
    }
}

struct OutputContext {
    backend: Box<dyn OutputBackend>,
    errors: std_mpsc::Sender<u64>,
    wake: std_mpsc::Sender<()>,
    retry_delay: Duration,
}

struct Engine {
    player: Player,
    shared: Arc<SharedOutput>,
    active_sink: Option<ActiveSink>,
    next_token: u64,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.drop_sink();
    }
}

impl Engine {
    fn open(volume: f32, output: &mut OutputContext) -> anyhow::Result<Self> {
        let (player, queue_output) = Player::new();
        player.set_volume(volume);
        let shared = Arc::new(SharedOutput {
            queue: Mutex::new(queue_output),
            active_token: AtomicU64::new(0),
        });
        let mut engine = Self {
            player,
            shared,
            active_sink: None,
            next_token: 1,
        };
        engine.reopen_sink(output)?;
        Ok(engine)
    }

    fn drop_sink(&mut self) {
        self.shared.active_token.store(0, Ordering::Release);
        self.active_sink = None;
    }

    fn reopen_sink(&mut self, output: &mut OutputContext) -> anyhow::Result<()> {
        self.drop_sink();
        let token = self.next_token;
        self.next_token += 1;
        self.shared.active_token.store(token, Ordering::Release);
        let bridge = OutputBridge {
            shared: self.shared.clone(),
            token,
        };
        match output
            .backend
            .open(bridge, token, output.errors.clone(), output.wake.clone())
        {
            Ok(handle) => {
                self.active_sink = Some(ActiveSink { token, handle });
                Ok(())
            }
            Err(error) => {
                self.shared.active_token.store(0, Ordering::Release);
                Err(error)
            }
        }
    }

    fn hot_switch_sink(&mut self, output: &mut OutputContext) -> anyhow::Result<()> {
        match self.reopen_sink(output) {
            Ok(()) => Ok(()),
            Err(first_error) => {
                if !output.retry_delay.is_zero() {
                    std::thread::sleep(output.retry_delay);
                    if self.reopen_sink(output).is_ok() {
                        return Ok(());
                    }
                }
                Err(first_error)
            }
        }
    }

    fn ensure_output_ready(&mut self, output: &mut OutputContext) -> anyhow::Result<()> {
        let needs_reopen = match &self.active_sink {
            None => true,
            Some(active) => {
                output.backend.poll_device_change(&active.handle) != DeviceChange::Unchanged
            }
        };
        if needs_reopen {
            self.hot_switch_sink(output)?;
        }
        Ok(())
    }
}

struct ActorContext {
    runtime: tokio::runtime::Handle,
    commands: std_mpsc::Receiver<PlayerCommand>,
    wake: std_mpsc::Receiver<()>,
    wake_tx: std_mpsc::Sender<()>,
    work_tx: std_mpsc::Sender<CompletedWork>,
    work_rx: std_mpsc::Receiver<CompletedWork>,
    sink_errors: std_mpsc::Receiver<u64>,
    events: mpsc::UnboundedSender<PlayerEvent>,
    output: OutputContext,
    device_poll_interval: Duration,
    samples: Arc<SampleBuffer>,
}

fn actor(context: ActorContext) {
    let ActorContext {
        runtime,
        commands,
        wake,
        wake_tx,
        work_tx,
        work_rx,
        sink_errors,
        events,
        mut output,
        device_poll_interval,
        samples,
    } = context;
    let mut engine: Option<Engine> = None;
    let mut active_generation: Option<u64> = None;
    let mut volume = 1.0_f32;
    let mut pending: Option<PendingWork> = None;
    let mut active_cancel: Option<StreamCancel> = None;
    let mut next_request = 1_u64;
    let mut last_tick = Instant::now();
    let mut last_device_poll = Instant::now();

    loop {
        let wait = TICK
            .saturating_sub(last_tick.elapsed())
            .min(device_poll_interval.saturating_sub(last_device_poll.elapsed()));
        let _ = wake.recv_timeout(wait);

        while let Ok(error_token) = sink_errors.try_recv() {
            let matches_active = engine
                .as_ref()
                .and_then(|engine| engine.active_sink.as_ref())
                .is_some_and(|active| active.token == error_token);
            if matches_active {
                handle_device_disconnect(engine.as_mut(), active_generation, &events);
            }
        }

        let mut disconnected = false;
        loop {
            let command = match commands.try_recv() {
                Ok(command) => command,
                Err(std_mpsc::TryRecvError::Empty) => break,
                Err(std_mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            };

            match command {
                PlayerCommand::PlayCached {
                    generation: g,
                    lease,
                } => {
                    drop(pending.take());
                    stop(&engine, &mut active_cancel);
                    samples.clear();
                    active_generation = None;
                    let request = next_request;
                    next_request += 1;
                    pending = Some(spawn_decode(
                        request,
                        g,
                        open_cached(lease),
                        work_tx.clone(),
                        wake_tx.clone(),
                    ));
                }
                PlayerCommand::PlayUrl {
                    generation: g,
                    url,
                    cache,
                    unm_source,
                } => {
                    drop(pending.take());
                    stop(&engine, &mut active_cancel);
                    samples.clear();
                    active_generation = None;
                    let request = next_request;
                    next_request += 1;
                    pending = Some(spawn_open(
                        &runtime,
                        request,
                        g,
                        UrlMedia {
                            url,
                            cache,
                            unm_source,
                        },
                        work_tx.clone(),
                        wake_tx.clone(),
                    ));
                }
                PlayerCommand::PlayBytes {
                    generation: g,
                    bytes,
                    unm_source,
                } => {
                    drop(pending.take());
                    stop(&engine, &mut active_cancel);
                    samples.clear();
                    active_generation = None;
                    let request = next_request;
                    next_request += 1;
                    pending = Some(spawn_decode(
                        request,
                        g,
                        open_bytes(bytes, unm_source),
                        work_tx.clone(),
                        wake_tx.clone(),
                    ));
                }
                PlayerCommand::Play => {
                    if let (Some(engine), Some(generation)) = (engine.as_mut(), active_generation) {
                        if let Err(error) = engine.ensure_output_ready(&mut output) {
                            cancel_active(&mut active_cancel);
                            engine.player.stop();
                            active_generation = None;
                            let _ = events.send(PlayerEvent::Failed {
                                generation,
                                message: format!("audio device unavailable: {error}"),
                                cached: None,
                                unm_source: false,
                            });
                        } else {
                            engine.player.play();
                        }
                    }
                }
                PlayerCommand::TogglePause => {
                    if let (Some(engine), Some(generation)) = (engine.as_mut(), active_generation) {
                        let paused = !engine.player.is_paused();
                        if paused {
                            engine.player.pause();
                            let _ = events.send(PlayerEvent::Paused { generation, paused });
                        } else if let Err(error) = engine.ensure_output_ready(&mut output) {
                            cancel_active(&mut active_cancel);
                            engine.player.stop();
                            active_generation = None;
                            let _ = events.send(PlayerEvent::Failed {
                                generation,
                                message: format!("audio device unavailable: {error}"),
                                cached: None,
                                unm_source: false,
                            });
                        } else {
                            engine.player.play();
                            let _ = events.send(PlayerEvent::Paused { generation, paused });
                        }
                    }
                }
                PlayerCommand::SeekTo(position) => {
                    if let (Some(engine), Some(generation)) = (&engine, active_generation) {
                        if let Err(error) = engine.player.try_seek(position) {
                            let _ = events.send(PlayerEvent::SeekFailed {
                                generation,
                                message: format!("seek failed: {error}"),
                            });
                        }
                    }
                }
                PlayerCommand::SetVolume(value) => {
                    volume = value.clamp(0.0, 1.5);
                    if let Some(engine) = &engine {
                        engine.player.set_volume(volume);
                    }
                }
                PlayerCommand::Stop => {
                    drop(pending.take());
                    stop(&engine, &mut active_cancel);
                    samples.clear();
                    active_generation = None;
                }
            }
        }

        if disconnected {
            drop(pending.take());
            stop(&engine, &mut active_cancel);
            engine = None;
            break;
        }

        while let Ok(completed) = work_rx.try_recv() {
            let is_current = pending.as_ref().is_some_and(|candidate| {
                candidate.request == completed.request
                    && candidate.generation == completed.generation
            });
            if !is_current {
                continue;
            }

            pending.take().expect("current player work").finish();
            match completed.result {
                WorkResult::Opened(Ok(media)) => {
                    pending = Some(spawn_decode(
                        completed.request,
                        completed.generation,
                        media,
                        work_tx.clone(),
                        wake_tx.clone(),
                    ));
                }
                WorkResult::Opened(Err(error)) => {
                    report_failure(&events, completed.generation, error);
                }
                WorkResult::Decoded(Ok(decoded)) => {
                    if let Some(started) = start_decoded(
                        &mut engine,
                        &mut output,
                        volume,
                        completed.generation,
                        &events,
                        decoded,
                        samples.clone(),
                    ) {
                        active_generation = Some(completed.generation);
                        active_cancel = started.cancel;
                    }
                }
                WorkResult::Decoded(Err(error)) => {
                    report_failure(&events, completed.generation, error);
                }
            }
        }

        if last_device_poll.elapsed() >= device_poll_interval {
            last_device_poll = Instant::now();
            if let Some(engine) = engine.as_mut() {
                let change = engine
                    .active_sink
                    .as_ref()
                    .map_or(DeviceChange::Unchanged, |active| {
                        output.backend.poll_device_change(&active.handle)
                    });
                match change {
                    DeviceChange::Unchanged => {}
                    DeviceChange::Disconnected => {
                        tracing::info!("audio output device disconnected; pausing playback");
                        handle_device_disconnect(Some(engine), active_generation, &events);
                    }
                    DeviceChange::Switched => {
                        tracing::info!("default audio output changed; switching sink");
                        if let Err(error) = engine.hot_switch_sink(&mut output) {
                            tracing::warn!(%error, "failed to switch audio output device");
                            handle_device_disconnect(Some(engine), active_generation, &events);
                        }
                    }
                }
            }
        }

        if last_tick.elapsed() >= TICK {
            last_tick = Instant::now();
            if let (Some(engine), Some(generation)) = (&engine, active_generation) {
                if engine.player.empty() {
                    active_generation = None;
                    active_cancel = None;
                    let _ = events.send(PlayerEvent::Ended { generation });
                } else if !engine.player.is_paused() {
                    let _ = events.send(PlayerEvent::Position {
                        generation,
                        position: engine.player.get_pos(),
                    });
                }
            }
        }
    }
}

fn handle_device_disconnect(
    engine: Option<&mut Engine>,
    active_generation: Option<u64>,
    events: &mpsc::UnboundedSender<PlayerEvent>,
) {
    let Some(engine) = engine else {
        return;
    };
    engine.drop_sink();
    if let Some(generation) = active_generation {
        if !engine.player.empty() && !engine.player.is_paused() {
            engine.player.pause();
            let _ = events.send(PlayerEvent::Paused {
                generation,
                paused: true,
            });
        }
    }
}

fn cancel_active(active_cancel: &mut Option<StreamCancel>) {
    if let Some(cancel) = active_cancel.take() {
        cancel();
    }
}

fn stop(engine: &Option<Engine>, active_cancel: &mut Option<StreamCancel>) {
    cancel_active(active_cancel);
    if let Some(engine) = engine {
        engine.player.stop();
    }
}

fn spawn_open(
    runtime: &tokio::runtime::Handle,
    request: u64,
    generation: u64,
    media: UrlMedia,
    completed: std_mpsc::Sender<CompletedWork>,
    wake: std_mpsc::Sender<()>,
) -> PendingWork {
    let task = runtime.spawn(async move {
        let result = open_url(&media.url, media.cache, media.unm_source)
            .await
            .map_err(|error| DecodeFailure {
                message: error.to_string(),
                cached: None,
                unm_source: media.unm_source,
            });
        let _ = completed.send(CompletedWork {
            request,
            generation,
            result: WorkResult::Opened(result),
        });
        let _ = wake.send(());
    });
    PendingWork {
        request,
        generation,
        cancel: None,
        task: PendingTask::Open(task),
    }
}

fn spawn_decode(
    request: u64,
    generation: u64,
    media: Media,
    completed: std_mpsc::Sender<CompletedWork>,
    wake: std_mpsc::Sender<()>,
) -> PendingWork {
    let cancel = media.cancel.clone();
    let task = std::thread::Builder::new()
        .name("ypm-decoder".into())
        .spawn(move || {
            let result = decode(media);
            let _ = completed.send(CompletedWork {
                request,
                generation,
                result: WorkResult::Decoded(result),
            });
            let _ = wake.send(());
        })
        .expect("spawn decoder thread");
    PendingWork {
        request,
        generation,
        cancel,
        task: PendingTask::Decode { _task: task },
    }
}

fn open_cached(lease: CacheLease) -> Media {
    let metadata = *lease.metadata();
    Media {
        reader: Box::new(lease),
        byte_len: Some(metadata.bytes),
        cached: Some(metadata),
        cancel: None,
        unm_source: false,
    }
}

fn open_bytes(bytes: Vec<u8>, unm_source: bool) -> Media {
    Media {
        byte_len: Some(bytes.len() as u64),
        reader: Box::new(Cursor::new(bytes)),
        cached: None,
        cancel: None,
        unm_source,
    }
}

/// stream-download's reqwest is built with `rustls-no-provider` so the whole
/// binary links a single crypto backend; that variant requires installing the
/// process default provider before the first client is built.
fn install_crypto_provider() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

async fn open_url(
    url: &str,
    cache: Option<CacheWritePlan>,
    unm_source: bool,
) -> anyhow::Result<Media> {
    install_crypto_provider();
    let (provider, import) = CacheStreamProvider::new()?;
    let (complete_tx, complete_rx) = oneshot::channel();
    let mut complete_tx = cache.as_ref().map(|_| complete_tx);
    let settings = Settings::default().on_progress(move |_, state, _| {
        if matches!(state.phase, StreamPhase::Complete) {
            if let Some(complete) = complete_tx.take() {
                let _ = complete.send(());
            }
        }
    });
    let reader = StreamDownload::new_http(url.parse()?, provider, settings).await?;
    let byte_len = reader.content_length();
    let cancellation = reader.cancellation_token();
    let cancel: StreamCancel = Arc::new(move || cancellation.cancel());
    if let Some(plan) = cache {
        spawn_cache_publish(reader.handle(), complete_rx, import, plan);
    }
    Ok(Media {
        reader: Box::new(reader),
        byte_len,
        cached: None,
        cancel: Some(cancel),
        unm_source,
    })
}

fn spawn_cache_publish(
    handle: StreamHandle,
    complete: oneshot::Receiver<()>,
    import: CacheImportReader,
    plan: CacheWritePlan,
) {
    tokio::spawn(async move {
        if complete.await.is_err() {
            return;
        }
        handle.wait_for_completion().await;
        match tokio::task::spawn_blocking(move || cache_stream::publish(import, plan)).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!(%error, "audio cache write failed"),
            Err(error) => tracing::warn!(%error, "audio cache worker failed"),
        }
    });
}

fn decode(media: Media) -> Result<Decoded, DecodeFailure> {
    let Media {
        reader,
        byte_len,
        cached,
        cancel,
        unm_source,
    } = media;
    let mut builder = Decoder::builder().with_data(reader).with_seekable(true);
    if let Some(byte_len) = byte_len {
        builder = builder.with_byte_len(byte_len);
    }
    let source = builder.build().map_err(|error| DecodeFailure {
        message: error.to_string(),
        cached,
        unm_source,
    })?;
    let total = rodio::Source::total_duration(&source);
    Ok(Decoded {
        source,
        total,
        cancel,
    })
}

fn start_decoded(
    engine: &mut Option<Engine>,
    output: &mut OutputContext,
    volume: f32,
    generation: u64,
    events: &mpsc::UnboundedSender<PlayerEvent>,
    decoded: Decoded,
    samples: Arc<SampleBuffer>,
) -> Option<StartedPlayback> {
    let engine = match engine {
        Some(engine) => {
            if let Err(error) = engine.ensure_output_ready(output) {
                let _ = events.send(PlayerEvent::Failed {
                    generation,
                    message: format!("audio device unavailable: {error}"),
                    cached: None,
                    unm_source: false,
                });
                return None;
            }
            engine
        }
        None => match Engine::open(volume, output) {
            Ok(opened) => engine.insert(opened),
            Err(error) => {
                let _ = events.send(PlayerEvent::Failed {
                    generation,
                    message: format!("audio device unavailable: {error}"),
                    cached: None,
                    unm_source: false,
                });
                return None;
            }
        },
    };

    engine.player.stop();
    engine.player.pause();
    engine
        .player
        .append(SampleTap::new(decoded.source, samples));
    let _ = events.send(PlayerEvent::Started {
        generation,
        total: decoded.total,
    });
    Some(StartedPlayback {
        cancel: decoded.cancel,
    })
}

fn report_failure(
    events: &mpsc::UnboundedSender<PlayerEvent>,
    generation: u64,
    failure: DecodeFailure,
) {
    let _ = events.send(PlayerEvent::Failed {
        generation,
        message: failure.message,
        cached: failure.cached,
        unm_source: failure.unm_source,
    });
}

#[cfg(test)]
struct SilentOutputBackend;

#[cfg(test)]
impl OutputBackend for SilentOutputBackend {
    fn open(
        &mut self,
        bridge: OutputBridge,
        _token: u64,
        _errors: std_mpsc::Sender<u64>,
        _wake: std_mpsc::Sender<()>,
    ) -> anyhow::Result<SinkHandle> {
        Ok(SinkHandle::Silent { _bridge: bridge })
    }

    fn poll_device_change(&mut self, _sink: &SinkHandle) -> DeviceChange {
        DeviceChange::Unchanged
    }
}

#[cfg(test)]
fn spawn_silent(
    runtime: tokio::runtime::Handle,
) -> (PlayerHandle, mpsc::UnboundedReceiver<PlayerEvent>) {
    spawn_with_backend(
        runtime,
        Box::new(SilentOutputBackend),
        DEVICE_POLL_INTERVAL,
        Duration::ZERO,
    )
}

#[cfg(test)]
fn start<F>(
    engine: &mut Option<Engine>,
    volume: f32,
    generation: u64,
    events: &mpsc::UnboundedSender<PlayerEvent>,
    open: F,
) -> bool
where
    F: FnOnce() -> anyhow::Result<Media>,
{
    let media = match open() {
        Ok(media) => media,
        Err(error) => {
            report_failure(
                events,
                generation,
                DecodeFailure {
                    message: error.to_string(),
                    cached: None,
                    unm_source: false,
                },
            );
            return false;
        }
    };
    let decoded = match decode(media) {
        Ok(decoded) => decoded,
        Err(error) => {
            report_failure(events, generation, error);
            return false;
        }
    };
    let (error_tx, _error_rx) = std_mpsc::channel();
    let (wake_tx, _wake_rx) = std_mpsc::channel();
    let mut output = OutputContext {
        backend: Box::new(SilentOutputBackend),
        errors: error_tx,
        wake: wake_tx,
        retry_delay: Duration::ZERO,
    };
    start_decoded(
        engine,
        &mut output,
        volume,
        generation,
        events,
        decoded,
        SampleBuffer::shared(),
    )
    .is_some()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::Path;
    use std::sync::{mpsc as std_mpsc, Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use tokio::sync::{mpsc, oneshot};
    use yesplaymusic_core::cache::{
        AudioCodec, AudioQuality, CacheKey, CacheWriteRequest, TrackCache,
    };

    use super::{
        open_cached, open_url, spawn, spawn_silent, spawn_with_backend, start, CacheWritePlan,
        DeviceChange, OutputBackend, OutputBridge, PlayerCommand, PlayerEvent, SinkHandle,
    };

    const AUDIO_BODY: &[u8] = b"complete cache body";
    const WAV_BODY: &[u8] = b"RIFF,\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0@\x1f\0\0\x80>\0\0\x02\0\x10\0data\x08\0\0\0\0\0\0\0\0\0\0\0";

    #[test]
    fn position_updates_are_frequent_enough_for_word_synced_lyrics() {
        assert!(super::TICK <= Duration::from_millis(100));
    }

    fn cache_request(track_id: i64, expected_bytes: u64) -> CacheWriteRequest {
        CacheWriteRequest::new(
            CacheKey::new(track_id, AudioQuality::High320),
            AudioCodec::Mp3,
            320_000,
        )
        .with_expected_bytes(expected_bytes)
        .with_expected_md5([
            0xe8, 0xa9, 0x92, 0x1b, 0xe8, 0x6b, 0xc2, 0x3f, 0x73, 0x2f, 0xa2, 0x62, 0x13, 0xec,
            0x6e, 0x05,
        ])
    }

    struct HttpServer {
        url: String,
        thread: thread::JoinHandle<()>,
    }

    impl HttpServer {
        fn complete(body: &'static [u8]) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind complete server");
            let address = listener.local_addr().expect("server address");
            let thread = thread::spawn(move || {
                let (mut socket, _) = listener.accept().expect("accept player request");
                read_request(&mut socket);
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .expect("write response headers");
                socket.write_all(body).expect("write response body");
            });
            Self {
                url: format!("http://{address}/audio"),
                thread,
            }
        }

        fn stalled_prefix(
            prefix: &'static [u8],
            content_length: usize,
        ) -> (Self, oneshot::Receiver<()>, std_mpsc::Receiver<()>) {
            let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind partial server");
            let address = listener.local_addr().expect("server address");
            let (prefix_tx, prefix_sent) = oneshot::channel();
            let (closed_tx, closed) = std_mpsc::channel();
            let thread = thread::spawn(move || {
                let (mut socket, _) = listener.accept().expect("accept player request");
                read_request(&mut socket);
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
                )
                .expect("write response headers");
                socket.write_all(prefix).expect("write response prefix");
                socket.flush().expect("flush response prefix");
                let _ = prefix_tx.send(());
                let mut buffer = [0_u8; 256];
                while socket.read(&mut buffer).is_ok_and(|read| read != 0) {}
                let _ = closed_tx.send(());
            });
            (
                Self {
                    url: format!("http://{address}/audio"),
                    thread,
                },
                prefix_sent,
                closed,
            )
        }

        fn join(self) {
            self.thread.join().expect("join HTTP server");
        }
    }

    fn read_request(socket: &mut std::net::TcpStream) {
        socket
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("set request timeout");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 256];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).expect("read request");
            assert_ne!(read, 0, "request ended before headers");
            request.extend_from_slice(&buffer[..read]);
        }
    }

    async fn wait_for_cache(root: &Path, key: CacheKey) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if TrackCache::open(root)
                    .expect("open cache")
                    .lookup(key)
                    .expect("lookup cache")
                    .is_some()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("completed stream should be cached");
    }

    struct StalledServer {
        url: String,
        accepted: Option<oneshot::Receiver<()>>,
        closed: std_mpsc::Receiver<()>,
        thread: thread::JoinHandle<()>,
    }

    impl StalledServer {
        fn spawn() -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind stalled server");
            let address = listener.local_addr().expect("server address");
            let (accepted_tx, accepted) = oneshot::channel();
            let (closed_tx, closed) = std_mpsc::channel();
            let thread = thread::spawn(move || {
                let (mut socket, _) = listener.accept().expect("accept player request");
                read_request(&mut socket);
                write!(
                    socket,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    WAV_BODY.len()
                )
                .expect("write stalled response headers");
                socket
                    .write_all(&WAV_BODY[..16])
                    .expect("write stalled response prefix");
                socket.flush().expect("flush stalled response prefix");
                socket
                    .set_read_timeout(Some(Duration::from_millis(20)))
                    .expect("set socket timeout");
                let _ = accepted_tx.send(());

                let mut buffer = [0_u8; 1024];
                loop {
                    match socket.read(&mut buffer) {
                        Ok(0) => {
                            let _ = closed_tx.send(());
                            break;
                        }
                        Ok(_) => {}
                        Err(error)
                            if matches!(
                                error.kind(),
                                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                            ) => {}
                        Err(_) => {
                            let _ = closed_tx.send(());
                            break;
                        }
                    }
                }
            });
            Self {
                url: format!("http://{address}/audio"),
                accepted: Some(accepted),
                closed,
                thread,
            }
        }

        async fn wait_until_requested(&mut self) {
            let accepted = self.accepted.take().expect("server request awaited once");
            tokio::time::timeout(Duration::from_secs(1), accepted)
                .await
                .expect("player should connect to local server")
                .expect("stalled server should report the connection");
        }

        fn wait_until_closed(&self, timeout: Duration) {
            self.closed
                .recv_timeout(timeout)
                .expect("cancelling the open should close its HTTP connection");
        }

        fn join(self) {
            self.thread.join().expect("join stalled server");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stop_cancels_a_stalled_decoder() {
        let mut server = StalledServer::spawn();
        let (player, mut events) = spawn(tokio::runtime::Handle::current());
        player.send(PlayerCommand::PlayUrl {
            generation: 1,
            url: server.url.clone(),
            cache: None,
            unm_source: false,
        });
        server.wait_until_requested().await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        player.send(PlayerCommand::Stop);
        server.wait_until_closed(Duration::from_millis(500));
        assert!(
            tokio::time::timeout(Duration::from_millis(100), events.recv())
                .await
                .is_err(),
            "the cancelled track must not emit a playback event"
        );

        drop(player);
        server.join();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_new_play_replaces_a_stalled_decoder_without_starting_it() {
        let mut stalled = StalledServer::spawn();
        let replacement = HttpServer::complete(WAV_BODY);
        let (player, mut events) = spawn_silent(tokio::runtime::Handle::current());
        player.send(PlayerCommand::PlayUrl {
            generation: 1,
            url: stalled.url.clone(),
            cache: None,
            unm_source: false,
        });
        stalled.wait_until_requested().await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        let deadline = std::time::Instant::now() + Duration::from_millis(500);
        player.send(PlayerCommand::PlayUrl {
            generation: 2,
            url: replacement.url.clone(),
            cache: None,
            unm_source: false,
        });
        let event = tokio::time::timeout(
            deadline.saturating_duration_since(std::time::Instant::now()),
            events.recv(),
        )
        .await
        .expect("replacement should start without waiting for the first connection")
        .expect("player event channel should remain open");
        assert!(matches!(event, PlayerEvent::Started { generation: 2, .. }));
        stalled.wait_until_closed(deadline.saturating_duration_since(std::time::Instant::now()));
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_millis(100), events.recv()).await
        {
            assert!(
                !matches!(
                    event,
                    PlayerEvent::Started { generation: 1, .. }
                        | PlayerEvent::Position { generation: 1, .. }
                        | PlayerEvent::Paused { generation: 1, .. }
                        | PlayerEvent::Ended { generation: 1 }
                        | PlayerEvent::Failed { generation: 1, .. }
                ),
                "the replaced track must not emit a playback event"
            );
        }

        drop(player);
        stalled.join();
        replacement.join();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn decoded_track_waits_for_explicit_play_before_reporting_position() {
        let server = HttpServer::complete(WAV_BODY);
        let (player, mut events) = spawn_silent(tokio::runtime::Handle::current());
        player.send(PlayerCommand::PlayUrl {
            generation: 7,
            url: server.url.clone(),
            cache: None,
            unm_source: false,
        });

        let started = tokio::time::timeout(Duration::from_secs(1), events.recv())
            .await
            .expect("decoded track should become ready")
            .expect("player event channel should remain open");
        assert!(matches!(
            started,
            PlayerEvent::Started { generation: 7, .. }
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(350), events.recv())
                .await
                .is_err(),
            "decoded audio must remain paused until the app finishes any restore seek"
        );

        player.send(PlayerCommand::Play);
        let position = tokio::time::timeout(Duration::from_millis(500), events.recv())
            .await
            .expect("explicit play should start position updates")
            .expect("player event channel should remain open");
        assert!(matches!(
            position,
            PlayerEvent::Position { generation: 7, .. }
        ));

        drop(player);
        server.join();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_memory_audio_uses_the_same_generation_stamped_decode_path() {
        let (player, mut events) = spawn_silent(tokio::runtime::Handle::current());

        player.send(PlayerCommand::PlayBytes {
            generation: 11,
            bytes: WAV_BODY.to_vec(),
            unm_source: true,
        });

        let event = tokio::time::timeout(Duration::from_secs(1), events.recv())
            .await
            .expect("in-memory audio should decode")
            .expect("player event channel should remain open");
        assert!(matches!(event, PlayerEvent::Started { generation: 11, .. }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unm_url_open_failure_keeps_its_origin() {
        let (player, mut events) = spawn_silent(tokio::runtime::Handle::current());

        player.send(PlayerCommand::PlayUrl {
            generation: 12,
            url: "not a url".into(),
            cache: None,
            unm_source: true,
        });

        let event = tokio::time::timeout(Duration::from_secs(1), events.recv())
            .await
            .expect("invalid UNM URL should fail")
            .expect("player event channel should remain open");
        assert!(matches!(
            event,
            PlayerEvent::Failed {
                generation: 12,
                cached: None,
                unm_source: true,
                ..
            }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unm_bytes_decode_failure_keeps_its_origin() {
        let (player, mut events) = spawn_silent(tokio::runtime::Handle::current());

        player.send(PlayerCommand::PlayBytes {
            generation: 13,
            bytes: Vec::new(),
            unm_source: true,
        });

        let event = tokio::time::timeout(Duration::from_secs(1), events.recv())
            .await
            .expect("invalid UNM bytes should fail")
            .expect("player event channel should remain open");
        assert!(matches!(
            event,
            PlayerEvent::Failed {
                generation: 13,
                cached: None,
                unm_source: true,
                ..
            }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn naturally_completed_stream_is_published_with_validated_bytes_and_md5() {
        let cache_dir = tempfile::tempdir().expect("cache directory");
        let server = HttpServer::complete(AUDIO_BODY);
        let request = cache_request(31, AUDIO_BODY.len() as u64);
        let key = request.key;
        let media = open_url(
            &server.url,
            Some(CacheWritePlan {
                root: cache_dir.path().to_path_buf(),
                request,
            }),
            false,
        )
        .await
        .expect("open complete stream");

        let downloaded = tokio::task::spawn_blocking(move || {
            let mut reader = media.reader;
            let mut downloaded = Vec::new();
            reader.read_to_end(&mut downloaded).expect("read stream");
            downloaded
        })
        .await
        .expect("join stream reader");
        assert_eq!(downloaded, AUDIO_BODY);
        wait_for_cache(cache_dir.path(), key).await;

        let cache = TrackCache::open(cache_dir.path()).expect("open published cache");
        let mut lease = cache
            .lookup(key)
            .expect("lookup published cache")
            .expect("published cache entry");
        assert_eq!(lease.metadata().bytes, AUDIO_BODY.len() as u64);
        let mut cached = Vec::new();
        lease.read_to_end(&mut cached).expect("read cached bytes");
        assert_eq!(cached, AUDIO_BODY);
        server.join();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stopping_a_partial_stream_does_not_publish_it() {
        let cache_dir = tempfile::tempdir().expect("cache directory");
        let (server, prefix_sent, closed) = HttpServer::stalled_prefix(b"part", AUDIO_BODY.len());
        let request = cache_request(32, AUDIO_BODY.len() as u64);
        let key = request.key;
        let media = open_url(
            &server.url,
            Some(CacheWritePlan {
                root: cache_dir.path().to_path_buf(),
                request,
            }),
            false,
        )
        .await
        .expect("open partial stream");
        tokio::time::timeout(Duration::from_secs(1), prefix_sent)
            .await
            .expect("server should send a prefix")
            .expect("prefix signal should arrive");

        drop(media);
        closed
            .recv_timeout(Duration::from_millis(500))
            .expect("stopping the stream should close the response");
        tokio::time::sleep(Duration::from_millis(50)).await;

        let cache = TrackCache::open(cache_dir.path()).expect("open cache");
        assert!(cache.lookup(key).expect("lookup cache").is_none());
        server.join();
    }

    #[test]
    fn cached_media_keeps_its_lease_until_the_decoder_source_is_dropped() {
        let cache_dir = tempfile::tempdir().expect("cache directory");
        let key = CacheKey::new(33, AudioQuality::High320);
        let cache = TrackCache::open(cache_dir.path()).expect("open cache");
        let mut writer = cache
            .begin_write(
                CacheWriteRequest::new(key, AudioCodec::Mp3, 320_000)
                    .with_expected_bytes(AUDIO_BODY.len() as u64),
            )
            .expect("begin cache write");
        writer.write_all(AUDIO_BODY).expect("write cache entry");
        writer.finish().expect("finish cache entry");
        let lease = cache
            .lookup(key)
            .expect("lookup cache")
            .expect("cache lease");
        let media = open_cached(lease);

        cache.set_max_bytes(0).expect("evict leased cache");
        assert_eq!(cache.total_bytes().expect("leased cache size"), 19);

        drop(media);
        cache.set_max_bytes(0).expect("evict released cache");
        assert_eq!(cache.total_bytes().expect("released cache size"), 0);
    }

    #[test]
    fn cached_decoder_failure_reports_the_entry_that_must_be_invalidated() {
        let cache_dir = tempfile::tempdir().expect("cache directory");
        let key = CacheKey::new(34, AudioQuality::High320);
        let cache = TrackCache::open(cache_dir.path()).expect("open cache");
        let mut writer = cache
            .begin_write(CacheWriteRequest::new(key, AudioCodec::Mp3, 320_000))
            .expect("begin cache write");
        writer.write_all(b"not audio").expect("write cache entry");
        let metadata = writer.finish().expect("finish cache entry");
        let lease = cache
            .lookup(key)
            .expect("lookup cache")
            .expect("cache lease");
        let (events, mut received) = mpsc::unbounded_channel();
        let mut engine = None;

        assert!(!start(&mut engine, 1.0, 8, &events, || {
            Ok(open_cached(lease))
        }));
        let event = received.try_recv().expect("decoder failure event");
        assert!(matches!(
            event,
            PlayerEvent::Failed {
                generation: 8,
                cached: Some(failed),
                ..
            } if failed == metadata
        ));
    }

    struct FakeOutputState {
        default_device: u64,
        present_devices: Vec<u64>,
        opened_devices: Vec<u64>,
        error_channel: Option<(u64, std_mpsc::Sender<u64>, std_mpsc::Sender<()>)>,
    }

    struct FakeOutputBackend {
        state: Arc<Mutex<FakeOutputState>>,
    }

    impl OutputBackend for FakeOutputBackend {
        fn open(
            &mut self,
            bridge: OutputBridge,
            token: u64,
            errors: std_mpsc::Sender<u64>,
            wake: std_mpsc::Sender<()>,
        ) -> anyhow::Result<SinkHandle> {
            let mut state = self.state.lock().unwrap();
            let device_id = state.default_device;
            state.opened_devices.push(device_id);
            state.error_channel = Some((token, errors, wake));
            Ok(SinkHandle::Fake {
                device_id,
                _bridge: bridge,
            })
        }

        fn poll_device_change(&mut self, sink: &SinkHandle) -> DeviceChange {
            let SinkHandle::Fake { device_id, .. } = sink else {
                return DeviceChange::Unchanged;
            };
            let state = self.state.lock().unwrap();
            if state.default_device == *device_id {
                DeviceChange::Unchanged
            } else if state.present_devices.contains(device_id) {
                DeviceChange::Switched
            } else {
                DeviceChange::Disconnected
            }
        }
    }

    async fn recv_until<F>(
        events: &mut mpsc::UnboundedReceiver<PlayerEvent>,
        limit: Duration,
        mut predicate: F,
    ) -> PlayerEvent
    where
        F: FnMut(&PlayerEvent) -> bool,
    {
        tokio::time::timeout(limit, async {
            loop {
                let event = events.recv().await.expect("player event channel open");
                if predicate(&event) {
                    return event;
                }
            }
        })
        .await
        .expect("expected player event before timeout")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_error_auto_pauses_and_resuming_reopens_the_output_device() {
        let state = Arc::new(Mutex::new(FakeOutputState {
            default_device: 1,
            present_devices: vec![1],
            opened_devices: Vec::new(),
            error_channel: None,
        }));
        let (player, mut events) = spawn_with_backend(
            tokio::runtime::Handle::current(),
            Box::new(FakeOutputBackend {
                state: state.clone(),
            }),
            Duration::from_secs(60),
            Duration::ZERO,
        );

        player.send(PlayerCommand::PlayBytes {
            generation: 21,
            bytes: WAV_BODY.to_vec(),
            unm_source: false,
        });
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Started { generation: 21, .. })
        })
        .await;
        player.send(PlayerCommand::Play);
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Position { generation: 21, .. })
        })
        .await;

        let (token, err_tx, wake_tx) = state.lock().unwrap().error_channel.clone().unwrap();
        err_tx.send(token).unwrap();
        wake_tx.send(()).unwrap();
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(
                event,
                PlayerEvent::Paused {
                    generation: 21,
                    paused: true
                }
            )
        })
        .await;

        {
            let mut guard = state.lock().unwrap();
            guard.default_device = 2;
            guard.present_devices = vec![2];
        }
        player.send(PlayerCommand::TogglePause);
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(
                event,
                PlayerEvent::Paused {
                    generation: 21,
                    paused: false
                }
            )
        })
        .await;
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Position { generation: 21, .. })
        })
        .await;

        assert_eq!(state.lock().unwrap().opened_devices, vec![1, 2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn manual_default_device_switch_hot_switches_without_pausing() {
        let state = Arc::new(Mutex::new(FakeOutputState {
            default_device: 1,
            present_devices: vec![1, 2],
            opened_devices: Vec::new(),
            error_channel: None,
        }));
        let (player, mut events) = spawn_with_backend(
            tokio::runtime::Handle::current(),
            Box::new(FakeOutputBackend {
                state: state.clone(),
            }),
            Duration::from_millis(20),
            Duration::ZERO,
        );

        player.send(PlayerCommand::PlayBytes {
            generation: 22,
            bytes: WAV_BODY.to_vec(),
            unm_source: false,
        });
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Started { generation: 22, .. })
        })
        .await;
        player.send(PlayerCommand::Play);
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Position { generation: 22, .. })
        })
        .await;
        while events.try_recv().is_ok() {}

        state.lock().unwrap().default_device = 2;
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.lock().unwrap().opened_devices != vec![1, 2] {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("default output switch should hot-switch the sink");

        let mut saw_position = false;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(250);
        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, events.recv()).await {
            assert!(
                !matches!(event, PlayerEvent::Paused { .. }),
                "manual device switch must not pause playback: {event:?}"
            );
            if matches!(event, PlayerEvent::Position { generation: 22, .. }) {
                saw_position = true;
            }
        }
        assert!(
            saw_position,
            "playback positions must continue after switch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn default_device_change_when_previous_device_disappears_auto_pauses() {
        let state = Arc::new(Mutex::new(FakeOutputState {
            default_device: 1,
            present_devices: vec![1, 2],
            opened_devices: Vec::new(),
            error_channel: None,
        }));
        let (player, mut events) = spawn_with_backend(
            tokio::runtime::Handle::current(),
            Box::new(FakeOutputBackend {
                state: state.clone(),
            }),
            Duration::from_millis(20),
            Duration::ZERO,
        );

        player.send(PlayerCommand::PlayBytes {
            generation: 23,
            bytes: WAV_BODY.to_vec(),
            unm_source: false,
        });
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Started { generation: 23, .. })
        })
        .await;
        player.send(PlayerCommand::Play);
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(event, PlayerEvent::Position { generation: 23, .. })
        })
        .await;

        {
            let mut guard = state.lock().unwrap();
            guard.default_device = 2;
            guard.present_devices = vec![2];
        }
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(
                event,
                PlayerEvent::Paused {
                    generation: 23,
                    paused: true
                }
            )
        })
        .await;
        assert_eq!(state.lock().unwrap().opened_devices, vec![1]);

        player.send(PlayerCommand::TogglePause);
        recv_until(&mut events, Duration::from_secs(1), |event| {
            matches!(
                event,
                PlayerEvent::Paused {
                    generation: 23,
                    paused: false
                }
            )
        })
        .await;
        assert_eq!(state.lock().unwrap().opened_devices, vec![1, 2]);
    }
}
