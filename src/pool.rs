use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use tokio::sync::oneshot;

use crate::error::VadError;
use crate::utils::lock;

// Sessions pool.
//
// Each session is owned by a thread of its own, which serves the inference requests of the
// streams (iterators) assigned to it in arrival order. Sessions keep no per-stream state: a
// request carries the model state and context it starts from, and the reply carries them back.
//
// The pool grows with demand: a session serves up to max_streams_per_session streams, and as soon
// as fewer than min_free_slots slots are free in all the sessions together, the next session is
// started in background, so that it is ready by the time new streams need it. A session left
// with no streams is closed, unless it is needed to keep min_free_slots slots free; the first
// one is always kept.

/// Runs the model. One engine per session, used by the session's thread only.
pub(crate) trait InferenceEngine: Send {
    /// Runs the model over every window of `audio` (a whole number of windows), starting from
    /// `state` and `context`. Returns the probability, state and context after each window.
    fn run(&mut self, sample_rate: i64, audio: &[f32], state: &[f32], context: &[f32])
    -> Result<WindowsOutput, VadError>;
}

pub(crate) type EngineFactory = Arc<dyn Fn() -> Result<Box<dyn InferenceEngine>, VadError> + Send + Sync>;

#[derive(Default)]
pub(crate) struct WindowsOutput {
    pub(crate) probs: Vec<f32>,
    pub(crate) states: Vec<Vec<f32>>,
    pub(crate) contexts: Vec<Vec<f32>>,
}

pub(crate) struct InferRequest {
    pub(crate) sample_rate: i64,
    pub(crate) audio: Vec<f32>,
    pub(crate) state: Vec<f32>,
    pub(crate) context: Vec<f32>,
    pub(crate) reply: oneshot::Sender<Result<InferReply, VadError>>,
}

pub(crate) struct InferReply {
    pub(crate) output: WindowsOutput,
    pub(crate) audio: Vec<f32>, // the request's buffer, handed back to be reused
}

/// Configuration of the sessions pool, see [`crate::initialize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadEnvConfig {
    /// Streams (iterators) served by one session. Default 16.
    pub max_streams_per_session: usize,
    /// A new session is started in background as soon as fewer than this many slots are free in
    /// all the sessions together, so that it is ready by the time new streams need it. Default 4.
    pub min_free_slots: usize,
}

impl Default for VadEnvConfig {
    fn default() -> Self {
        VadEnvConfig {
            max_streams_per_session: 16,
            min_free_slots: 4,
        }
    }
}

/// A snapshot of the sessions pool, see [`crate::pool_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// Sessions open, including the ones still starting.
    pub sessions: usize,
    /// Sessions ready to serve requests.
    pub ready_sessions: usize,
    /// Streams (iterators) assigned to the sessions.
    pub streams: usize,
}

#[derive(Default)]
struct SessionStatus {
    ready: AtomicBool,
    failed: AtomicBool,
}

pub(crate) struct SessionEntry {
    tx: mpsc::Sender<InferRequest>,
    status: Arc<SessionStatus>,
}

impl SessionEntry {
    fn ready(&self) -> bool {
        self.status.ready.load(Ordering::Acquire)
    }
    fn failed(&self) -> bool {
        self.status.failed.load(Ordering::Acquire)
    }
}

struct SessionInfo {
    entry: Arc<SessionEntry>,
    streams: usize,
}

struct PoolInner {
    sessions: Vec<SessionInfo>,
    next_id: usize,
}

impl PoolInner {
    fn free_slots(&self, max: usize) -> usize {
        self.sessions
            .iter()
            .filter(|s| !s.entry.failed())
            .map(|s| max.saturating_sub(s.streams))
            .sum()
    }

    // the session a new stream joins: the least loaded ready one with a free slot,
    // otherwise the least loaded one still starting
    fn pick(&self, max: usize) -> Option<usize> {
        self.sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| s.streams < max && !s.entry.failed())
            .min_by_key(|(_, s)| (!s.entry.ready(), s.streams))
            .map(|(i, _)| i)
    }

    fn least_loaded_healthy(&self) -> Option<usize> {
        self.sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.entry.failed())
            .min_by_key(|(_, s)| s.streams)
            .map(|(i, _)| i)
    }
}

pub(crate) struct SessionPool {
    config: VadEnvConfig,
    factory: EngineFactory,
    inner: Mutex<PoolInner>,
}

impl SessionPool {
    /// Creates the pool and its first session, which is built here so that a broken setup
    /// (wrong library, wrong model) fails right away rather than at the first stream.
    pub(crate) fn new(config: VadEnvConfig, factory: EngineFactory) -> Result<Arc<Self>, VadError> {
        if config.max_streams_per_session == 0 {
            return Err(VadError::InvalidParams("max_streams_per_session must be at least 1".into()));
        }

        let first = factory()?;

        let pool = Arc::new(SessionPool {
            config,
            factory,
            inner: Mutex::new(PoolInner {
                sessions: Vec::new(),
                next_id: 0,
            }),
        });

        {
            let mut inner = lock(&pool.inner);
            pool.spawn_session(&mut inner, Some(first));
            if inner.sessions[0].entry.failed() {
                return Err(VadError::SessionFailed("could not start the session thread".into()));
            }
            pool.ensure_free_slots(&mut inner);
        }

        Ok(pool)
    }

    pub(crate) fn acquire(self: &Arc<Self>) -> SessionSlot {
        SessionSlot {
            entry: self.acquire_entry(),
            pool: Arc::clone(self),
        }
    }

    pub(crate) fn stats(&self) -> PoolStats {
        let inner = lock(&self.inner);
        let healthy = || inner.sessions.iter().filter(|s| !s.entry.failed());
        PoolStats {
            sessions: healthy().count(),
            ready_sessions: healthy().filter(|s| s.entry.ready()).count(),
            streams: inner.sessions.iter().map(|s| s.streams).sum(),
        }
    }

    fn acquire_entry(&self) -> Arc<SessionEntry> {
        let max = self.config.max_streams_per_session;
        let mut inner = lock(&self.inner);

        inner.sessions.retain(|s| !(s.entry.failed() && s.streams == 0));

        let idx = match inner.pick(max) {
            Some(idx) => idx,
            None => {
                // every session is full: this happens only if streams arrive faster than
                // sessions start, as one is started in advance when they are getting full
                self.spawn_session(&mut inner, None);
                // if the new session could not even start its thread, over-subscribe a healthy one
                inner
                    .pick(max)
                    .or_else(|| inner.least_loaded_healthy())
                    .unwrap_or(inner.sessions.len() - 1)
            }
        };

        inner.sessions[idx].streams += 1;
        let entry = Arc::clone(&inner.sessions[idx].entry);

        self.ensure_free_slots(&mut inner);

        entry
    }

    fn release(&self, entry: &Arc<SessionEntry>) {
        let max = self.config.max_streams_per_session;
        let mut inner = lock(&self.inner);

        if let Some(s) = inner.sessions.iter_mut().find(|s| Arc::ptr_eq(&s.entry, entry)) {
            s.streams -= 1;
        }

        // close the empty sessions that are not needed to keep min_free_slots slots free,
        // always keeping one
        while let Some(i) = inner.sessions.iter().rposition(|s| s.streams == 0) {
            let healthy = inner.sessions.iter().filter(|s| !s.entry.failed()).count();
            let keep = !inner.sessions[i].entry.failed()
                && (healthy <= 1 || inner.free_slots(max) < self.config.min_free_slots + max);
            if keep {
                break;
            }
            // dropping the entry closes the channel once no stream holds it: the thread exits
            inner.sessions.remove(i);
        }
    }

    // the sessions are getting full: start the next one now, so that it is ready when needed
    fn ensure_free_slots(&self, inner: &mut PoolInner) {
        while inner.free_slots(self.config.max_streams_per_session) < self.config.min_free_slots {
            self.spawn_session(inner, None);
            if inner.sessions.last().is_some_and(|s| s.entry.failed()) {
                break; // cannot start threads right now, try again at the next stream
            }
        }
    }

    fn spawn_session(&self, inner: &mut PoolInner, engine: Option<Box<dyn InferenceEngine>>) {
        let (tx, rx) = mpsc::channel();
        let status = Arc::new(SessionStatus::default());

        let id = inner.next_id;
        inner.next_id += 1;

        let factory = Arc::clone(&self.factory);
        let worker_status = Arc::clone(&status);
        let spawned = std::thread::Builder::new()
            .name(format!("silero-vad-session-{id}"))
            .spawn(move || session_worker(engine, factory, rx, worker_status));
        if spawned.is_err() {
            status.failed.store(true, Ordering::Release);
        }

        inner.sessions.push(SessionInfo {
            entry: Arc::new(SessionEntry { tx, status }),
            streams: 0,
        });
    }
}

// The session's thread: builds the engine (unless the pool built it already), then serves the
// requests in arrival order until every stream assigned to the session is gone.
fn session_worker(
    engine: Option<Box<dyn InferenceEngine>>,
    factory: EngineFactory,
    rx: mpsc::Receiver<InferRequest>,
    status: Arc<SessionStatus>,
) {
    // a panic leaves the session failed, so the pool stops assigning streams to it
    struct FailOnPanic(Arc<SessionStatus>);
    impl Drop for FailOnPanic {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0.failed.store(true, Ordering::Release);
            }
        }
    }
    let _guard = FailOnPanic(Arc::clone(&status));

    let mut engine = match engine.map_or_else(|| factory(), Ok) {
        Ok(engine) => {
            status.ready.store(true, Ordering::Release);
            engine
        }
        Err(e) => {
            status.failed.store(true, Ordering::Release);
            let msg = e.to_string();
            for req in rx {
                let _ = req.reply.send(Err(VadError::SessionFailed(msg.clone())));
            }
            return;
        }
    };

    for req in rx {
        let InferRequest {
            sample_rate,
            audio,
            state,
            context,
            reply,
        } = req;
        let result = engine
            .run(sample_rate, &audio, &state, &context)
            .map(|output| InferReply { output, audio });
        // an error here means the caller stopped waiting (its future was dropped): nothing to do
        let _ = reply.send(result);
    }
}

/// A stream's place in a session. Dropping it frees the slot.
pub(crate) struct SessionSlot {
    pool: Arc<SessionPool>,
    entry: Arc<SessionEntry>,
}

impl SessionSlot {
    pub(crate) fn send(&self, req: InferRequest) -> Result<(), VadError> {
        self.entry.tx.send(req).map_err(|_| VadError::SessionClosed)
    }

    /// Moves the stream from a session that failed to another one. Sessions keep no per-stream
    /// state, so the stream can go on transparently.
    pub(crate) fn reassign(&mut self) {
        self.entry.status.failed.store(true, Ordering::Release);
        let entry = self.pool.acquire_entry();
        let old = std::mem::replace(&mut self.entry, entry);
        self.pool.release(&old);
    }
}

impl Drop for SessionSlot {
    fn drop(&mut self) {
        self.pool.release(&self.entry);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Condvar;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::iterator::{FindSpeechSegmentsParams, VadIterator};
    use crate::testutil::{FakeEngine, Rng, fake_factory, gen_probs, offline_ref, probs_to_audio, stream_async};

    fn session_index(pool: &SessionPool, it: &VadIterator) -> usize {
        let entry = &it.core.slot.as_ref().unwrap().entry;
        lock(&pool.inner).sessions.iter().position(|s| Arc::ptr_eq(&s.entry, entry)).unwrap()
    }

    fn wait_ready(pool: &SessionPool, n: usize) {
        let t0 = Instant::now();
        while pool.stats().ready_sessions < n {
            assert!(t0.elapsed() < Duration::from_secs(5), "sessions not ready");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn new_it(pool: &Arc<SessionPool>) -> VadIterator {
        VadIterator::on_pool(16_000, None, pool).unwrap()
    }

    #[test]
    fn grows_in_advance_and_shrinks() {
        let config = VadEnvConfig { max_streams_per_session: 4, min_free_slots: 1 };
        let pool = SessionPool::new(config, fake_factory(None)).unwrap();
        assert_eq!(pool.stats().sessions, 1);
        wait_ready(&pool, 1);

        let mut its: Vec<VadIterator> = (0..3).map(|_| new_it(&pool)).collect();
        assert_eq!(pool.stats(), PoolStats { sessions: 1, ready_sessions: 1, streams: 3 });

        // the 4th stream fills the first session: the second one starts right away, before it is needed
        its.push(new_it(&pool));
        assert_eq!(pool.stats().sessions, 2);
        wait_ready(&pool, 2);
        its.push(new_it(&pool));
        assert_eq!(session_index(&pool, &its[4]), 1);

        // the second session, now empty, is kept: without it there would be no free slot
        its.pop();
        assert_eq!(pool.stats().sessions, 2);
        // with a free slot in the first one, it is closed
        its.remove(0);
        assert_eq!(pool.stats().sessions, 1);

        its.clear();
        assert_eq!(pool.stats(), PoolStats { sessions: 1, ready_sessions: 1, streams: 0 });
    }

    #[test]
    fn ready_sessions_are_preferred_to_starting_ones() {
        // every build after the first waits for the gate
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let builds = Arc::new(AtomicUsize::new(0));
        let factory: EngineFactory = {
            let (gate, builds) = (Arc::clone(&gate), Arc::clone(&builds));
            Arc::new(move || {
                if builds.fetch_add(1, Ordering::SeqCst) > 0 {
                    let (open, cv) = &*gate;
                    let _open = cv.wait_while(lock(open), |open| !*open).unwrap();
                }
                Ok(Box::new(FakeEngine::default()) as Box<dyn InferenceEngine>)
            })
        };
        let config = VadEnvConfig { max_streams_per_session: 4, min_free_slots: 2 };
        let pool = SessionPool::new(config, factory).unwrap();
        wait_ready(&pool, 1);

        let mut its: Vec<VadIterator> = (0..3).map(|_| new_it(&pool)).collect();
        assert_eq!(pool.stats(), PoolStats { sessions: 2, ready_sessions: 1, streams: 3 });

        // the starting session has fewer streams, but the ready one still has a slot
        its.push(new_it(&pool));
        assert_eq!(session_index(&pool, &its[3]), 0);
        // now the ready one is full: the stream waits for the starting one
        its.push(new_it(&pool));
        assert_eq!(session_index(&pool, &its[4]), 1);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let audio = probs_to_audio(&[0.9; 20]);
        let mut it = its.pop().unwrap();
        let task = rt.spawn(async move { it.vad(&audio).await.map(|_| ()) });
        std::thread::sleep(Duration::from_millis(50));
        assert!(!task.is_finished());

        *lock(&gate.0) = true;
        gate.1.notify_all();
        rt.block_on(task).unwrap().unwrap();
    }

    #[test]
    fn stream_moves_away_from_a_failed_session() {
        // the second build fails (after a while, so that the pool still sees it as starting)
        let builds = Arc::new(AtomicUsize::new(0));
        let factory: EngineFactory = {
            let builds = Arc::clone(&builds);
            Arc::new(move || match builds.fetch_add(1, Ordering::SeqCst) {
                1 => {
                    std::thread::sleep(Duration::from_millis(50));
                    Err(VadError::SessionFailed("boom".into()))
                }
                _ => Ok(Box::new(FakeEngine::default()) as Box<dyn InferenceEngine>),
            })
        };
        let config = VadEnvConfig { max_streams_per_session: 2, min_free_slots: 0 };
        let pool = SessionPool::new(config, factory).unwrap();

        let _its: Vec<VadIterator> = (0..2).map(|_| new_it(&pool)).collect();
        let mut it = new_it(&pool); // lands on the second session, whose build fails
        assert_eq!(session_index(&pool, &it), 1);

        let audio = probs_to_audio(&[0.9; 20]);
        it.vad_blocking(&audio).unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 3);
        assert_eq!(pool.stats(), PoolStats { sessions: 2, ready_sessions: 2, streams: 3 });
    }

    // many streams at once through the pool, with chunks of any size (also longer than one
    // request): each must give exactly what the offline reference gives for its audio. The fake
    // engine also checks that every request starts from the right model state
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_streams() {
        let config = VadEnvConfig { max_streams_per_session: 4, min_free_slots: 2 };
        let pool = SessionPool::new(config, fake_factory(Some(Duration::from_micros(200)))).unwrap();
        let params = FindSpeechSegmentsParams { pad_speech_ms: 64.0, ..Default::default() };

        let its: Vec<VadIterator> = (0..24).map(|_| VadIterator::on_pool(16_000, Some(params), &pool).unwrap()).collect();
        let peak = pool.stats();

        let mut tasks = Vec::new();
        for (k, mut it) in its.into_iter().enumerate() {
            tasks.push(tokio::spawn(async move {
                let mut rng = Rng(k as u64);
                let probs = gen_probs(&mut rng, 600, 0.15);
                let reference = offline_ref(&probs, it.params());
                let (got, _) = stream_async(&mut it, &probs_to_audio(&probs), || 1 + rng.below(80 * 512), false).await;
                assert_eq!(got, reference, "stream {k}");
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert!(peak.sessions >= 6, "{peak:?}");
        assert_eq!(pool.stats().streams, 0);
    }

    // dropping vad() futures while their request is in flight leaves the iterator as it was
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_calls_leave_the_iterator_consistent() {
        let pool = SessionPool::new(VadEnvConfig::default(), fake_factory(Some(Duration::from_millis(1)))).unwrap();
        let mut rng = Rng(7);
        let probs = gen_probs(&mut rng, 400, 0.15);
        let mut it = VadIterator::on_pool(16_000, None, &pool).unwrap();
        let reference = offline_ref(&probs, it.params());
        let (got, _) = stream_async(&mut it, &probs_to_audio(&probs), || 1 + rng.below(20 * 512), true).await;
        assert_eq!(got, reference);
    }

    #[test]
    fn vad_future_is_send() {
        fn assert_send<T: Send>(_: &T) {}
        let mut it = VadIterator::detached(16_000, None).unwrap();
        let f = it.vad(&[]);
        assert_send(&f);
    }
}
