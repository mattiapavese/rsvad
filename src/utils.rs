use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Wake, Waker};

use crate::iterator::{SR8K, SR16K};

pub(crate) fn ms_to_samples(ms: f32, sr: i64) -> i64 {
    (ms as f64 * sr as f64 / 1000.0) as i64
}

pub(crate) fn get_window_sz_and_context_sz(sample_rate: i64) -> (i64, i64) {
    match sample_rate {
        SR16K => (512, 64),
        SR8K => (256, 32),
        _ => (0, 0),
    }
}

// a panic while holding one of our mutexes cannot leave the data inconsistent, so poisoning is ignored
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// Minimal executor for the blocking API: polls the future on the current thread, parking it
// while waiting. The only thing awaited inside is a reply from a session's thread.
pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::thread::park(),
        }
    }
}
