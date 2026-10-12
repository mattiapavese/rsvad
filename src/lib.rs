//! Streaming voice activity detection with [Silero VAD](https://github.com/snakers4/silero-vad).
//!
//! [`VadIterator`] processes a live audio stream passed in chunks of any size. Unlike the usual
//! streaming iterators, it does not report speech as soon as a window goes over the threshold:
//! speech starting after silence is held back (the *lookback*) and processed again together with
//! the audio that follows, until it is long enough to be a segment
//! ([`FindSpeechSegmentsParams::min_speech_duration_ms`]) or it ends too short and is dropped.
//!
//! With `min_speech_duration_ms` under one window (32ms) it does, as silero's `VADIterator`: speech
//! is reported in the call in which it starts, and every speech run is a segment.
//!
//! More generally, every call emits only audio whose classification can no longer change and
//! holds back the rest, so the segments are the same however the stream is split in calls.
//!
//! # Sessions
//!
//! [`initialize`] loads ONNX Runtime and the model, and starts a pool of sessions shared by all the
//! iterators. Each session runs on a thread of its own and serves up to
//! [`VadEnvConfig::max_streams_per_session`] iterators; new sessions are started in background
//! before the existing ones are full, and closed when no longer needed. [`VadIterator::vad`] is
//! async: it hands the audio to its session's thread and awaits the result, so it never blocks the
//! async runtime. [`VadIterator::vad_blocking`] is there for code that is not async.
//!
//! # Noise gate
//!
//! With [`FindSpeechSegmentsParams::noise_gate`] set, a noise gate (a port of JUCE's, the one
//! pedalboard runs, see [`NoiseGateParams`]) runs before the model on the same 32ms windows. A
//! window during which the gate stays closed is non-speech and is not run through the model, which
//! saves its inference; a call on gated audio only does not even reach the session. The model state
//! is kept as it is across gated windows. Segments are the same as without the gate when it closes
//! only in short pauses; after long gated pauses, edges can move by a window or two.
//!
//! # Caller contract
//!
//! The caller keeps the audio from [`VadIterator::offset_samples`] on and passes it again, followed
//! by the new audio, at the next call. Segments are absolute sample positions in the stream.
//!
//! ```no_run
//! use silero_vad_rs::{VadIterator, initialize};
//! # async fn chunks() -> Vec<Vec<f32>> { vec![] }
//! # async fn run() -> Result<(), silero_vad_rs::VadError> {
//! initialize("libonnxruntime.dylib", "silero_vad.onnx", None)?; // once, at startup
//!
//! // then, for each stream
//! let mut it = VadIterator::new(16_000, None)?;
//! let mut buf: Vec<f32> = Vec::new(); // audio from it.offset_samples() on
//! for chunk in chunks().await {
//!     buf.extend_from_slice(&chunk);
//!     let start = it.offset_samples();
//!     let res = it.vad(&buf).await?;
//!     for (i, s) in res.speech_segments.iter().enumerate() {
//!         let speech = &buf[(s.start - start) as usize..(s.end - start) as usize];
//!         // i == 0 && res.continues_speech -> continues the last segment of the previous call
//!         // s.end < it.offset_samples()    -> the segment is complete
//!     }
//!     buf.drain(..(it.offset_samples() - start) as usize); // keep only what was held back
//! }
//! let res = it.flush(&buf).await?; // stream over: emits what was held back, then resets
//! # Ok(())
//! # }
//! ```

mod core;
mod error;
mod gate;
mod iterator;
mod ortenv;
mod pool;
mod reset;
#[cfg(test)]
mod testutil;
mod utils;

pub use error::VadError;
pub use gate::NoiseGateParams;
pub use iterator::{FindSpeechSegmentsParams, SpeechSegment, VadIterationResult, VadIterator};
pub use ortenv::{finalize, initialize, pool_stats};
pub use pool::{PoolStats, VadEnvConfig};
