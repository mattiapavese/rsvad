use std::borrow::Cow;
use std::sync::Arc;

use crate::core::{InferenceOutput, VadCore, pad_and_merge};
use crate::error::VadError;
use crate::gate::{GateState, NoiseGate, NoiseGateParams};
use crate::ortenv::{CHUNK_LEN_MS, shared_pool};
use crate::pool::SessionPool;
use crate::utils::{block_on, get_window_sz_and_context_sz, ms_to_samples};

pub(crate) const SR8K: i64 = 8_000;
pub(crate) const SR16K: i64 = 16_000;

/// Streaming VAD over a live audio stream. See the crate docs for the caller contract.
///
/// Each iterator takes a slot in a session of the pool started by [`crate::initialize`], and
/// frees it when dropped.
pub struct VadIterator {
    pub(crate) core: VadCore,

    pub(crate) params: FindSpeechSegmentsParams,

    pub(crate) decision: DecisionState,
    pub(crate) offset_samples: i64,
    pub(crate) emitted_until: Option<i64>, // end of the last emitted segment
}

/// State of the speech/silence state machine at a window boundary. One snapshot per window
/// is kept during a call, so that the iterator can rewind to any of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DecisionState {
    pub(crate) triggered: bool,
    pub(crate) start: i64,            // start of the current raw segment, meaningful only if triggered
    pub(crate) temp_end: Option<i64>, // start of the current silence run inside a segment
    pub(crate) prev_end: Option<i64>, // end of the last kept raw segment
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FindSpeechSegmentsParams {
    /// A window with probability >= threshold is speech. <= 0 means 0.5, minimum 0.05.
    pub threshold: f32,
    /// Inside a segment, a window with probability < neg_threshold starts a silence run.
    /// <= 0 means max(0.01, threshold - 0.15). Must not be greater than threshold.
    pub neg_threshold: f32,
    /// Speech shorter than this is not a segment. <= 0 means 250.
    ///
    /// Under one window (32ms) the lookback is off: as speech lasts at least one window, it is never
    /// held back for being short. It is emitted in the call in which it starts, as silero's
    /// VADIterator does, and every speech run is a segment. What is undecided is still held back:
    /// while not in speech the last `pad_speech_ms` (the leading pad of speech that may start next),
    /// in a silence run the audio after its trailing pad, until the run is `min_silence_duration_ms`
    /// long or turns out to be a pause.
    pub min_speech_duration_ms: f32,
    /// A silence run this long ends a segment. <= 0 means 250.
    pub min_silence_duration_ms: f32,
    /// Padding added on both sides of each segment, rounded up to a multiple of 32ms
    /// (one window). 0 means no padding, < 0 means 64.
    pub pad_speech_ms: f32,
    /// Noise gate run before the model. `None` (the default) runs the model on every window.
    ///
    /// A window during which the gate stays closed is non-speech (probability 0) and is not run
    /// through the model. The model state is kept as it was before the gated windows, so the model
    /// resumes from a state close to the one it would have had without the gate. Close, not equal:
    /// after a long gated pause, segment edges can move by a window or two compared with no gate.
    pub noise_gate: Option<NoiseGateParams>,
}

impl Default for FindSpeechSegmentsParams {
    fn default() -> Self {
        FindSpeechSegmentsParams {
            threshold: 0.5,
            neg_threshold: 0.0,
            min_speech_duration_ms: 250.0,
            min_silence_duration_ms: 250.0,
            pad_speech_ms: 0.0,
            noise_gate: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VadIterationResult {
    /// Final speech segments in [old offset_samples(), new offset_samples()), padded and merged,
    /// in absolute sample positions. A segment with `end < offset_samples()` is complete; one with
    /// `end == offset_samples()` may continue in the next call.
    pub speech_segments: Vec<SpeechSegment>,
    /// `Some(offset_samples())` if part of the audio passed must be passed again in the next call.
    pub look_back_from: Option<i64>,
    /// `speech_segments[0]` continues the last segment emitted by the previous call.
    pub continues_speech: bool,
    /// How many of the windows that became final in this call the noise gate classified as
    /// non-speech without running the model (0 without a gate). Added up over a stream, it is the
    /// same however the stream is split in calls.
    pub gated_windows: usize,
}

impl VadIterator {
    /// Creates an iterator, taking a slot in a session of the pool started by [`crate::initialize`].
    /// It does not block: if a new session is needed, it is started in background and the first
    /// call to [`vad`](Self::vad) waits for it. `None` params means the defaults of [`FindSpeechSegmentsParams`].
    pub fn new(sample_rate: i64, params: Option<FindSpeechSegmentsParams>) -> Result<Self, VadError> {
        let pool = shared_pool().ok_or(VadError::NotInitialized)?;
        Self::on_pool(sample_rate, params, &pool)
    }

    pub(crate) fn on_pool(
        sample_rate: i64,
        params: Option<FindSpeechSegmentsParams>,
        pool: &Arc<SessionPool>,
    ) -> Result<Self, VadError> {
        let mut it = Self::detached(sample_rate, params)?;
        it.core.slot = Some(pool.acquire());
        Ok(it)
    }

    // An iterator without session: find_speech_segments works, inference does not.
    pub(crate) fn detached(sample_rate: i64, params: Option<FindSpeechSegmentsParams>) -> Result<Self, VadError> {
        if !(sample_rate == SR8K || sample_rate == SR16K) {
            return Err(VadError::UnsupportedSampleRate(sample_rate));
        }

        // params are taken by value: the caller's struct is never modified
        let mut p = params.unwrap_or_default();

        if p.min_speech_duration_ms <= 0.0 {
            p.min_speech_duration_ms = 250.0;
        }
        if p.min_silence_duration_ms <= 0.0 {
            p.min_silence_duration_ms = 250.0;
        }
        if p.threshold <= 0.0 {
            p.threshold = 0.5;
        }
        p.threshold = p.threshold.max(0.05);

        if p.neg_threshold <= 0.0 {
            p.neg_threshold = (p.threshold - 0.15).max(0.01);
        }
        if p.neg_threshold > p.threshold {
            return Err(VadError::InvalidParams(format!(
                "neg_threshold ({}) must not be greater than threshold ({})",
                p.neg_threshold, p.threshold
            )));
        }

        // if 0 leave it 0, if > 0 round it up to a multiple of the window length,
        // so that the pad is a whole number of windows and lookback can rewind to it
        if p.pad_speech_ms > 0.0 {
            p.pad_speech_ms = (p.pad_speech_ms / CHUNK_LEN_MS as f32).ceil() * CHUNK_LEN_MS as f32;
        } else if p.pad_speech_ms < 0.0 {
            p.pad_speech_ms = 64.0;
        }

        let gate = match &p.noise_gate {
            Some(g) => {
                // written so that NaN fails too
                if !(g.attack_ms >= 0.0 && g.release_ms >= 0.0) || g.threshold_db.is_nan() {
                    return Err(VadError::InvalidParams(format!(
                        "noise_gate: attack_ms ({}) and release_ms ({}) must not be negative, threshold_db ({}) must be a number",
                        g.attack_ms, g.release_ms, g.threshold_db
                    )));
                }
                Some(NoiseGate::new(g, sample_rate))
            }
            None => None,
        };

        Ok(VadIterator {
            core: VadCore::new(sample_rate, gate, None),
            params: p,
            decision: DecisionState::default(),
            offset_samples: 0,
            emitted_until: None,
        })
    }

    /// Position (absolute, in samples) where the audio passed to the next call to [`vad`](Self::vad)
    /// must start. Everything before it is final; everything from it on must be passed again.
    pub fn offset_samples(&self) -> i64 {
        self.offset_samples
    }

    /// True if a speech segment is open at [`offset_samples`](Self::offset_samples).
    pub fn triggered(&self) -> bool {
        self.decision.triggered
    }

    /// The params in use, after defaults and rounding have been applied.
    pub fn params(&self) -> &FindSpeechSegmentsParams {
        &self.params
    }

    pub fn sample_rate(&self) -> i64 {
        self.core.sample_rate
    }

    /// Processes `audio_mono_pcm_f32`, which must start at [`offset_samples`](Self::offset_samples):
    /// the audio held back by the previous call (if any) followed by the new audio.
    ///
    /// Returns the speech segments that became final in [old `offset_samples()`, new `offset_samples()`).
    /// The audio from the new `offset_samples()` on is held back, as its classification depends on audio
    /// that has not arrived yet, and must be passed again at the beginning of the next call.
    ///
    /// Inference runs on the session's thread, this only awaits it. Cancel safe: if the future is
    /// dropped before completing, the iterator is left as it was, and the same audio can be passed again.
    pub async fn vad(&mut self, audio_mono_pcm_f32: &[f32]) -> Result<VadIterationResult, VadError> {
        let output = self.core.inference_loop(audio_mono_pcm_f32).await?;
        Ok(self.after_inference(audio_mono_pcm_f32.len(), output, false))
    }

    /// Processes the last audio of a stream that has ended (from [`offset_samples`](Self::offset_samples)
    /// to the end of the stream) and emits everything the iterator was still holding back.
    /// The iterator is reset afterwards and can be reused for a new stream.
    pub async fn flush(&mut self, audio_mono_pcm_f32: &[f32]) -> Result<VadIterationResult, VadError> {
        let (window_size_samples, _) = get_window_sz_and_context_sz(self.core.sample_rate);

        // a trailing partial window is not processed by vad (it is held back and passed
        // again in the next call), flush zero-pads it as the stream is over
        let mut input_buffer = Cow::Borrowed(audio_mono_pcm_f32);
        let rem = audio_mono_pcm_f32.len() % window_size_samples as usize;
        if rem != 0 {
            input_buffer
                .to_mut()
                .resize(audio_mono_pcm_f32.len() + window_size_samples as usize - rem, 0.0);
        }

        let result = match self.core.inference_loop(&input_buffer).await {
            Ok(output) => Ok(self.after_inference(audio_mono_pcm_f32.len(), output, true)),
            Err(e) => Err(e),
        };
        self.reset();
        result
    }

    /// Blocking version of [`vad`](Self::vad), for code that is not async. It blocks the calling
    /// thread while waiting for the session: do not call it from async code.
    pub fn vad_blocking(&mut self, audio_mono_pcm_f32: &[f32]) -> Result<VadIterationResult, VadError> {
        block_on(self.vad(audio_mono_pcm_f32))
    }

    /// Blocking version of [`flush`](Self::flush), see [`vad_blocking`](Self::vad_blocking).
    pub fn flush_blocking(&mut self, audio_mono_pcm_f32: &[f32]) -> Result<VadIterationResult, VadError> {
        block_on(self.flush(audio_mono_pcm_f32))
    }

    fn after_inference(&mut self, audio_len: usize, output: InferenceOutput, flush: bool) -> VadIterationResult {
        let (probs, states, contexts, gate_states, gated) = output;
        let init_offset_samples = self.offset_samples;
        let audio_end = self.offset_samples + audio_len as i64;
        let stream_end = flush.then_some(audio_end);

        let mut result = self.find_speech_segments(&probs, states, contexts, gate_states, stream_end);

        // the windows now final: the ones before the new offset_samples, all of them if the stream is over
        let (window_size_samples, _) = get_window_sz_and_context_sz(self.core.sample_rate);
        let final_windows = if flush {
            gated.len()
        } else {
            ((self.offset_samples - init_offset_samples) / window_size_samples) as usize
        };
        result.gated_windows = gated[..final_windows].iter().filter(|&&g| g).count();

        if !flush && self.offset_samples < audio_end {
            result.look_back_from = Some(self.offset_samples);
        }

        result
    }

    /// Runs the speech/silence state machine over `probs`, emits what became final and rewinds
    /// the iterator (model state, context, gate state and decision state) to the first position
    /// that is not final yet. That position is:
    ///
    /// - not triggered: the last `pad_samples`, which can still become the leading pad of a
    ///   segment starting in the next call;
    /// - triggered, speech still shorter than `min_speech_duration_ms` (LOOKBACK): the start of the
    ///   segment minus its leading pad, as whether it is a segment at all depends on what comes next.
    ///   Never with min speech under one window, as speech lasts at least one;
    /// - triggered, speech long enough, in a silence run: the end of the speech plus the trailing
    ///   pad, as the silence run can still end the segment or turn out to be a pause;
    /// - triggered, speech long enough, no silence run: the end of the audio processed.
    ///
    /// The emitted segments are therefore the same for any way the stream is split in calls.
    /// `stream_end` set means the stream is over, and everything is decided and emitted up to it.
    pub(crate) fn find_speech_segments(
        &mut self,
        probs: &[f32],
        mut states: Vec<Vec<f32>>,
        mut contexts: Vec<Vec<f32>>,
        gate_states: Vec<GateState>,
        stream_end: Option<i64>,
    ) -> VadIterationResult {
        // TODO i could compute all of this once at initialization
        let sample_rate = self.core.sample_rate;
        let (window_size_samples, _) = get_window_sz_and_context_sz(sample_rate);
        let min_silence_samples = ms_to_samples(self.params.min_silence_duration_ms, sample_rate);
        let min_speech_samples = ms_to_samples(self.params.min_speech_duration_ms, sample_rate);
        let pad_samples = self.params.pad_speech_ms as i64 / CHUNK_LEN_MS * window_size_samples;
        //

        let init_offset_samples = self.offset_samples;
        let tot_samples = window_size_samples * probs.len() as i64 + init_offset_samples;

        let mut result = VadIterationResult::default();

        let mut d = self.decision;

        // decisions[k] is the decision state before window k, like states[k], contexts[k] and gate_states[k]
        let mut decisions = Vec::with_capacity(probs.len() + 1);
        decisions.push(d);

        let mut unmerged_segments = Vec::with_capacity(4); // just preallocate a bit of capacity

        // the trailing pad of a segment that ended before this call may still reach past init_offset_samples
        if let Some(prev_end) = d.prev_end
            && prev_end + pad_samples > init_offset_samples
        {
            unmerged_segments.push(SpeechSegment { start: prev_end, end: prev_end });
        }

        for (i, &p) in probs.iter().enumerate() {
            let cur_sample = window_size_samples * i as i64 + init_offset_samples;

            if p >= self.params.threshold {
                d.temp_end = None;
                if !d.triggered {
                    d.triggered = true;
                    d.start = cur_sample;
                }
            } else if d.triggered {
                if p < self.params.neg_threshold && d.temp_end.is_none() {
                    d.temp_end = Some(cur_sample);
                }

                // unlike silero's reference impl, once a silence run has started also a window in
                // [neg_threshold, threshold) can close the segment: otherwise a long run of such windows
                // would keep the segment undecided, and the audio held back would grow unbounded
                if let Some(temp_end) = d.temp_end
                    && cur_sample - temp_end >= min_silence_samples
                {
                    if temp_end - d.start > min_speech_samples {
                        unmerged_segments.push(SpeechSegment { start: d.start, end: temp_end });
                        d.prev_end = Some(temp_end);
                    }
                    d.triggered = false;
                    d.start = 0;
                    d.temp_end = None;
                }
            }

            decisions.push(d);
        }

        // end of the speech of the open segment, as far as it is known now
        let speech_end = d.temp_end.unwrap_or(tot_samples);

        let final_until = if let Some(stream_end) = stream_end {
            // nothing comes next: decide about the open segment now
            let speech_end = speech_end.min(stream_end);
            if d.triggered && speech_end - d.start > min_speech_samples {
                unmerged_segments.push(SpeechSegment { start: d.start, end: speech_end });
            }
            stream_end
        } else if !d.triggered {
            tot_samples - pad_samples
        } else if speech_end - d.start <= min_speech_samples {
            // LOOKBACK: too little speech to tell whether this is a segment. Hold it back, together
            // with its leading pad, and decide when the next call brings what comes after it
            d.start - pad_samples
        } else if let Some(temp_end) = d.temp_end {
            // a segment whatever comes next, but in a silence run that may still end it
            unmerged_segments.push(SpeechSegment { start: d.start, end: temp_end });
            temp_end + pad_samples
        } else {
            // a segment whatever comes next, speech up to the end of the audio processed
            unmerged_segments.push(SpeechSegment { start: d.start, end: tot_samples });
            tot_samples
        };

        let final_until = if stream_end.is_none() {
            let final_until = final_until.max(init_offset_samples).min(tot_samples);

            let kept_idx = ((final_until - init_offset_samples) / window_size_samples) as usize;
            self.core.state = std::mem::take(&mut states[kept_idx]);
            self.core.context = std::mem::take(&mut contexts[kept_idx]);
            self.core.gate_state = gate_states[kept_idx];
            self.decision = decisions[kept_idx];

            final_until
        } else {
            final_until
        };

        self.offset_samples = final_until;

        result.speech_segments = pad_and_merge(&unmerged_segments, pad_samples, init_offset_samples, final_until);

        if let (Some(first), Some(last)) = (result.speech_segments.first(), result.speech_segments.last()) {
            result.continues_speech =
                first.start == init_offset_samples && self.emitted_until == Some(init_offset_samples);
            self.emitted_until = Some(last.end);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testutil::{Rng, WS, gen_probs, offline_ref, runs};

    fn params(pad: f32, min_sil: f32, min_sp: f32) -> FindSpeechSegmentsParams {
        FindSpeechSegmentsParams {
            threshold: 0.5,
            neg_threshold: 0.0,
            min_speech_duration_ms: min_sp,
            min_silence_duration_ms: min_sil,
            pad_speech_ms: pad,
            noise_gate: None,
        }
    }

    // min speech under one window: the lookback is off
    fn no_look_back(pad: f32, min_sil: f32) -> FindSpeechSegmentsParams {
        params(pad, min_sil, 1.0)
    }

    fn dummy(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<GateState>) {
        (vec![Vec::new(); n + 1], vec![Vec::new(); n + 1], vec![GateState::default(); n + 1])
    }

    #[derive(Default)]
    struct StreamOut {
        merged: Vec<SpeechSegment>,
        violations: Vec<String>,
        max_held_wins: usize,
    }

    // Caller following the protocol: keeps the audio from offset_samples() on, appends the new
    // chunk, calls the iterator, finally flushes what is left. Probabilities stand for audio.
    fn run_stream(probs: &[f32], mut chunk: impl FnMut() -> usize, params: FindSpeechSegmentsParams) -> StreamOut {
        let mut it = VadIterator::detached(16_000, Some(params)).unwrap();
        let mut out = StreamOut::default();
        let mut emitted_end = None;

        let mut collect = |out: &mut StreamOut, r: &VadIterationResult, lo: i64, hi: i64| {
            for (k, s) in r.speech_segments.iter().enumerate() {
                if s.end <= s.start || s.start < lo || s.end > hi {
                    out.violations.push(format!("bad segment {s:?} in [{lo},{hi})"));
                }
                let touches = emitted_end == Some(s.start);
                if k == 0 && r.continues_speech != touches {
                    out.violations.push(format!("continues_speech={} but touches={touches} at {s:?}", r.continues_speech));
                }
                if emitted_end.is_some_and(|e| s.start < e) {
                    out.violations.push(format!("overlap {s:?} < {emitted_end:?}"));
                }
                match out.merged.last_mut() {
                    Some(last) if touches => last.end = s.end,
                    _ => out.merged.push(*s),
                }
                emitted_end = Some(s.end);
            }
        };

        let mut avail = 0;
        while avail < probs.len() {
            avail = (avail + chunk()).min(probs.len());
            let lo = it.offset_samples();
            let window = &probs[(lo / WS) as usize..avail];
            let (st, ct, gt) = dummy(window.len());
            let mut r = it.find_speech_segments(window, st, ct, gt, None);
            if lo + window.len() as i64 * WS > it.offset_samples() {
                r.look_back_from = Some(it.offset_samples());
            }
            if it.offset_samples() % WS != 0 {
                out.violations.push(format!("offset {} not on a window boundary", it.offset_samples()));
            }
            collect(&mut out, &r, lo, it.offset_samples());
            out.max_held_wins = out.max_held_wins.max(avail - (it.offset_samples() / WS) as usize);
        }
        let lo = it.offset_samples();
        let window = &probs[(lo / WS) as usize..];
        let (st, ct, gt) = dummy(window.len());
        let end = probs.len() as i64 * WS;
        let r = it.find_speech_segments(window, st, ct, gt, Some(end));
        collect(&mut out, &r, lo, end);
        out
    }

    #[test]
    fn params_normalization() {
        for (input, want) in [(-1.0, 64.0), (0.0, 0.0), (1.0, 32.0), (30.0, 32.0), (32.0, 32.0), (33.0, 64.0), (64.0, 64.0), (96.0, 96.0)] {
            let it = VadIterator::detached(16_000, Some(FindSpeechSegmentsParams { pad_speech_ms: input, ..Default::default() })).unwrap();
            assert_eq!(it.params().pad_speech_ms, want, "pad_speech_ms {input}");
        }

        let it = VadIterator::detached(8_000, None).unwrap();
        let p = it.params();
        assert_eq!((p.threshold, p.min_speech_duration_ms, p.min_silence_duration_ms, p.pad_speech_ms), (0.5, 250.0, 250.0, 0.0));
        assert!((p.neg_threshold - 0.35).abs() < 1e-6);

        let bad = FindSpeechSegmentsParams { threshold: 0.5, neg_threshold: 0.6, ..Default::default() };
        assert!(matches!(VadIterator::detached(16_000, Some(bad)), Err(VadError::InvalidParams(_))));
        for gate in [
            NoiseGateParams { attack_ms: -1.0, ..Default::default() },
            NoiseGateParams { release_ms: f32::NAN, ..Default::default() },
            NoiseGateParams { threshold_db: f32::NAN, ..Default::default() },
        ] {
            let bad = FindSpeechSegmentsParams { noise_gate: Some(gate), ..Default::default() };
            assert!(matches!(VadIterator::detached(16_000, Some(bad)), Err(VadError::InvalidParams(_))), "{gate:?}");
        }
        // 0 is an instant attack or release, as in JUCE
        let instant = NoiseGateParams { attack_ms: 0.0, release_ms: 0.0, ..Default::default() };
        assert!(VadIterator::detached(16_000, Some(FindSpeechSegmentsParams { noise_gate: Some(instant), ..Default::default() })).is_ok());
        assert!(matches!(VadIterator::detached(44_100, None), Err(VadError::UnsupportedSampleRate(44_100))));
    }

    #[test]
    fn new_requires_initialize() {
        assert!(matches!(VadIterator::new(16_000, None), Err(VadError::NotInitialized)));
    }

    #[test]
    fn streaming_equals_offline_for_any_chunking() {
        let mut rng = Rng(1);
        let (mut total, mut failures) = (0, 0);
        let mut worst_held = [0, 0]; // lookback on, off (min speech 30, under one window)
        for pad in [0.0, 30.0, 64.0, 96.0, 320.0] {
            for min_sil in [50.0, 100.0, 250.0, 500.0] {
                for min_sp in [30.0, 100.0, 250.0, 600.0] {
                    for hyst in [0.0, 0.15, 0.6] {
                        let p = params(pad, min_sil, min_sp);
                        let normalized = *VadIterator::detached(16_000, Some(p)).unwrap().params();
                        let probs = gen_probs(&mut rng, 1500, hyst);
                        let reference = offline_ref(&probs, &normalized);

                        for chunking in ["all", "1", "2", "3", "5", "8", "13", "31", "64", "rand"] {
                            total += 1;
                            let out = match chunking {
                                "all" => run_stream(&probs, || probs.len(), p),
                                "rand" => {
                                    let mut r = Rng(total as u64);
                                    run_stream(&probs, move || 1 + r.below(40), p)
                                }
                                n => {
                                    let n: usize = n.parse().unwrap();
                                    run_stream(&probs, move || n, p)
                                }
                            };
                            let held = &mut worst_held[(min_sp < 32.0) as usize];
                            *held = (*held).max(out.max_held_wins);
                            if out.merged != reference || !out.violations.is_empty() {
                                failures += 1;
                                if failures <= 5 {
                                    eprintln!(
                                        "pad={pad} min_sil={min_sil} min_sp={min_sp} hyst={hyst} chunk={chunking}: got {} segs, want {}, violations {:?}",
                                        out.merged.len(),
                                        reference.len(),
                                        &out.violations[..out.violations.len().min(3)]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "runs={total} failures={failures} worst held back: {} windows with lookback, {} without",
            worst_held[0], worst_held[1]
        );
        assert_eq!(failures, 0);
        // pad 320 + min speech 600 + min silence 500 + chunk 40 windows (+ rounding)
        assert!(worst_held[0] <= 10 + 19 + 16 + 40 + 2);
        // lookback off: either the pad (320: 10 windows) or a silence run not yet min silence
        // long (500: at most 16 windows, from its first window to the last processed) minus the pad
        assert!(worst_held[1] <= 16);
    }

    // a long run of windows in [neg_threshold, threshold) after a blip must not keep growing the held-back audio
    #[test]
    fn held_back_is_bounded() {
        let mut probs = runs(&[(10, 0.0), (2, 0.9), (1, 0.1)]);
        probs.resize(3000, 0.45);
        let out = run_stream(&probs, || 10, params(64.0, 250.0, 250.0));
        assert!(out.max_held_wins <= 40, "held back {} windows", out.max_held_wins);
        assert!(out.merged.is_empty());

        // lookback off: the blip is a segment, ended after min silence
        let out = run_stream(&probs, || 10, no_look_back(64.0, 250.0));
        assert!(out.max_held_wins <= 40, "held back {} windows", out.max_held_wins);
        assert_eq!(out.merged, vec![SpeechSegment { start: 8 * WS, end: 14 * WS }]);
    }

    // streams an iterator that rewinds only for the lookback, and only to the trigger window, gets wrong
    #[test]
    fn call_boundary_regressions() {
        let p = params(64.0, 250.0, 250.0);
        let cases = [
            // speech starting at window 18 with chunks of 10: lookback, leading pad was lost
            ("leading pad after lookback", runs(&[(18, 0.0), (30, 0.9), (40, 0.0)]), 10, vec![SpeechSegment { start: 8192, end: 25600 }]),
            // chunks of 160ms < 250ms min silence: the first segment never closed
            (
                "silence spanning chunks",
                runs(&[(10, 0.0), (30, 0.9), (100, 0.0), (30, 0.9), (30, 0.0)]),
                5,
                vec![SpeechSegment { start: 4096, end: 21504 }, SpeechSegment { start: 70656, end: 88064 }],
            ),
            // 96ms blip, chunk boundary 6 windows into the silence after it: was emitted as speech
            ("blip + short trailing silence", runs(&[(3, 0.0), (3, 0.9), (60, 0.0)]), 12, vec![]),
        ];
        for (name, probs, chunk, want) in cases {
            let out = run_stream(&probs, || chunk, p);
            assert_eq!(out.merged, want, "{name}");
            assert_eq!(out.merged, offline_ref(&probs, VadIterator::detached(16_000, Some(p)).unwrap().params()), "{name}");
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Event {
        Start(i64),
        End(i64),
    }

    // silero's VADIterator.__call__ (utils_vad.py), called on each window in turn, on probabilities
    // instead of audio (16kHz, threshold - 0.15 as neg threshold). Returns its events with the window
    // that emits them, and the end of the segment still open when the stream ends, which silero
    // leaves to the caller: closed here as flush does.
    // hysteresis_closes makes the one change VadIterator makes to it: once a silence run has started,
    // a window in [threshold - 0.15, threshold) can end the segment too.
    fn silero_vad_iterator(probs: &[f32], p: &FindSpeechSegmentsParams, hysteresis_closes: bool) -> (Vec<(usize, Event)>, Option<i64>) {
        let (threshold, window_size_samples) = (p.threshold as f64, WS);
        let min_silence_samples = 16_000.0 * p.min_silence_duration_ms as f64 / 1000.0;
        let speech_pad_samples = 16_000.0 * p.pad_speech_ms as f64 / 1000.0;
        let (mut triggered, mut temp_end, mut current_sample) = (false, 0i64, 0i64);
        let mut events = Vec::new();

        for (i, &speech_prob) in probs.iter().enumerate() {
            let speech_prob = speech_prob as f64;
            current_sample += window_size_samples;

            if speech_prob >= threshold && temp_end != 0 {
                temp_end = 0;
            }
            if speech_prob >= threshold && !triggered {
                triggered = true;
                let speech_start = (current_sample as f64 - speech_pad_samples - window_size_samples as f64).max(0.0);
                events.push((i, Event::Start(speech_start as i64)));
                continue;
            }
            let silence = speech_prob < threshold - 0.15 || (hysteresis_closes && speech_prob < threshold && temp_end != 0);
            if silence && triggered {
                if temp_end == 0 {
                    temp_end = current_sample;
                }
                if ((current_sample - temp_end) as f64) < min_silence_samples {
                    continue;
                }
                let speech_end = temp_end as f64 + speech_pad_samples - window_size_samples as f64;
                temp_end = 0;
                triggered = false;
                events.push((i, Event::End(speech_end as i64)));
            }
        }

        let tot = probs.len() as i64 * WS;
        let open_end = triggered.then(|| {
            let end = if temp_end != 0 { temp_end + speech_pad_samples as i64 - window_size_samples } else { tot };
            end.min(tot)
        });
        (events, open_end)
    }

    // the segments of silero_vad_iterator, with the ones that overlap or touch once padded merged,
    // as VadIterator merges them (silero emits them overlapping)
    fn silero_segments(probs: &[f32], p: &FindSpeechSegmentsParams, hysteresis_closes: bool) -> Vec<SpeechSegment> {
        let (events, open_end) = silero_vad_iterator(probs, p, hysteresis_closes);
        let mut segs: Vec<SpeechSegment> = Vec::new();
        for (_, e) in events {
            match e {
                Event::Start(start) => segs.push(SpeechSegment { start, end: start }),
                Event::End(end) => segs.last_mut().unwrap().end = end,
            }
        }
        if let Some(end) = open_end {
            segs.last_mut().unwrap().end = end;
        }
        pad_and_merge(&segs, 0, 0, probs.len() as i64 * WS)
    }

    // With the lookback off (min speech under one window), for any chunking, the segments of silero's
    // VADIterator, which has no min speech: exactly, without
    // windows in [neg_threshold, threshold); with them, once the one rule VadIterator changes is
    // changed in silero's too (see hysteresis_windows_close_unlike_in_silero).
    #[test]
    fn without_look_back_equals_silero_vad_iterator() {
        let mut rng = Rng(2);
        let mut differ_from_silero = [0, 0]; // of 20 streams with hyst 0.15, 0.6
        for pad in [0.0, 32.0, 64.0, 96.0, 320.0] {
            for min_sil in [50.0, 100.0, 250.0, 500.0] {
                for hyst in [0.0, 0.15, 0.6] {
                    let p = no_look_back(pad, min_sil);
                    let probs = gen_probs(&mut rng, 1500, hyst);
                    let reference = silero_segments(&probs, &p, hyst > 0.0);
                    // segments before merging
                    assert!(silero_vad_iterator(&probs, &p, hyst > 0.0).0.len() > 20);
                    if hyst > 0.0 {
                        differ_from_silero[(hyst > 0.5) as usize] += (silero_segments(&probs, &p, false) != reference) as usize;
                    }
                    for chunk in [1, 2, 3, 8, 13, 31, 64, probs.len()] {
                        let out = run_stream(&probs, || chunk, p);
                        assert_eq!(out.violations, Vec::<String>::new(), "pad={pad} min_sil={min_sil} hyst={hyst} chunk={chunk}");
                        assert_eq!(out.merged, reference, "pad={pad} min_sil={min_sil} hyst={hyst} chunk={chunk}");
                    }
                }
            }
        }
        eprintln!(
            "streams that differ from unchanged silero: {} of 20 with 15% of windows in [neg_threshold, threshold), {} of 20 with 60%",
            differ_from_silero[0], differ_from_silero[1]
        );
    }

    // With the lookback off, called one window at a time, as silero's VADIterator is, a segment
    // starts in the call in which silero emits its start, and is complete (end < offset_samples())
    // in the call in which silero emits its end, at the same positions. With pads and min silences
    // that keep segments apart: two segments closer than 2 pads are merged, and the first is not
    // complete until the second cannot be merged with it anymore (here min silence + 1 window > 2 pads).
    #[test]
    fn without_look_back_events_come_with_silero_s() {
        let mut rng = Rng(3);
        for pad in [0.0, 32.0, 64.0] {
            for min_sil in [100.0, 250.0, 500.0] {
                for hyst in [0.0, 0.15, 0.6] {
                    let p = no_look_back(pad, min_sil);
                    let probs = gen_probs(&mut rng, 1500, hyst);
                    let (want, _) = silero_vad_iterator(&probs, &p, hyst > 0.0);
                    assert!(want.len() > 20);

                    let mut it = VadIterator::detached(16_000, Some(p)).unwrap();
                    let mut got = Vec::new();
                    let mut open = None; // end of the last segment emitted, while it may continue
                    for i in 0..probs.len() {
                        let window = &probs[(it.offset_samples() / WS) as usize..=i];
                        let (st, ct, gt) = dummy(window.len());
                        let r = it.find_speech_segments(window, st, ct, gt, None);
                        for (k, s) in r.speech_segments.iter().enumerate() {
                            if !(k == 0 && r.continues_speech) {
                                got.push((i, Event::Start(s.start)));
                            }
                            open = Some(s.end);
                        }
                        if let Some(end) = open
                            && end < it.offset_samples()
                        {
                            got.push((i, Event::End(end)));
                            open = None;
                        }
                    }
                    assert_eq!(got, want, "pad={pad} min_sil={min_sil} hyst={hyst}");
                }
            }
        }
    }

    // The one rule that differs from silero's. After a silence run has started, windows in
    // [neg_threshold, threshold) for longer than min silence, then speech again: silero keeps the
    // segment open until a window under neg_threshold comes, so the speech after it is the same
    // segment. VadIterator ends the segment after min silence (otherwise the audio held back would
    // grow without bound, see held_back_is_bounded) and the speech after it is a new one.
    #[test]
    fn hysteresis_windows_close_unlike_in_silero() {
        let p = no_look_back(64.0, 250.0);
        let probs = runs(&[(5, 0.0), (10, 0.9), (1, 0.1), (20, 0.4), (10, 0.9), (20, 0.0)]);
        assert_eq!(silero_segments(&probs, &p, false), vec![SpeechSegment { start: 3 * WS, end: 48 * WS }]);
        let want = vec![SpeechSegment { start: 3 * WS, end: 17 * WS }, SpeechSegment { start: 34 * WS, end: 48 * WS }];
        assert_eq!(silero_segments(&probs, &p, true), want);
        assert_eq!(run_stream(&probs, || 1, p).merged, want);
    }

    #[test]
    fn flush_resets() {
        let mut it = VadIterator::detached(16_000, None).unwrap();
        let probs = runs(&[(20, 0.9)]);
        let (st, ct, gt) = dummy(probs.len());
        it.find_speech_segments(&probs, st, ct, gt, None);
        assert!(it.triggered() && it.offset_samples() > 0);
        // empty audio: no inference needed, so this works without a session
        let r = it.flush_blocking(&[]).unwrap();
        assert!(r.speech_segments.is_empty() && r.look_back_from.is_none());
        assert!(!it.triggered() && it.offset_samples() == 0 && it.emitted_until.is_none());
    }
}
