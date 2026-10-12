// Helpers shared by the unit tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::core::pad_and_merge;
use crate::error::VadError;
use crate::gate::{GateState, NoiseGate, NoiseGateParams};
use crate::iterator::{FindSpeechSegmentsParams, SpeechSegment, VadIterationResult, VadIterator};
use crate::pool::{EngineFactory, InferenceEngine, WindowsOutput};
use crate::utils::{get_window_sz_and_context_sz, ms_to_samples};

pub(crate) const WS: i64 = 512;

// splitmix64: deterministic, no dependency
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub(crate) fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// runs of speech and silence with random lengths; hyst is the share of windows in [neg, thr)
pub(crate) fn gen_probs(rng: &mut Rng, n: usize, hyst: f32) -> Vec<f32> {
    let mut probs = Vec::with_capacity(n);
    let mut speech = false;
    while probs.len() < n {
        let l = 1 + rng.below(30);
        for _ in 0..l {
            if probs.len() == n {
                break;
            }
            let v = if rng.f32() < hyst {
                0.36 + rng.f32() * 0.13
            } else if speech {
                0.5 + rng.f32() * 0.5
            } else {
                rng.f32() * 0.34
            };
            probs.push(v);
        }
        speech = !speech;
    }
    probs
}

pub(crate) fn runs(parts: &[(usize, f32)]) -> Vec<f32> {
    parts.iter().flat_map(|&(n, v)| std::iter::repeat_n(v, n)).collect()
}

// Spec: single pass over the whole stream (16kHz), no streaming involved.
pub(crate) fn offline_ref(probs: &[f32], p: &FindSpeechSegmentsParams) -> Vec<SpeechSegment> {
    let min_sil = ms_to_samples(p.min_silence_duration_ms, 16_000);
    let min_sp = ms_to_samples(p.min_speech_duration_ms, 16_000);
    let pad = p.pad_speech_ms as i64 / 32 * WS;
    let mut segs = Vec::new();
    let (mut triggered, mut start, mut temp_end) = (false, 0i64, None::<i64>);
    for (i, &v) in probs.iter().enumerate() {
        let cur = i as i64 * WS;
        if v >= p.threshold {
            temp_end = None;
            if !triggered {
                (triggered, start) = (true, cur);
            }
            continue;
        }
        if !triggered {
            continue;
        }
        if v < p.neg_threshold && temp_end.is_none() {
            temp_end = Some(cur);
        }
        if let Some(t) = temp_end
            && cur - t >= min_sil
        {
            if t - start > min_sp {
                segs.push(SpeechSegment { start, end: t });
            }
            (triggered, temp_end) = (false, None);
        }
    }
    let tot = probs.len() as i64 * WS;
    if triggered {
        let end = temp_end.unwrap_or(tot);
        if end - start > min_sp {
            segs.push(SpeechSegment { start, end });
        }
    }
    pad_and_merge(&segs, pad, 0, tot)
}

// Audio for FakeEngine (16kHz), all windows at the same level.
pub(crate) fn probs_to_audio(probs: &[f32]) -> Vec<f32> {
    encode_audio(probs, &vec![0.1; probs.len()], None)
}

// Audio for FakeEngine (16kHz). Every sample of window k has magnitude levels[k], so that the gate
// sees a steady level, and the signs, which the gate does not see, carry 32 bit values: in the
// first 96 samples k, the model state the window must be run from and its probability, in the last
// 32 samples k again, so that the context passed with a window tells which window came before it.
// FakeEngine's state is 1 + the last window it ran: k without a gate, and with a gate 1 + the last
// window before k that the gate left open, as the model state does not move on gated windows.
pub(crate) fn encode_audio(probs: &[f32], levels: &[f32], gate: Option<&NoiseGateParams>) -> Vec<f32> {
    let ws = WS as usize;
    let mut audio: Vec<f32> = levels.iter().flat_map(|&l| std::iter::repeat_n(l, ws)).collect();
    let gated = gated_windows(&audio, gate);
    let mut state = 0;
    for (k, w) in audio.chunks_exact_mut(ws).enumerate() {
        write_bits(&mut w[..32], k as u32);
        write_bits(&mut w[32..64], state);
        write_bits(&mut w[64..96], probs[k].to_bits());
        write_bits(&mut w[ws - 32..], k as u32);
        if !gated[k] {
            state = k as u32 + 1;
        }
    }
    audio
}

// The windows of a whole stream (16kHz) the gate closes on.
pub(crate) fn gated_windows(audio: &[f32], gate: Option<&NoiseGateParams>) -> Vec<bool> {
    match gate {
        Some(p) => NoiseGate::new(p, 16_000).run(GateState::default(), audio, WS as usize).0,
        None => vec![false; audio.len() / WS as usize],
    }
}

fn write_bits(samples: &mut [f32], v: u32) {
    for (i, x) in samples.iter_mut().enumerate() {
        if v >> i & 1 == 1 {
            *x = -*x;
        }
    }
}

fn read_bits(samples: &[f32]) -> u32 {
    samples.iter().enumerate().fold(0, |v, (i, x)| v | (x.is_sign_negative() as u32) << i)
}

// Stands in for the model, see encode_audio. A request starting from the wrong state or context
// (e.g. after a wrong rewind) is an error.
#[derive(Default)]
pub(crate) struct FakeEngine {
    pub(crate) delay: Option<Duration>,
    pub(crate) windows_run: Option<Arc<AtomicUsize>>,
}

impl InferenceEngine for FakeEngine {
    fn run(&mut self, sample_rate: i64, audio: &[f32], state: &[f32], context: &[f32]) -> Result<WindowsOutput, VadError> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }
        let (ws, cs) = get_window_sz_and_context_sz(sample_rate);
        let (ws, cs) = (ws as usize, cs as usize);
        let fail = |msg: String| Err(VadError::InvalidParams(format!("fake engine: {msg}")));

        if let Some(w) = audio.get(..ws) {
            let k = read_bits(&w[..32]);
            let context_ok = if k == 0 { context.iter().all(|&x| x == 0.0) } else { read_bits(&context[cs - 32..]) == k - 1 };
            if !context_ok {
                return fail(format!("window {k} passed with the context of another window"));
            }
        }

        let mut s = state.to_vec();
        let mut out = WindowsOutput::default();
        for w in audio.chunks_exact(ws) {
            let k = read_bits(&w[..32]);
            let want = read_bits(&w[32..64]);
            if s[0] != want as f32 {
                return fail(format!("window {k} run from state {}, want {want}", s[0]));
            }
            s[0] = (k + 1) as f32;
            out.probs.push(f32::from_bits(read_bits(&w[64..96])));
            out.states.push(s.clone());
            out.contexts.push(w[ws - cs..].to_vec());
        }
        if let Some(n) = &self.windows_run {
            n.fetch_add(out.probs.len(), Ordering::Relaxed);
        }
        Ok(out)
    }
}

pub(crate) fn fake_factory(delay: Option<Duration>) -> EngineFactory {
    Arc::new(move || Ok(Box::new(FakeEngine { delay, windows_run: None }) as Box<dyn InferenceEngine>))
}

// Caller following the protocol: keeps the audio from offset_samples() on, appends the new chunk
// (`chunk()` samples), calls vad, finally flushes. With `cancel`, calls are given a random short
// timeout and retried until one completes. Returns the segments merged across calls and the
// windows gated in total.
pub(crate) async fn stream_async(
    it: &mut VadIterator,
    audio: &[f32],
    mut chunk: impl FnMut() -> usize,
    cancel: bool,
) -> (Vec<SpeechSegment>, usize) {
    let mut merged: Vec<SpeechSegment> = Vec::new();
    let mut gated = 0;
    let mut add = |merged: &mut Vec<SpeechSegment>, r: &VadIterationResult| {
        gated += r.gated_windows;
        for (k, s) in r.speech_segments.iter().enumerate() {
            match merged.last_mut() {
                Some(last) if k == 0 && r.continues_speech => {
                    assert_eq!(last.end, s.start);
                    last.end = s.end;
                }
                _ => merged.push(*s),
            }
        }
    };
    let mut timeouts = Rng(99);
    let mut cancelled = 0;

    let mut buf: Vec<f32> = Vec::new();
    let mut fed = 0;
    while fed < audio.len() {
        let n = chunk().min(audio.len() - fed);
        buf.extend_from_slice(&audio[fed..fed + n]);
        fed += n;
        let start = it.offset_samples();
        let r = loop {
            match if cancel { timeouts.below(3) } else { 2 } {
                0 => {
                    // dropped right after its (first) request is sent
                    let res = tokio::select! {
                        biased;
                        r = it.vad(&buf) => Some(r),
                        _ = std::future::ready(()) => None,
                    };
                    if let Some(r) = res {
                        break r.unwrap();
                    }
                }
                1 => {
                    // dropped at a random point, possibly after some of its requests completed
                    let limit = Duration::from_micros(timeouts.below(3000) as u64);
                    if let Ok(r) = tokio::time::timeout(limit, it.vad(&buf)).await {
                        break r.unwrap();
                    }
                }
                _ => break it.vad(&buf).await.unwrap(),
            }
            cancelled += 1;
            assert_eq!(it.offset_samples(), start);
        };
        assert_eq!(r.look_back_from.is_some(), it.offset_samples() < start + buf.len() as i64);
        add(&mut merged, &r);
        buf.drain(..(it.offset_samples() - start) as usize);
    }
    let r = it.flush(&buf).await.unwrap();
    add(&mut merged, &r);
    assert!(!cancel || cancelled > 10, "only {cancelled} calls cancelled");
    (merged, gated)
}
