// Streams a real file through the real model and requires the same segments:
// - at several chunk sizes (including sizes that are not a multiple of the window) as a single call;
// - with many streams running at once on the sessions pool;
// - with the blocking API.
// With the noise gate (see noise_gate below for what is compared with what):
// - the same segments as without it, when it never closes or closes only in short pauses;
// - after long gated pauses, the same segments with edges moved by a bounded number of windows;
// - the same segments at any chunk size.
// With the lookback off (min speech under one window), the same segments at any chunk size.
//
// Skipped unless VAD_ORT_LIB, VAD_MODEL and VAD_WAV (16kHz mono 16 bit) are set:
//
//   VAD_ORT_LIB=onnxruntime/onnxruntime-osx-arm64-1.28.3/lib/libonnxruntime.dylib \
//   VAD_MODEL=models/silero_vad.onnx VAD_WAV=audio_assets/test.wav \
//   cargo test --release --test real_model -- --nocapture

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use silero_vad_rs::{
    FindSpeechSegmentsParams, NoiseGateParams, SpeechSegment, VadEnvConfig, VadIterationResult, VadIterator,
    initialize, pool_stats,
};

// the tests share the pool and real_model checks its stats: one test at a time
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn setup() -> Option<Vec<f32>> {
    let (Ok(ort_lib), Ok(model), Ok(wav)) =
        (std::env::var("VAD_ORT_LIB"), std::env::var("VAD_MODEL"), std::env::var("VAD_WAV"))
    else {
        eprintln!("skipped: set VAD_ORT_LIB, VAD_MODEL and VAD_WAV to run");
        return None;
    };
    let config = VadEnvConfig { max_streams_per_session: 16, min_free_slots: 4 };
    initialize(ort_lib, model, Some(config)).unwrap();

    let mut reader = hound::WavReader::open(wav).unwrap();
    let spec = reader.spec();
    assert_eq!((spec.sample_rate, spec.channels, spec.bits_per_sample), (16_000, 1, 16), "need 16kHz mono 16 bit");
    Some(reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect())
}

fn params(noise_gate: Option<NoiseGateParams>) -> FindSpeechSegmentsParams {
    FindSpeechSegmentsParams {
        threshold: 0.5,
        min_speech_duration_ms: 250.0,
        min_silence_duration_ms: 250.0,
        pad_speech_ms: 64.0,
        noise_gate,
        ..Default::default()
    }
}

fn gate(threshold_db: f32) -> Option<NoiseGateParams> {
    Some(NoiseGateParams { threshold_db, attack_ms: 1.0, release_ms: 1000.0 })
}

fn add(merged: &mut Vec<SpeechSegment>, gated: &mut usize, r: &VadIterationResult) {
    *gated += r.gated_windows;
    for (k, s) in r.speech_segments.iter().enumerate() {
        match merged.last_mut() {
            Some(last) if k == 0 && r.continues_speech => last.end = s.end,
            _ => merged.push(*s),
        }
    }
}

// the segments merged across calls, and the windows gated in total
async fn stream(
    audio: &[f32],
    sample_rate: i64,
    chunk: usize,
    params: FindSpeechSegmentsParams,
) -> (Vec<SpeechSegment>, usize) {
    let mut it = VadIterator::new(sample_rate, Some(params)).unwrap();
    let (mut merged, mut gated) = (Vec::new(), 0);
    let mut buf: Vec<f32> = Vec::new();
    for c in audio.chunks(chunk) {
        buf.extend_from_slice(c);
        let start = it.offset_samples();
        let r = it.vad(&buf).await.unwrap();
        assert_eq!(r.look_back_from.is_some(), it.offset_samples() < start + buf.len() as i64);
        add(&mut merged, &mut gated, &r);
        buf.drain(..(it.offset_samples() - start) as usize);
    }
    add(&mut merged, &mut gated, &it.flush(&buf).await.unwrap());
    assert_eq!(it.offset_samples(), 0);
    (merged, gated)
}

fn stream_blocking(audio: &[f32], sample_rate: i64, chunk: usize) -> Vec<SpeechSegment> {
    let mut it = VadIterator::new(sample_rate, Some(params(None))).unwrap();
    let (mut merged, mut gated) = (Vec::new(), 0);
    let mut buf: Vec<f32> = Vec::new();
    for c in audio.chunks(chunk) {
        buf.extend_from_slice(c);
        let start = it.offset_samples();
        add(&mut merged, &mut gated, &it.vad_blocking(&buf).unwrap());
        buf.drain(..(it.offset_samples() - start) as usize);
    }
    add(&mut merged, &mut gated, &it.flush_blocking(&buf).unwrap());
    merged
}

async fn check_chunk_invariance(
    audio: &[f32],
    sample_rate: i64,
    chunks: &[usize],
    params: FindSpeechSegmentsParams,
) -> Vec<SpeechSegment> {
    let reference = stream(audio, sample_rate, audio.len(), params).await;
    eprintln!("{sample_rate}Hz single call: {} segments, {} windows gated {:?}", reference.0.len(), reference.1, reference.0);
    assert!(!reference.0.is_empty());
    for &chunk in chunks {
        let got = stream(audio, sample_rate, chunk, params).await;
        eprintln!("{sample_rate}Hz chunk={chunk:>5} samples: equal={}", got == reference);
        assert_eq!(got, reference, "{sample_rate}Hz chunk={chunk}");
    }
    reference.0
}

// 8kHz (256 samples windows): decimate by averaging pairs, good enough to exercise the path
fn to_8k(audio: &[f32]) -> Vec<f32> {
    audio.as_chunks::<2>().0.iter().map(|[a, b]| (a + b) / 2.0).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_model() {
    let _serial = SERIAL.lock().await;
    let Some(audio) = setup() else { return };

    let reference = check_chunk_invariance(&audio, 16_000, &[512, 1000, 1600, 4096, 8000, 16000, 33333], params(None)).await;
    check_chunk_invariance(&to_8k(&audio), 8_000, &[256, 500, 800, 2048, 4000, 16667], params(None)).await;

    // blocking API, from a thread outside the async runtime
    let (a, r) = (audio.clone(), reference.clone());
    std::thread::spawn(move || assert_eq!(stream_blocking(&a, 16_000, 1600), r)).join().unwrap();
    eprintln!("blocking API: equal=true");

    // 40 streams at once: 3 sessions of 16 needed
    let audio = Arc::new(audio);
    let peak = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::new();
    for k in 0..40usize {
        let (audio, reference) = (Arc::clone(&audio), reference.clone());
        tasks.push(tokio::spawn(async move {
            let (got, _) = stream(&audio, 16_000, 800 + 37 * k, params(None)).await;
            assert_eq!(got, reference, "stream {k}");
        }));
    }
    // sample the pool while the streams run
    while !tasks.iter().all(|t| t.is_finished()) {
        peak.fetch_max(pool_stats().unwrap().sessions, Ordering::Relaxed);
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    for t in tasks {
        t.await.unwrap();
    }
    let peak = peak.load(Ordering::Relaxed);
    eprintln!("40 concurrent streams: all equal, peak sessions {peak}, now {:?}", pool_stats().unwrap());
    assert!(peak >= 3, "peak sessions {peak}");
    assert_eq!(pool_stats().unwrap().streams, 0);
}

// Lookback off (min speech under one window): the same segments at any chunk size, also with the
// gate, and every segment found with lookback is inside one found without (the speech runs are the
// same, without lookback the short ones are kept too).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_look_back() {
    let _serial = SERIAL.lock().await;
    let Some(audio) = setup() else { return };
    let no_look_back = |p| FindSpeechSegmentsParams { min_speech_duration_ms: 1.0, ..p };

    let without = check_chunk_invariance(&audio, 16_000, &[512, 1000, 1600, 4096, 33333], no_look_back(params(None))).await;
    check_chunk_invariance(&to_8k(&audio), 8_000, &[256, 800, 16667], no_look_back(params(None))).await;
    check_chunk_invariance(&audio, 16_000, &[512, 1600, 33333], no_look_back(params(gate(-40.0)))).await;

    let (with, _) = stream(&audio, 16_000, audio.len(), params(None)).await;
    eprintln!("{} segments with lookback, {} without", with.len(), without.len());
    for s in &with {
        assert!(without.iter().any(|w| w.start <= s.start && s.end <= w.end), "{s:?} not in {without:?}");
    }
}

// test.wav plus a noise floor at -60 dBFS, with 3s of it inserted in the middle of every pause
// between `segments`: a stream with long pauses, which test.wav (speech for 3/4) lacks
fn with_long_pauses(audio: &[f32], segments: &[SpeechSegment]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut from = 0;
    for pair in segments.windows(2) {
        let cut = ((pair[0].end + pair[1].start) / 2) as usize;
        out.extend_from_slice(&audio[from..cut]);
        out.extend(std::iter::repeat_n(0.0, 3 * 16_000));
        from = cut;
    }
    out.extend_from_slice(&audio[from..]);

    // uniform noise in [-a, a] has rms a/sqrt(3): a = sqrt(3) * 10^(-60/20). A literal, so that
    // the audio is the same bit for bit wherever it is built
    let a = 0.001_732_050_8f32;
    let mut seed = 0x2545f4914f6cdd1du64;
    for x in &mut out {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *x += a * (2.0 * ((seed >> 40) as f32 / (1u64 << 24) as f32) - 1.0);
    }
    out
}

// largest distance between corresponding edges of two lists of segments of the same length
fn max_edge_shift(a: &[SpeechSegment], b: &[SpeechSegment]) -> i64 {
    a.iter().zip(b).map(|(x, y)| (x.start - y.start).abs().max((x.end - y.end).abs())).max().unwrap_or(0)
}

// What the gate can and cannot change. A gated window is not run through the model, so the model
// state does not move there (it is kept as it was): without the gate it keeps following the
// audio. After a short gated pause the two states are close enough that no probability ends up
// on the other side of a threshold; after a long one, the first probabilities after the pause can
// (e.g. 0.351 without the gate, 0.773 with it), and an edge moves by a whole window. Equality is
// therefore required where it holds, and a bound where it cannot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn noise_gate() {
    let _serial = SERIAL.lock().await;
    let Some(audio) = setup() else { return };
    let windows = audio.len() / 512;

    let (reference, _) = stream(&audio, 16_000, audio.len(), params(None)).await;

    // a gate that never closes (-200 dB: only on digital silence) changes nothing
    assert_eq!(stream(&audio, 16_000, audio.len(), params(gate(-200.0))).await, (reference.clone(), 0));

    // test.wav has only short pauses: at sensible thresholds the gate closes in them and the
    // segments do not change
    for (threshold_db, min_gated) in [(-50.0, windows / 50), (-40.0, windows / 15)] {
        let (got, gated) = stream(&audio, 16_000, audio.len(), params(gate(threshold_db))).await;
        eprintln!("gate {threshold_db}dB: {gated} of {windows} windows gated, equal={}", got == reference);
        assert!(gated >= min_gated, "gate {threshold_db}dB: only {gated} windows gated");
        assert_eq!(got, reference, "gate {threshold_db}dB");
    }

    // streaming with the gate equals a single call
    check_chunk_invariance(&audio, 16_000, &[512, 1000, 1600, 8000, 33333], params(gate(-40.0))).await;
    check_chunk_invariance(&to_8k(&audio), 8_000, &[256, 800, 16667], params(gate(-40.0))).await;

    // long pauses: the same segments, edges moved by at most MAX_SHIFT windows. 2 is what was
    // measured at all three thresholds (ONNX Runtime 1.28.3 on arm64), with 9-10 segments of 17 moved
    const MAX_SHIFT: i64 = 2;
    let long = with_long_pauses(&audio, &reference);
    let windows = long.len() / 512;
    let (without, _) = stream(&long, 16_000, long.len(), params(None)).await;
    for threshold_db in [-55.0, -50.0, -45.0] {
        let (with, gated) = stream(&long, 16_000, long.len(), params(gate(threshold_db))).await;
        let moved = with.iter().zip(&without).filter(|(a, b)| a != b).count();
        let shift = max_edge_shift(&with, &without);
        eprintln!(
            "long pauses, gate {threshold_db}dB: {gated} of {windows} windows gated, {} segments ({} without), {moved} moved, max shift {} windows",
            with.len(),
            without.len(),
            shift as f64 / 512.0
        );
        assert!(gated >= windows / 4, "gate {threshold_db}dB: only {gated} windows gated");
        assert_eq!(with.len(), without.len(), "gate {threshold_db}dB");
        assert!(shift <= MAX_SHIFT * 512, "gate {threshold_db}dB: an edge moved by {shift} samples");
    }
    check_chunk_invariance(&long, 16_000, &[512, 1600, 33333], params(gate(-50.0))).await;
}
