// Noise gate run before the model, a port of JUCE's dsp::NoiseGate (the one pedalboard wraps).
//
// The gate does not touch the audio, it only classifies the windows: a window during which the gate
// stays closed is non-speech, and is not run through the model at all (see VadCore::inference_loop).

/// Noise gate run before the model, see [`FindSpeechSegmentsParams::noise_gate`](crate::FindSpeechSegmentsParams::noise_gate).
///
/// A port of JUCE's `dsp::NoiseGate`, which is what `pedalboard.NoiseGate` runs, with the same
/// parameters and units: values tuned with pedalboard can be used as they are. The level of the
/// signal is followed by an RMS detector (attack 0, release 50 ms, fixed as in JUCE) and then by an
/// envelope with the attack and release below; the gate is open while the envelope is above the
/// threshold. There is no ratio: the gate never attenuates the audio, it only decides which windows
/// are non-speech without running the model.
///
/// Times follow JUCE's convention: a time of `t` ms is the time the envelope takes to cover 99.8%
/// (-55 dB) of a step, i.e. its time constant is `t / 2π`. With `release_ms: 1000.0` the envelope
/// falls by about 55 dB in the second after the signal stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseGateParams {
    /// The gate opens when the envelope goes above this level, in dBFS (relative to a full scale of
    /// 1.0). At -200 or less it closes only on digital silence. Default -60.
    pub threshold_db: f32,
    /// Attack of the envelope, in ms. Must not be negative; below 0.001 it is instant. Default 1.
    pub attack_ms: f32,
    /// Release of the envelope, in ms. Must not be negative; below 0.001 it is instant. Default 1000.
    pub release_ms: f32,
}

impl Default for NoiseGateParams {
    fn default() -> Self {
        NoiseGateParams {
            threshold_db: -60.0,
            attack_ms: 1.0,
            release_ms: 1000.0,
        }
    }
}

/// State of the gate's two filters at a window boundary. Like the model state, one snapshot per
/// window is kept during a call, so that the iterator can rewind it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct GateState {
    mean_square: f32, // the RMS detector keeps the mean square, as in JUCE
    envelope: f32,
}

pub(crate) struct NoiseGate {
    threshold: f32,   // linear
    rms_release: f32, // the RMS detector's attack is 0
    attack: f32,
    release: f32,
}

// JUCE's BallisticsFilter::calculateLimitedCte: the weight of the previous value at each sample
fn smoothing_coeff(time_ms: f32, sample_rate: i64) -> f32 {
    if time_ms < 1.0e-3 {
        return 0.0;
    }
    let exp_factor = -2.0 * std::f64::consts::PI * 1000.0 / sample_rate as f64;
    (exp_factor / time_ms as f64).exp() as f32
}

// denormals are very slow on some CPUs, and in digital silence the filters decay through them
fn flush_denormal(x: f32) -> f32 {
    if x < f32::MIN_POSITIVE { 0.0 } else { x }
}

impl NoiseGate {
    pub(crate) fn new(params: &NoiseGateParams, sample_rate: i64) -> Self {
        NoiseGate {
            // JUCE's Decibels::decibelsToGain, with -200 dB as minus infinity
            threshold: if params.threshold_db > -200.0 { 10f32.powf(params.threshold_db * 0.05) } else { 0.0 },
            rms_release: smoothing_coeff(50.0, sample_rate),
            attack: smoothing_coeff(params.attack_ms, sample_rate),
            release: smoothing_coeff(params.release_ms, sample_rate),
        }
    }

    /// Runs the gate over `window` from `state`, which is updated. True if the gate stayed closed
    /// for the whole window: a single open sample is enough to send the window to the model.
    pub(crate) fn closed(&self, state: &mut GateState, window: &[f32]) -> bool {
        let mut closed = true;
        for &x in window {
            // NoiseGate::processSample: RMS detector, then envelope, then threshold.
            // BallisticsFilter::processSample is y = x + c * (y - x), c the attack or the release
            let sq = x * x;
            let c = if sq > state.mean_square { 0.0 } else { self.rms_release };
            state.mean_square = flush_denormal(sq + c * (state.mean_square - sq));

            let rms = state.mean_square.sqrt();
            let c = if rms > state.envelope { self.attack } else { self.release };
            state.envelope = flush_denormal(rms + c * (state.envelope - rms));

            if state.envelope > self.threshold {
                closed = false;
            }
        }
        closed
    }

    /// Runs the gate over the windows of `audio` (a whole number of windows) from `state`. Returns
    /// whether each window is gated, and the gate state before each window plus the one after the
    /// last (`windows + 1` entries).
    pub(crate) fn run(&self, mut state: GateState, audio: &[f32], window_size: usize) -> (Vec<bool>, Vec<GateState>) {
        let n_windows = audio.len() / window_size;
        let mut gated = Vec::with_capacity(n_windows);
        let mut states = Vec::with_capacity(n_windows + 1);
        states.push(state);
        for window in audio.chunks_exact(window_size) {
            gated.push(self.closed(&mut state, window));
            states.push(state);
        }
        (gated, states)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::iterator::{FindSpeechSegmentsParams, SpeechSegment, VadIterator};
    use crate::pool::{EngineFactory, InferenceEngine, SessionPool, VadEnvConfig};
    use crate::testutil::{FakeEngine, Rng, encode_audio, fake_factory, gated_windows, gen_probs, offline_ref, stream_async};

    // bursts of noise at levels from -140 to -10 dBFS
    fn bursts() -> Vec<f32> {
        let parts: [(usize, f32); 12] = [
            (4800, 1e-4), (8000, 0.1), (6400, 1e-3), (4800, 0.03), (16000, 1e-5), (3200, 0.3),
            (9600, 3e-4), (1600, 0.05), (800, 2e-3), (12000, 1e-7), (4000, 0.01), (8000, 1e-4),
        ];
        let mut rng = Rng(42);
        let mut x = Vec::new();
        for (n, level) in parts {
            for _ in 0..n {
                x.push(level * (2.0 * rng.f32() - 1.0));
            }
        }
        x
    }

    // The samples of bursts() where pedalboard's gate opens or closes: where its output starts or
    // stops being equal to its input. Made with pedalboard 0.9.26 and numpy, bursts() rebuilt with
    // the same splitmix64 and the same float32 operations:
    //
    //   y = pedalboard.NoiseGate(threshold_db, 10, attack_ms, release_ms)(x.reshape(1, -1), sr)[0]
    //   is_open = y == x
    //   flips = [i for i in range(len(x)) if is_open[i] != (is_open[i - 1] if i > 0 else False)]
    #[test]
    fn matches_pedalboard() {
        let golden: [(i64, f32, f32, f32, &[usize]); 5] = [
            (16000, -40.0, 1.0, 100.0, &[4800, 13775, 19201, 24576, 40000, 44506, 52801, 55153]),
            (16000, -50.0, 1.0, 1000.0, &[4800, 29969, 40000, 61651, 67201, 74417]),
            (16000, -60.0, 0.0, 500.0, &[4800, 28613, 40000, 51120, 52800, 59639, 67200, 74497]),
            (16000, -30.0, 5.0, 300.0, &[4807, 13953, 40002, 45190, 52815, 54975]),
            (8000, -50.0, 1.0, 1000.0, &[4800, 17725, 19200, 26982, 40000, 49208, 52800, 58039, 67201, 72801]),
        ];
        let x = bursts();
        for (sample_rate, threshold_db, attack_ms, release_ms, want) in golden {
            let gate = NoiseGate::new(&NoiseGateParams { threshold_db, attack_ms, release_ms }, sample_rate);
            let mut state = GateState::default();
            let mut flips = Vec::new();
            let mut was_open = false;
            for (i, &sample) in x.iter().enumerate() {
                let open = !gate.closed(&mut state, &[sample]);
                if open != was_open {
                    flips.push(i);
                    was_open = open;
                }
            }
            assert_eq!(flips, want, "{sample_rate}Hz {threshold_db}dB attack {attack_ms}ms release {release_ms}ms");
        }
    }

    #[test]
    fn one_open_sample_keeps_the_window() {
        let gate = NoiseGate::new(&NoiseGateParams::default(), 16_000);
        let mut state = GateState::default();
        let mut window = [1e-5; 512];
        assert!(gate.closed(&mut state.clone(), &window));
        window[511] = 0.5;
        assert!(!gate.closed(&mut state, &window));
    }

    // without flushing denormals the envelope would never reach 0: the smallest denormal times a
    // release coefficient close to 1 rounds back to itself
    #[test]
    fn digital_silence_decays_to_zero() {
        let gate = NoiseGate::new(&NoiseGateParams { threshold_db: -200.0, ..Default::default() }, 16_000);
        let mut state = GateState::default();
        assert!(!gate.closed(&mut state, &[0.5; 512]));
        // the envelope falls ~55 dB per release time (1s): from -6 dB to f32::MIN_POSITIVE in ~14s
        let closed: Vec<bool> = (0..20 * 16_000 / 512).map(|_| gate.closed(&mut state, &[0.0; 512])).collect();
        assert_eq!(state, GateState::default());
        assert!(closed[closed.len() - 1] && !closed[10 * 16_000 / 512]);
    }

    // windows in runs of random length at random levels from -90 to -10 dBFS, so that the gate
    // opens and closes often, also in the middle of runs
    fn gen_levels(rng: &mut Rng, n: usize) -> Vec<f32> {
        let mut levels = Vec::with_capacity(n);
        while levels.len() < n {
            let level = 10f32.powf((-90.0 + 80.0 * rng.f32()) / 20.0);
            let len = (1 + rng.below(40)).min(n - levels.len());
            levels.extend(std::iter::repeat_n(level, len));
        }
        levels
    }

    fn gated_probs(probs: &[f32], gated: &[bool]) -> Vec<f32> {
        probs.iter().zip(gated).map(|(&p, &g)| if g { 0.0 } else { p }).collect()
    }

    fn params(pad: f32, gate: Option<NoiseGateParams>) -> FindSpeechSegmentsParams {
        FindSpeechSegmentsParams { pad_speech_ms: pad, noise_gate: gate, ..Default::default() }
    }

    // Many streams through the pool with the gate on, chunks of any size, some with calls cancelled:
    // each must give exactly what one offline pass gives over its probabilities with the gated
    // windows set to 0. FakeEngine checks that every window it runs starts from the right state
    // and context: the state left by the last window before it that the gate left open
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gated_streams_equal_the_gated_reference() {
        let config = VadEnvConfig { max_streams_per_session: 4, min_free_slots: 2 };
        let pool = SessionPool::new(config, fake_factory(Some(Duration::from_micros(200)))).unwrap();
        let gates = [
            NoiseGateParams::default(),
            NoiseGateParams { threshold_db: -40.0, attack_ms: 0.0, release_ms: 100.0 },
            NoiseGateParams { threshold_db: -50.0, attack_ms: 5.0, release_ms: 300.0 },
        ];

        let mut tasks = Vec::new();
        for k in 0..24 {
            let gate = gates[k % gates.len()];
            let pad = [0.0, 64.0, 96.0, 320.0][k / gates.len() % 4];
            let mut it = VadIterator::on_pool(16_000, Some(params(pad, Some(gate))), &pool).unwrap();
            tasks.push(tokio::spawn(async move {
                let mut rng = Rng(k as u64);
                let probs = gen_probs(&mut rng, 800, 0.15);
                let audio = encode_audio(&probs, &gen_levels(&mut rng, probs.len()), Some(&gate));
                let gated = gated_windows(&audio, Some(&gate));
                let n_gated = gated.iter().filter(|&&g| g).count();
                assert!(n_gated >= 20 && n_gated <= probs.len() - 20, "stream {k}: {n_gated} gated");

                let reference = offline_ref(&gated_probs(&probs, &gated), it.params());
                // calls on gated audio only need no request, and cannot be cancelled: smaller chunks
                // when cancelling, so that enough calls are
                let cancel = k % 4 == 0;
                let max_chunk = if cancel { 10 * 512 } else { 40 * 512 };
                let (got, got_gated) = stream_async(&mut it, &audio, || 1 + rng.below(max_chunk), cancel).await;
                assert_eq!(got, reference, "stream {k}");
                assert_eq!(got_gated, n_gated, "stream {k}");
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
    }

    async fn stream(pool: &Arc<SessionPool>, audio: &[f32], gate: Option<NoiseGateParams>) -> (Vec<SpeechSegment>, usize) {
        let mut it = VadIterator::on_pool(16_000, Some(params(64.0, gate)), pool).unwrap();
        stream_async(&mut it, audio, || 3000, false).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gate_that_never_closes_changes_nothing() {
        let pool = SessionPool::new(VadEnvConfig::default(), fake_factory(None)).unwrap();
        let mut rng = Rng(3);
        let probs = gen_probs(&mut rng, 600, 0.15);
        let levels = gen_levels(&mut rng, probs.len());
        // at -200 dB the gate closes only on digital silence
        let gate = NoiseGateParams { threshold_db: -200.0, ..Default::default() };
        let audio = encode_audio(&probs, &levels, Some(&gate));
        assert_eq!(audio, encode_audio(&probs, &levels, None));

        let (without, _) = stream(&pool, &audio, None).await;
        assert_eq!(stream(&pool, &audio, Some(gate)).await, (without, 0));
    }

    #[test]
    fn gated_windows_are_not_run_through_the_model() {
        let windows_run = Arc::new(AtomicUsize::new(0));
        let factory: EngineFactory = {
            let windows_run = Arc::clone(&windows_run);
            Arc::new(move || {
                let windows_run = Some(Arc::clone(&windows_run));
                Ok(Box::new(FakeEngine { delay: None, windows_run }) as Box<dyn InferenceEngine>)
            })
        };
        let pool = SessionPool::new(VadEnvConfig::default(), factory).unwrap();

        let gate = NoiseGateParams::default();
        let mut rng = Rng(5);
        let probs = gen_probs(&mut rng, 600, 0.15);
        let audio = encode_audio(&probs, &gen_levels(&mut rng, probs.len()), Some(&gate));
        let n_gated = gated_windows(&audio, Some(&gate)).iter().filter(|&&g| g).count();

        // a single call processes every window once
        let mut it = VadIterator::on_pool(16_000, Some(params(0.0, Some(gate))), &pool).unwrap();
        let r = it.flush_blocking(&audio).unwrap();
        assert_eq!(r.gated_windows, n_gated);
        assert_eq!(windows_run.load(Ordering::Relaxed), probs.len() - n_gated);
    }

    #[test]
    fn gated_audio_needs_no_session() {
        // detached: no session, any request to the model fails
        let mut it = VadIterator::detached(16_000, Some(params(0.0, Some(NoiseGateParams::default())))).unwrap();
        let quiet = vec![1e-5; 100 * 512]; // -100 dBFS
        let r = it.vad_blocking(&quiet).unwrap();
        assert_eq!((r.gated_windows, it.offset_samples()), (100, quiet.len() as i64));
        assert!(it.vad_blocking(&[0.1; 512]).is_err());
    }
}
