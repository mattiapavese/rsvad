// Simulates a live stream from a wav file (16kHz or 8kHz, mono, 16 bit) and prints the speech
// segments as they become final. With a gate threshold (dBFS), runs the noise gate before the model
// and prints how many windows it gated.
//
//   cargo run --example stream_wav -- <libonnxruntime> <silero_vad.onnx> <file.wav> [chunk_ms] [gate_threshold_db]

use silero_vad_rs::{FindSpeechSegmentsParams, NoiseGateParams, VadIterator, initialize};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <libonnxruntime> <silero_vad.onnx> <file.wav> [chunk_ms] [gate_threshold_db]", args[0]);
        std::process::exit(2);
    }
    let chunk_ms: usize = args.get(4).map_or(Ok(100), |s| s.parse())?;
    let noise_gate = match args.get(5) {
        Some(s) => Some(NoiseGateParams { threshold_db: s.parse()?, ..Default::default() }),
        None => None,
    };

    initialize(&args[1], &args[2], None)?;

    let mut reader = hound::WavReader::open(&args[3])?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("need a mono 16 bit wav".into());
    }
    let sample_rate = spec.sample_rate as i64;
    let audio: Vec<f32> = reader.samples::<i16>().map(|s| s.map(|s| s as f32 / 32768.0)).collect::<Result<_, _>>()?;

    let params = FindSpeechSegmentsParams { pad_speech_ms: 64.0, noise_gate, ..Default::default() };
    let mut it = VadIterator::new(sample_rate, Some(params))?;
    let secs = |samples: i64| samples as f64 / sample_rate as f64;

    let mut buf: Vec<f32> = Vec::new(); // audio from it.offset_samples() on
    let mut chunks = audio.chunks(chunk_ms * sample_rate as usize / 1000).peekable();
    let mut received = 0;
    let mut gated = 0;
    while let Some(chunk) = chunks.next() {
        buf.extend_from_slice(chunk);
        received += chunk.len() as i64;
        let start = it.offset_samples();

        let last = chunks.peek().is_none();
        let res = if last { it.flush(&buf).await? } else { it.vad(&buf).await? };
        let final_until = if last { received } else { it.offset_samples() };
        gated += res.gated_windows;

        for (i, s) in res.speech_segments.iter().enumerate() {
            let speech = &buf[(s.start - start) as usize..(s.end - start) as usize];
            println!(
                "at {:>7.3}s  speech {:>7.3}s - {:>7.3}s ({} samples){}{}",
                secs(received),
                secs(s.start),
                secs(s.end),
                speech.len(),
                if i == 0 && res.continues_speech { "  continues" } else { "" },
                if s.end < final_until { "  [complete]" } else { "" },
            );
        }

        if !last {
            buf.drain(..(it.offset_samples() - start) as usize);
        }
    }
    if noise_gate.is_some() {
        let windows = (received as usize).div_ceil(if sample_rate == 16_000 { 512 } else { 256 });
        println!("noise gate: {gated} of {windows} windows gated, not run through the model");
    }
    Ok(())
}
