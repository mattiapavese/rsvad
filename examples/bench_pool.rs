// Runs N concurrent real-time streams (the wav file played in loop, one chunk every chunk_ms)
// on the sessions pool and reports how long vad() calls take. Use it to tune
// max_streams_per_session on the target hardware.
//
//   cargo run --release --example bench_pool -- <libonnxruntime> <silero_vad.onnx> <file.wav> \
//       [streams=64] [max_streams_per_session=16] [chunk_ms=100] [seconds=20]

use std::sync::Arc;
use std::time::{Duration, Instant};

use rsvad::{VadEnvConfig, VadIterator, initialize, pool_stats};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <libonnxruntime> <silero_vad.onnx> <file.wav> [streams=64] [max_streams_per_session=16] [chunk_ms=100] [seconds=20]",
            args[0]
        );
        std::process::exit(2);
    }
    let arg = |i: usize, default: usize| args.get(i).map_or(Ok(default), |s| s.parse::<usize>());
    let (streams, per_session, chunk_ms, seconds) = (arg(4, 64)?, arg(5, 16)?, arg(6, 100)?, arg(7, 20)?);

    let config = VadEnvConfig { max_streams_per_session: per_session, ..Default::default() };
    let t0 = Instant::now();
    initialize(&args[1], &args[2], Some(config))?;
    let init_time = t0.elapsed();

    let mut reader = hound::WavReader::open(&args[3])?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("need a mono 16 bit wav".into());
    }
    let sample_rate = spec.sample_rate as i64;
    let audio: Arc<Vec<f32>> =
        Arc::new(reader.samples::<i16>().map(|s| s.map(|s| s as f32 / 32768.0)).collect::<Result<_, _>>()?);

    let chunk_len = chunk_ms * sample_rate as usize / 1000;
    let n_chunks = seconds * 1000 / chunk_ms;

    println!("{streams} streams, {per_session} streams per session, {chunk_ms}ms chunks, {seconds}s");
    println!("initialize (loads the library, creates the first session): {:.1}ms", init_time.as_secs_f64() * 1000.0);

    let started = Instant::now();
    let mut tasks = Vec::new();
    for k in 0..streams {
        let audio = Arc::clone(&audio);
        tasks.push(tokio::spawn(async move {
            // streams start spread over one chunk, not all in the same instant
            tokio::time::sleep(Duration::from_millis((k * chunk_ms / streams) as u64)).await;
            let mut it = VadIterator::new(sample_rate, None).unwrap();
            let mut ticker = tokio::time::interval(Duration::from_millis(chunk_ms as u64));
            let mut latencies = Vec::with_capacity(n_chunks);
            let mut buf: Vec<f32> = Vec::new();
            let mut pos = (k * 7919 * chunk_len) % audio.len(); // each stream at a different point of the file
            for _ in 0..n_chunks {
                ticker.tick().await;
                for _ in 0..chunk_len {
                    buf.push(audio[pos]);
                    pos = (pos + 1) % audio.len();
                }
                let start = it.offset_samples();
                let t0 = Instant::now();
                it.vad(&buf).await.unwrap();
                latencies.push((t0.duration_since(started), t0.elapsed()));
                buf.drain(..(it.offset_samples() - start) as usize);
            }
            it.flush(&buf).await.unwrap();
            latencies
        }));
    }

    let mut peak = pool_stats().unwrap();
    while !tasks.iter().all(|t| t.is_finished()) {
        let s = pool_stats().unwrap();
        if s.sessions > peak.sessions {
            peak = s;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let mut all = Vec::new();
    for t in tasks {
        all.extend(t.await?);
    }

    // the first second has every stream starting at once, and the sessions they need being created
    let warmup = Duration::from_secs(1);
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let report = |name: &str, mut latencies: Vec<Duration>| {
        if latencies.is_empty() {
            return;
        }
        latencies.sort();
        let pct = |p: f64| latencies[((latencies.len() - 1) as f64 * p) as usize];
        println!(
            "{name}: {} vad() calls  p50 {:.2}ms  p90 {:.2}ms  p99 {:.2}ms  max {:.2}ms",
            latencies.len(),
            ms(pct(0.5)),
            ms(pct(0.9)),
            ms(pct(0.99)),
            ms(*latencies.last().unwrap())
        );
    };
    report("first second", all.iter().filter(|(at, _)| *at < warmup).map(|&(_, l)| l).collect());
    report("steady state", all.iter().filter(|(at, _)| *at >= warmup).map(|&(_, l)| l).collect());
    println!("sessions at peak: {}  wall time {:.1}s", peak.sessions, started.elapsed().as_secs_f64());
    Ok(())
}
