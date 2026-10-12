use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::core::OrtEngine;
use crate::error::VadError;
use crate::pool::{EngineFactory, InferenceEngine, PoolStats, SessionPool, VadEnvConfig};
use crate::utils::lock;

struct OrtEnv {
    // the ort environment (and the shared library loaded with it) is process wide:
    // once committed it lives as long as the process, finalize() only drops the pool
    env_committed: bool,
    pool: Option<Arc<SessionPool>>,
}

static ORT_ENV: Mutex<OrtEnv> = Mutex::new(OrtEnv {
    env_committed: false,
    pool: None,
});

pub(crate) const CHUNK_LEN_MS: i64 = 32;

/// Loads the ONNX Runtime shared library and the Silero VAD model, and starts the sessions pool
/// shared by all the iterators created afterwards. The first session is created here, so a wrong
/// library or model fails right away; the next ones are created as streams need them, see
/// [`VadEnvConfig`]. `None` config means the defaults. Calling it again while initialized does nothing.
///
/// The shared library can be loaded only once per process: after [`finalize`], `initialize`
/// reuses the library loaded the first time and ignores `onnx_shared_lib_path`.
pub fn initialize(
    onnx_shared_lib_path: impl AsRef<Path>,
    onnx_silero_vad_model_path: impl AsRef<Path>,
    config: Option<VadEnvConfig>,
) -> Result<(), VadError> {
    let mut env = lock(&ORT_ENV);

    if env.pool.is_some() {
        return Ok(());
    }

    if !env.env_committed {
        ort::init_from(onnx_shared_lib_path.as_ref())
            .map_err(VadError::LoadOrt)?
            .commit();
        env.env_committed = true;
    }

    // read once: every session of the pool is built from these bytes
    let model: Arc<[u8]> = std::fs::read(onnx_silero_vad_model_path)
        .map_err(VadError::ModelRead)?
        .into();
    let factory: EngineFactory =
        Arc::new(move || Ok(Box::new(OrtEngine::new(&model)?) as Box<dyn InferenceEngine>));

    env.pool = Some(SessionPool::new(config.unwrap_or_default(), factory)?);
    Ok(())
}

/// Releases the sessions pool. Iterators created before keep working on their sessions, which
/// close when the last of them is dropped; creating new iterators requires calling [`initialize`] again.
pub fn finalize() {
    lock(&ORT_ENV).pool = None;
}

/// Sessions and streams in the pool, `None` if not initialized.
pub fn pool_stats() -> Option<PoolStats> {
    lock(&ORT_ENV).pool.as_ref().map(|pool| pool.stats())
}

pub(crate) fn shared_pool() -> Option<Arc<SessionPool>> {
    lock(&ORT_ENV).pool.clone()
}
