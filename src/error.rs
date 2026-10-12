use std::fmt;

#[derive(Debug)]
pub enum VadError {
    UnsupportedSampleRate(i64),
    InvalidParams(String),
    /// initialize() was not called, or finalize() was called after it
    NotInitialized,
    LoadOrt(ort::LoadDynamicError),
    ModelRead(std::io::Error),
    Ort(ort::Error),
    /// A session of the pool could not be created
    SessionFailed(String),
    /// A session's thread is gone
    SessionClosed,
}

impl VadError {
    // the session failed, not the request: it can be retried on another session
    pub(crate) fn is_session_failure(&self) -> bool {
        matches!(self, VadError::SessionFailed(_) | VadError::SessionClosed)
    }
}

impl fmt::Display for VadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VadError::UnsupportedSampleRate(sr) => write!(f, "vad: unsupported sample rate: {sr}"),
            VadError::InvalidParams(msg) => write!(f, "vad: invalid params: {msg}"),
            VadError::NotInitialized => write!(f, "vad: initialize must be called before creating an iterator"),
            VadError::LoadOrt(e) => write!(f, "failed to load onnx runtime shared library: {e}"),
            VadError::ModelRead(e) => write!(f, "failed to read silero vad model: {e}"),
            VadError::Ort(e) => write!(f, "ort vad: {e}"),
            VadError::SessionFailed(msg) => write!(f, "vad: session failed: {msg}"),
            VadError::SessionClosed => write!(f, "vad: session closed"),
        }
    }
}

impl std::error::Error for VadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VadError::Ort(e) => Some(e),
            VadError::ModelRead(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ort::Error> for VadError {
    fn from(e: ort::Error) -> Self {
        VadError::Ort(e)
    }
}

impl From<ort::Error<ort::session::builder::SessionBuilder>> for VadError {
    fn from(e: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        VadError::Ort(e.into())
    }
}
