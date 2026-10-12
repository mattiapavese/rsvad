use crate::iterator::{DecisionState, VadIterator};

impl VadIterator {
    /// Brings the iterator back to the start of a new stream.
    pub fn reset(&mut self) {
        self.core.reset_model_state();
        self.decision = DecisionState::default();
        self.offset_samples = 0;
        self.emitted_until = None;
    }
}
