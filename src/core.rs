use ort::session::Session;
use ort::value::TensorRef;
use tokio::sync::oneshot;

use crate::error::VadError;
use crate::gate::{GateState, NoiseGate};
use crate::iterator::SpeechSegment;
use crate::pool::{InferRequest, InferenceEngine, SessionSlot, WindowsOutput};
use crate::utils::get_window_sz_and_context_sz;

const STATE_DIM: usize = 128;

// A long input (e.g. a backlog after a network stall) is sent to the session in more requests,
// so that the other streams sharing the session do not wait for all of it. 32 windows ~ 1s of audio.
const MAX_WINDOWS_PER_REQUEST: usize = 32;

// probs, states, contexts, gate states, gated
pub(crate) type InferenceOutput = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<GateState>, Vec<bool>);

pub(crate) struct VadCore {
    pub(crate) sample_rate: i64,

    pub(crate) slot: Option<SessionSlot>, // None only for iterators built by the tests without a pool

    pub(crate) state: Vec<f32>,   // shape (2, 1, STATE_DIM)
    pub(crate) context: Vec<f32>, // shape (1, context size)

    pub(crate) gate: Option<NoiseGate>, // None: every window is run through the model
    pub(crate) gate_state: GateState,

    request_buffer: Vec<f32>, // the audio of a request, reused across requests
}

impl VadCore {
    pub(crate) fn new(sample_rate: i64, gate: Option<NoiseGate>, slot: Option<SessionSlot>) -> Self {
        let mut c = VadCore {
            sample_rate,
            slot,
            state: Vec::new(),
            context: Vec::new(),
            gate,
            gate_state: GateState::default(),
            request_buffer: Vec::new(),
        };
        c.reset_model_state();
        c
    }

    pub(crate) fn reset_model_state(&mut self) {
        let (_, context_size) = get_window_sz_and_context_sz(self.sample_rate);
        self.state = vec![0.0; 2 * STATE_DIM];
        self.context = vec![0.0; context_size as usize];
        self.gate_state = GateState::default();
    }

    /// Runs the gate (if any) and the model over every full window of `input_buffer`, starting from
    /// `self.state`, `self.context` and `self.gate_state`, which are left untouched: the caller
    /// decides which state to keep. If the future is dropped before completing, nothing changes.
    ///
    /// Returns one probability per window, and the model state, context and gate state *before* each
    /// window plus the ones after the last window (`probs.len() + 1` entries), so that VadIterator's
    /// lookback can rewind to any window boundary. Also returns which windows the gate closed on:
    /// those are not run through the model, their probability is 0.
    pub(crate) async fn inference_loop(&mut self, input_buffer: &[f32]) -> Result<InferenceOutput, VadError> {
        let (window_size, context_size) = get_window_sz_and_context_sz(self.sample_rate);
        let (window_size, context_size) = (window_size as usize, context_size as usize);
        let n_windows = input_buffer.len() / window_size;
        let audio = &input_buffer[..n_windows * window_size];

        // the gate runs here, before the model: only the windows it leaves open go to the session
        let (gated, gate_states) = match &self.gate {
            Some(gate) => gate.run(self.gate_state, audio, window_size),
            None => (vec![false; n_windows], vec![self.gate_state; n_windows + 1]),
        };

        let mut probs = Vec::with_capacity(n_windows);
        let mut states = Vec::with_capacity(n_windows + 1);
        let mut contexts = Vec::with_capacity(n_windows + 1);

        states.push(self.state.clone());
        contexts.push(self.context.clone());

        let mut k = 0;
        while k < n_windows {
            if gated[k] {
                // non-speech without running the model. The model state is kept as it is: resetting it
                // changes the probabilities after the gate opens again much more than keeping it.
                // The context is just the audio before the next window
                let window_end = (k + 1) * window_size;
                probs.push(0.0);
                states.push(states.last().unwrap().clone());
                contexts.push(audio[window_end - context_size..window_end].to_vec());
                k += 1;
                continue;
            }

            // the windows up to the next one gated, in requests of at most MAX_WINDOWS_PER_REQUEST
            let mut end = k + 1;
            while end < n_windows && end - k < MAX_WINDOWS_PER_REQUEST && !gated[end] {
                end += 1;
            }
            let piece = &audio[k * window_size..end * window_size];
            k = end;

            let state: &Vec<f32> = states.last().unwrap();
            let context: &Vec<f32> = contexts.last().unwrap();

            let mut moved = false;
            let reply = loop {
                let slot = self.slot.as_mut().ok_or(VadError::NotInitialized)?;

                // the audio is copied in the request: the session's thread may still be reading it
                // when this future is dropped, so it cannot borrow the caller's buffer
                let mut audio = std::mem::take(&mut self.request_buffer);
                audio.clear();
                audio.extend_from_slice(piece);

                let (reply_tx, reply_rx) = oneshot::channel();
                let req = InferRequest {
                    sample_rate: self.sample_rate,
                    audio,
                    state: state.clone(),
                    context: context.clone(),
                    reply: reply_tx,
                };
                let res = match slot.send(req) {
                    Ok(()) => reply_rx.await.unwrap_or(Err(VadError::SessionClosed)),
                    Err(e) => Err(e),
                };

                match res {
                    // the session failed: the request carries all the stream's state, so it can
                    // simply go to another session
                    Err(e) if e.is_session_failure() && !moved => {
                        slot.reassign();
                        moved = true;
                    }
                    res => break res?,
                }
            };

            self.request_buffer = reply.audio;
            probs.extend(reply.output.probs);
            states.extend(reply.output.states);
            contexts.extend(reply.output.contexts);
        }

        Ok((probs, states, contexts, gate_states, gated))
    }
}

/// The model, run by a session's thread.
pub(crate) struct OrtEngine {
    session: Session,
}

impl OrtEngine {
    pub(crate) fn new(model: &[u8]) -> Result<Self, VadError> {
        // one thread per run: the model is tiny and runs window by window, a thread pool only adds overhead
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_memory(model)?;
        Ok(OrtEngine { session })
    }
}

impl InferenceEngine for OrtEngine {
    fn run(&mut self, sample_rate: i64, audio: &[f32], state: &[f32], context: &[f32]) -> Result<WindowsOutput, VadError> {
        let (window_size, context_size) = get_window_sz_and_context_sz(sample_rate);
        let (window_size, context_size) = (window_size as usize, context_size as usize);
        let sr = [sample_rate];

        let mut out = WindowsOutput::default();
        let mut tmp_state = state.to_vec();
        let mut tmp_context = context.to_vec();
        let mut input = Vec::with_capacity(context_size + window_size);

        for window in audio.chunks_exact(window_size) {
            // the model input is the context followed by the window
            input.clear();
            input.extend_from_slice(&tmp_context);
            input.extend_from_slice(window);

            let input_ort = TensorRef::from_array_view(([1usize, input.len()], input.as_slice()))?;
            let state_ort = TensorRef::from_array_view(([2usize, 1, STATE_DIM], tmp_state.as_slice()))?;
            let sr_ort = TensorRef::from_array_view(([1usize], &sr[..]))?;

            let outputs = self.session.run(ort::inputs![
                "input" => input_ort,
                "state" => state_ort,
                "sr" => sr_ort,
            ])?;

            let (_, output) = outputs["output"].try_extract_tensor::<f32>()?;
            let (_, state_n) = outputs["stateN"].try_extract_tensor::<f32>()?;
            out.probs.push(output[0]);
            tmp_state = state_n.to_vec();
            drop(outputs);

            tmp_context = input[input.len() - context_size..].to_vec();

            out.states.push(tmp_state.clone());
            out.contexts.push(tmp_context.clone());
        }

        Ok(out)
    }
}

/// Pads each segment on both sides, clamps to [lo, hi] and merges the segments
/// that end up overlapping or touching. Two segments separated by up to 2*pad_samples
/// of silence are therefore merged, preserving the silence in the middle.
/// Segments left empty by the clamping are dropped.
pub(crate) fn pad_and_merge(segs: &[SpeechSegment], pad_samples: i64, lo: i64, hi: i64) -> Vec<SpeechSegment> {
    let mut merged: Vec<SpeechSegment> = Vec::with_capacity(segs.len());
    for seg in segs {
        let start = (seg.start - pad_samples).max(lo);
        let end = (seg.end + pad_samples).min(hi);
        if end <= start {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && start <= last.end
        {
            last.end = last.end.max(end);
            continue;
        }
        merged.push(SpeechSegment { start, end });
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: i64, end: i64) -> SpeechSegment {
        SpeechSegment { start, end }
    }

    #[test]
    fn pad_and_merge_pads_clamps_merges() {
        // padded to [0,300] [600,1100], clamped to [100, 1000]
        assert_eq!(pad_and_merge(&[seg(100, 200), seg(700, 1000)], 100, 100, 1000), vec![seg(100, 300), seg(600, 1000)]);
        // touching after padding -> merged
        assert_eq!(pad_and_merge(&[seg(100, 200), seg(400, 500)], 100, 0, 1000), vec![seg(0, 600)]);
        // empty after clamping -> dropped
        assert_eq!(pad_and_merge(&[seg(500, 500)], 0, 0, 1000), vec![]);
        assert_eq!(pad_and_merge(&[seg(100, 200)], 50, 300, 1000), vec![]);
        assert_eq!(pad_and_merge(&[], 50, 0, 1000), vec![]);
    }
}
