use crate::kan::kan_error::KanError;

/// Indicates that an error was encountered during training
///
/// If displayed, this error will show the epoch and sample at which the error was encountered, as well as the [KanError] that caused the error.
#[derive(Clone, PartialEq, Debug)]
pub struct TrainingError {
    /// The error that caused the training error
    pub source: KanError,
    /// The epoch at which the error was encountered
    pub epoch: usize,
    /// The sample within the epoch at which the error was encountered
    pub sample: usize,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "epoch {} sample {} encountered error {}",
            self.epoch, self.sample, self.source
        )
    }
}

impl std::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
