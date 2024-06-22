/// Structs implementing this trait can be used to observe the training process .
pub trait TrainingObserver {
    /// called by [`crate::train_model`] at the end of each epoch. validation loss will be NaN if no validation set is provided.
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f32, validation_loss: f32);

    /// called by [`crate::train_model`] at the end of each sample. useful for logging or benchmarking.
    fn on_sample_end(&self);
}
