/// Structs implementing this trait can be used to observe the training process .
pub trait TrainingObserver {
    /// called at the end of each epoch of training with the epoch count, the training loss of the epoch, and the validation loss of the epoch
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f32, validation_loss: f32);

    /// called at the end of each sample of training. Useful for logging or debugging
    fn on_sample_end(&self);
}
