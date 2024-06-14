pub trait TrainingObserver {
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f32, validation_loss: f32);

    fn on_sample_end(&self);
}
