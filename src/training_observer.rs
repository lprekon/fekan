/// Structs implementing this trait can be used to observe the training process.
pub trait TrainingObserver {
    /// called by [`crate::train_model`] at the end of each epoch. validation loss will be NaN if the training option field `each_epoch` is set to [`DoNotValidateModel`](crate::training_options::EachEpoch).
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f64, validation_loss: f64);

    /// called by [`crate::train_model`] when the knots are extended. useful for understanding changes in the model.
    fn on_knot_extension(&self, old_length: usize, new_length: usize);
}
