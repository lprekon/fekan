use fekan::training_observer::TrainingObserver;
pub struct TestObserver {}

impl TestObserver {
    pub fn new() -> Self {
        TestObserver {}
    }
}

impl TrainingObserver for TestObserver {
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f32, validation_loss: f32) {
        println!(
            "Epoch: {}, Epoch Loss: {}, Validation Loss: {}",
            epoch, epoch_loss, validation_loss
        );
    }

    fn on_sample_end(&self) {
        // do nothing
    }
}
