use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model,
    training_observer::TrainingObserver,
    validate_model, Sample, TrainingOptions,
};
use rand::{thread_rng, Rng};

mod classification {
    use super::*;
    /// Build a model and train it on the function f(x, y, z) = x + y + z > 0. Tests that the trained validation loss is less than the untrained validation loss
    #[test]
    fn classifier_sum_greater_than_zero() {
        // select 10000 random x's, y's, and z's in the range -1000 to 1000 to train on, and 100 random x's, y's, and z's in the range -1000 to 1000 to validate on
        let training_data = (0..10000)
            .map(|_| {
                let x = thread_rng().gen_range(-1000.0..1000.0);
                let y = thread_rng().gen_range(-1000.0..1000.0);
                let z = thread_rng().gen_range(-1000.0..1000.0);
                let label = ((x + y + z) > 0.0) as u32;
                Sample::new(vec![x, y, z], label as f32)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(-1000.0..1000.0);
                let y = thread_rng().gen_range(-1000.0..1000.0);
                let z = thread_rng().gen_range(-1000.0..1000.0);
                let label = ((x + y + z) > 0.0) as u32;
                Sample::new(vec![x, y, z], label as f32)
            })
            .collect::<Vec<Sample>>();

        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 3,
            layer_sizes: vec![2],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
            class_map: None,
        });
        let untrained_validation_loss =
            validate_model(&validation_data, &mut untrained_model, &TestObserver::new());
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            fekan::EachEpoch::ValidateModel(&validation_data),
            &TestObserver::new(),
            TrainingOptions {
                num_epochs: 50,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss =
            validate_model(&validation_data, &mut trained_model, &TestObserver::new());
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
    );
    }
}
mod regression {
    use super::*;
    /// build a model and train it on the function f(x, y) = xy
    #[test]
    fn regressor_xy() {
        // select 1000 random points in the range -1000 to 1000 to train on, and 100 random points in the range -1000 to 1000 to validate on
        let mut rand = rand::thread_rng();
        let mut training_data = Vec::with_capacity(1000);
        let mut validation_data = Vec::with_capacity(100);
        for _ in 0..1000 {
            let x = rand.gen_range(-1000.0..1000.0);
            let y = rand.gen_range(-1000.0..1000.0);
            training_data.push(Sample::new(vec![x, y], x * y));
        }
        for _ in 0..100 {
            let x = rand.gen_range(-1000.0..1000.0);
            let y = rand.gen_range(-1000.0..1000.0);
            validation_data.push(Sample::new(vec![x, y], x * y));
        }
        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 2,
            layer_sizes: vec![3, 2, 1],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Regression,
            class_map: None,
        });
        let untrained_validation_loss =
            validate_model(&validation_data, &mut untrained_model, &TestObserver::new());
        let training_result = train_model(
            untrained_model,
            &training_data,
            fekan::EachEpoch::ValidateModel(&validation_data), // this way if the test fails, we can see the validation loss over time
            &TestObserver::new(),
            TrainingOptions {
                num_epochs: 50,
                ..TrainingOptions::default()
            },
        );
        if let Err(e) = training_result {
            panic!("Error training model: {:#?}", e);
        }
        let mut trained_model = training_result.unwrap();
        let validation_loss =
            validate_model(&validation_data, &mut trained_model, &TestObserver::new());
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
    );
    }

    /// build a model and train it on the function f(x, y) = e^(sin(pi*x) * y^2)
    #[test]
    fn exp_sin_pix_y_squared() {
        let training_data = (0..1000)
            .map(|_| {
                let x = thread_rng().gen_range(-1.0..1.0);
                let y = thread_rng().gen_range(-1.0..1.0);
                let label = ((std::f32::consts::PI * x).sin() * (y * y)).exp();
                Sample::new(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(-1.0..1.0);
                let y = thread_rng().gen_range(-1.0..1.0);
                let label = ((std::f32::consts::PI * x).sin() * (y * y)).exp();
                Sample::new(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 2,
            layer_sizes: vec![3, 2, 1],
            degree: 3,
            coef_size: 5,
            model_type: ModelType::Regression,
            class_map: None,
        });

        let untrained_validation_loss =
            validate_model(&validation_data, &mut untrained_model, &TestObserver::new());
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            fekan::EachEpoch::ValidateModel(&validation_data),
            &TestObserver::new(),
            TrainingOptions {
                num_epochs: 50,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss =
            validate_model(&validation_data, &mut trained_model, &TestObserver::new());
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
    );
    }
}
struct TestObserver {}

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
