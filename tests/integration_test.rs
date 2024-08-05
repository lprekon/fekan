use fekan::{
    kan::{Kan, KanOptions, ModelType},
    preset_knot_ranges, train_model,
    training_options::TrainingOptions,
    validate_model, Sample,
};

use rand::{thread_rng, Rng};

mod classification {
    use super::*;
    use test_log::test;
    /// Build a model and train it on the function f(x, y, z) = x + y + z > 0. Tests that the trained validation loss is less than the untrained validation loss
    #[test]
    fn classifier_sum_greater_than_zero() {
        // select 10000 random x's, y's, and z's in the range -1000 to 1000 to train on, and 100 random x's, y's, and z's in the range -1000 to 1000 to validate on
        let function_domain = -100.0..100.0;
        let training_data = (0..1000)
            .map(|_| {
                let x = thread_rng().gen_range(function_domain.clone());
                let y = thread_rng().gen_range(function_domain.clone());
                let z = thread_rng().gen_range(function_domain.clone());
                let label = ((x + y + z) > 0.0) as u32;
                Sample::new(vec![x, y, z], label as f64)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(function_domain.clone());
                let y = thread_rng().gen_range(function_domain.clone());
                let z = thread_rng().gen_range(function_domain.clone());
                let label = ((x + y + z) > 0.0) as u32;
                Sample::new(vec![x, y, z], label as f64)
            })
            .collect::<Vec<Sample>>();

        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 3,
            layer_sizes: vec![3, 2],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
            class_map: None,
        });

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        preset_knot_ranges(&mut untrained_model, &training_data).unwrap();
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_epochs: 100,
                num_threads: 8,
                each_epoch: fekan::training_options::EachEpoch::ValidateModel(&validation_data),
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );
        for i in 0..100 {
            let _ = trained_model.forward(training_data[i].features().clone());
        }
        trained_model.test_and_set_symbolic(0.98);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss <= validation_loss,
            "Symbolification did not improve loss. Before {}, After {}. ",
            validation_loss,
            symbolic_loss,
        );
    }
}
mod regression {
    use super::*;
    use test_log::test;
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
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        preset_knot_ranges(&mut untrained_model, &training_data).unwrap();
        let training_result = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_epochs: 50,
                num_threads: 8,
                each_epoch: fekan::training_options::EachEpoch::ValidateModel(&validation_data), // this way if the test fails, we can see the validation loss over time
                ..TrainingOptions::default()
            },
        );
        if let Err(e) = training_result {
            panic!("Error training model: {:#?}", e);
        }
        let mut trained_model = training_result.unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );

        for i in 0..100 {
            let _ = trained_model.forward(training_data[i].features().clone());
        }
        trained_model.test_and_set_symbolic(0.95);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss <= validation_loss,
            "Symbolification did not improve loss. Before {}, After {}.",
            validation_loss,
            symbolic_loss,
        );
    }

    /// build a model and train it on the function f(x, y) = e^(sin(pi*x) * y^2)
    #[test]
    fn exp_sin_pix_y_squared() {
        let training_data = (0..1000)
            .map(|_| {
                let x = thread_rng().gen_range(-1.0..1.0);
                let y = thread_rng().gen_range(-1.0..1.0);
                let label = ((std::f64::consts::PI * x).sin() * (y * y)).exp();
                Sample::new(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(-1.0..1.0);
                let y = thread_rng().gen_range(-1.0..1.0);
                let label = ((std::f64::consts::PI * x).sin() * (y * y)).exp();
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

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        preset_knot_ranges(&mut untrained_model, &training_data).unwrap();
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_epochs: 50,
                num_threads: 8,
                each_epoch: fekan::training_options::EachEpoch::ValidateModel(&validation_data),
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );

        for i in 0..100 {
            let _ = trained_model.forward(training_data[i].features().clone());
        }
        trained_model.test_and_set_symbolic(0.95);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss <= validation_loss,
            "Symbolification did not improve loss. Before {}, After {}",
            validation_loss,
            symbolic_loss,
        );
    }

    #[test]
    fn sin_x() {
        let input_range = 0.0..6.0;

        fn true_function(x: f64) -> f64 {
            4.0 * x.sin()
        }

        let training_data = (0..1000)
            .map(|_| {
                let x: f64 = thread_rng().gen_range(input_range.clone());
                let label = true_function(x);
                Sample::new(vec![x], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x: f64 = thread_rng().gen_range(input_range.clone());
                let label = true_function(x);
                Sample::new(vec![x], label)
            })
            .collect::<Vec<Sample>>();
        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 1,
            layer_sizes: vec![1],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
        });
        // if log::log_enabled!(log::Level::Trace) {
        //     let mut training_samples = training_data.clone();
        //     let mut validation_samples = validation_data.clone();
        //     training_samples.sort_by(|a, b| a.features()[0].partial_cmp(&b.features()[0]).unwrap());
        //     validation_samples
        //         .sort_by(|a, b| a.features()[0].partial_cmp(&b.features()[0]).unwrap());
        //     let training_inputs: Vec<f64> =
        //         training_samples.iter().map(|s| s.features()[0]).collect();
        //     let training_outputs: Vec<f64> = training_samples.iter().map(|s| s.label()).collect();
        //     let validation_inputs: Vec<f64> =
        //         validation_samples.iter().map(|s| s.features()[0]).collect();
        //     let validation_outputs: Vec<f64> =
        //         validation_samples.iter().map(|s| s.label()).collect();
        //     log::trace!(
        //         "Training inputs: {:?}\nTraining outputs: {:?}",
        //         training_inputs,
        //         training_outputs
        //     );
        //     log::trace!(
        //         "Validation inputs: {:?}\nValidation outputs: {:?}",
        //         validation_inputs,
        //         validation_outputs
        //     );
        // }

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 8,
                knot_update_interval: 1001,
                num_epochs: 500,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::ValidateModel(&validation_data),
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );
        trained_model.test_and_set_symbolic(0.99);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss < validation_loss,
            "Symbolification did not improve loss. Before {}, After {}",
            validation_loss,
            symbolic_loss,
        );
    }

    #[test]
    fn x_cubed() {
        let input_range = -2.0..2.0;

        fn true_function(x: f64) -> f64 {
            -0.5 * (x - 4.0).powi(3) + 2.0
        }

        let training_data = (0..1000)
            .map(|_| {
                let x: f64 = thread_rng().gen_range(input_range.clone());
                let label = true_function(x);
                Sample::new(vec![x], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x: f64 = thread_rng().gen_range(input_range.clone());
                let label = true_function(x);
                Sample::new(vec![x], label)
            })
            .collect::<Vec<Sample>>();
        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 1,
            layer_sizes: vec![1],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
        });
        // if log::log_enabled!(log::Level::Trace) {
        //     let mut training_samples = training_data.clone();
        //     let mut validation_samples = validation_data.clone();
        //     training_samples.sort_by(|a, b| a.features()[0].partial_cmp(&b.features()[0]).unwrap());
        //     validation_samples
        //         .sort_by(|a, b| a.features()[0].partial_cmp(&b.features()[0]).unwrap());
        //     let training_inputs: Vec<f64> =
        //         training_samples.iter().map(|s| s.features()[0]).collect();
        //     let training_outputs: Vec<f64> = training_samples.iter().map(|s| s.label()).collect();
        //     let validation_inputs: Vec<f64> =
        //         validation_samples.iter().map(|s| s.features()[0]).collect();
        //     let validation_outputs: Vec<f64> =
        //         validation_samples.iter().map(|s| s.label()).collect();
        //     log::trace!(
        //         "Training inputs: {:?}\nTraining outputs: {:?}",
        //         training_inputs,
        //         training_outputs
        //     );
        //     log::trace!(
        //         "Validation inputs: {:?}\nValidation outputs: {:?}",
        //         validation_inputs,
        //         validation_outputs
        //     );
        // }

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 8,
                knot_update_interval: 1001,
                num_epochs: 500,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::ValidateModel(&validation_data),
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );
        trained_model.test_and_set_symbolic(0.99);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss < validation_loss,
            "Symbolification did not improve loss. Before {}, After {}",
            validation_loss,
            symbolic_loss,
        );
    }

    #[test]
    fn root_x() {
        let input_range = -12.0..2.99;

        fn true_function(x: f64) -> f64 {
            -1.0 * (-1.0 * x + 3.0).sqrt() - 4.0
        }

        let training_data = (0..1000)
            .map(|_| {
                let x: f64 = thread_rng().gen_range(input_range.clone());
                let label = true_function(x);
                Sample::new(vec![x], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x: f64 = thread_rng().gen_range(input_range.clone());
                let label = true_function(x);
                Sample::new(vec![x], label)
            })
            .collect::<Vec<Sample>>();
        let mut untrained_model = Kan::new(&KanOptions {
            input_size: 1,
            layer_sizes: vec![1],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
        });
        // if log::log_enabled!(log::Level::Trace) {
        //     let mut training_samples = training_data.clone();
        //     let mut validation_samples = validation_data.clone();
        //     training_samples.sort_by(|a, b| a.features()[0].partial_cmp(&b.features()[0]).unwrap());
        //     validation_samples
        //         .sort_by(|a, b| a.features()[0].partial_cmp(&b.features()[0]).unwrap());
        //     let training_inputs: Vec<f64> =
        //         training_samples.iter().map(|s| s.features()[0]).collect();
        //     let training_outputs: Vec<f64> = training_samples.iter().map(|s| s.label()).collect();
        //     let validation_inputs: Vec<f64> =
        //         validation_samples.iter().map(|s| s.features()[0]).collect();
        //     let validation_outputs: Vec<f64> =
        //         validation_samples.iter().map(|s| s.label()).collect();
        //     log::trace!(
        //         "Training inputs: {:?}\nTraining outputs: {:?}",
        //         training_inputs,
        //         training_outputs
        //     );
        //     log::trace!(
        //         "Validation inputs: {:?}\nValidation outputs: {:?}",
        //         validation_inputs,
        //         validation_outputs
        //     );
        // }

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 8,
                knot_update_interval: 1001,
                num_epochs: 500,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::ValidateModel(&validation_data),
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );
        trained_model.test_and_set_symbolic(0.99);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss < validation_loss,
            "Symbolification did not improve loss. Before {}, After {}",
            validation_loss,
            symbolic_loss,
        );
        assert!(false);
    }
}
