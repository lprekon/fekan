use fekan::{
    kan::{Kan, KanOptions, ModelType},
    preset_knot_ranges, train_model,
    training_options::TrainingOptions,
    validate_model, Sample,
};

use rand::{thread_rng, Rng};

mod classification {
    use super::*;
    use fekan::embedding_layer::EmbeddingOptions;
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
                Sample::new_classification_sample(vec![x, y, z], label as usize)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(function_domain.clone());
                let y = thread_rng().gen_range(function_domain.clone());
                let z = thread_rng().gen_range(function_domain.clone());
                let label = ((x + y + z) > 0.0) as u32;
                Sample::new_classification_sample(vec![x, y, z], label as usize)
            })
            .collect::<Vec<Sample>>();

        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 3,
            layer_sizes: vec![3, 2],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
            class_map: None,
            embedding_options: None,
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
        // for i in 0..100 {
        //     let _ = trained_model.forward(training_data[i].features().clone());
        // }
        // trained_model.test_and_set_symbolic(0.98);
        // let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        // assert!(
        //     symbolic_loss <= validation_loss,
        //     "Symbolification did not improve loss. Before {}, After {}. ",
        //     validation_loss,
        //     symbolic_loss,
        // );
    }

    #[test]
    fn even_or_odd() {
        let function_domain = 0..10;
        fn true_function(x: f64) -> f64 {
            (x % 2.0).round()
        }
        let training_data: Vec<Sample> = function_domain
            .clone()
            .map(|x| {
                Sample::new_classification_sample(vec![x as f64], true_function(x as f64) as usize)
            })
            .collect();
        let untrained_model = Kan::new(&KanOptions {
            num_features: 1,
            layer_sizes: vec![2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Classification,
            class_map: None,
            embedding_options: Some(EmbeddingOptions {
                embedded_features: vec![0],
                vocab_size: 10,
                embedding_dimension: 2,
                full_input_dimension: 1,
            }),
        });
        let initialization_loss = validate_model(&training_data, &untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_epochs: 300,
                num_threads: 8,
                learning_rate: 0.001,
                l1_penalty: 0.0,
                entropy_penalty: 0.0,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let trained_loss = validate_model(&training_data, &trained_model);
        assert!(
            trained_loss < initialization_loss,
            "Training did not improve loss. Before {}, After {}",
            initialization_loss,
            trained_loss
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
            training_data.push(Sample::new_regression_sample(vec![x, y], x * y));
        }
        for _ in 0..100 {
            let x = rand.gen_range(-1000.0..1000.0);
            let y = rand.gen_range(-1000.0..1000.0);
            validation_data.push(Sample::new_regression_sample(vec![x, y], x * y));
        }
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 2,
            layer_sizes: vec![3, 2, 1],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
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
            panic!("Error training model: {}", e);
        }
        let mut trained_model = training_result.unwrap();
        let validation_loss = validate_model(&validation_data, &mut trained_model);
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
        fn true_function(x: f64, y: f64) -> f64 {
            ((std::f64::consts::PI * x).sin() + y.powi(2)).exp()
        }
        let training_data = (0..1000)
            .map(|_| {
                let x = thread_rng().gen_range(-1.0..1.0);
                let y = thread_rng().gen_range(-1.0..1.0);
                let label = true_function(x, y);
                Sample::new_regression_sample(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(-1.0..1.0);
                let y = thread_rng().gen_range(-1.0..1.0);
                let label = true_function(x, y);
                Sample::new_regression_sample(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 2,
            layer_sizes: vec![5, 1],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_epochs: 100,
                num_threads: 8,
                learning_rate: 0.0005,
                l1_penalty: 0.5,
                entropy_penalty: 0.5,
                batch_size: 125,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
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
        trained_model.test_and_set_symbolic(0.95);
        let symbolic_loss = validate_model(&validation_data, &mut trained_model);
        assert!(
            symbolic_loss - validation_loss < 0.01,
            "Symbolification significantly degraded loss. Before {}, After {}",
            validation_loss,
            symbolic_loss,
        );
        // let pruning_results = trained_model.prune(1e-2); // I've decided to remove this for now, as it's not really necessary for the test
        // assert_ne!(pruning_results, vec![]);
    }

    #[test]
    fn prune_an_edge() {
        fn true_function(x: f64, _y: f64) -> f64 {
            x.sin()
        }
        let x_range = 0.0..6.0;
        let y_range = -1.0..1.0;
        let training_data = (0..10000)
            .map(|_| {
                let x = thread_rng().gen_range(x_range.clone());
                let y = thread_rng().gen_range(y_range.clone());
                let label = true_function(x, y);
                Sample::new_regression_sample(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let validation_data = (0..1000)
            .map(|_| {
                let x = thread_rng().gen_range(x_range.clone());
                let y = thread_rng().gen_range(y_range.clone());
                let label = true_function(x, y);
                Sample::new_regression_sample(vec![x, y], label)
            })
            .collect::<Vec<Sample>>();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 2,
            layer_sizes: vec![1],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });

        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let mut trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_epochs: 100,
                num_threads: 8,
                learning_rate: 0.01,
                l1_penalty: 0.1,
                entropy_penalty: 0.1,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
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
        let pruning_samples: Vec<Vec<f64>> =
            training_data.iter().map(|s| s.features().clone()).collect();
        let prune_results = trained_model.prune(pruning_samples, 1e-2).unwrap();
        assert_eq!(prune_results, vec![(0, 1)])
    }

    #[test]
    // tests a model that trains on both labels every sample
    fn full_multiregression() {
        fn true_function(x: f64) -> Vec<f64> {
            let new_x = x.powi(2);
            vec![new_x.sin(), new_x.exp()]
        }
        let input_range = 0.0..2.5;
        let rng = &mut thread_rng();
        let training_data: Vec<Sample> = (0..1000)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let labels = true_function(x);
                Sample::new_multiregression_sample(vec![x], labels, vec![true, true])
            })
            .collect();
        let validation_data: Vec<Sample> = (0..100)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let labels = true_function(x);
                Sample::new_multiregression_sample(vec![x], labels, vec![true, true])
            })
            .collect();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 1,
            layer_sizes: vec![1, 2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 5,
                num_epochs: 100,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &trained_model);
        assert!(
            validation_loss < untrained_validation_loss,
            "Validation loss did not decrease after training. Before training: {}, After training: {}",
            untrained_validation_loss,
            validation_loss
            );
    }

    #[test]
    // tests a model that trains on one of two labels every sample, but both target functions are built from the same base function (let y = x^2, then f1 = sin(y), f2 = exp(y))
    // which label each sample uses is determined by the value of x
    fn partial_multiregression_identical_base_determined_split() {
        fn true_function(x: f64) -> Vec<f64> {
            let new_x = x.powi(2);
            vec![new_x.sin(), new_x.exp()]
        }
        let input_range = 0.0..2.5;
        let rng = &mut thread_rng();
        let training_data: Vec<Sample> = (0..1000)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let mut labels = true_function(x);
                let mut label_mask = vec![true, true];
                if x > 3.0 {
                    labels[0] = 0.0;
                    label_mask[0] = false;
                } else {
                    labels[1] = 0.0;
                    label_mask[1] = false;
                }
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let validation_data: Vec<Sample> = (0..100)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let labels = true_function(x);
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 1,
            layer_sizes: vec![1, 2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 5,
                num_epochs: 100,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &trained_model);
        assert!(
            validation_loss < untrained_validation_loss,
            "Validation loss did not decrease after training. Before training: {}, After training: {}",
            untrained_validation_loss,
            validation_loss
            );
    }

    #[test]
    // tests a model that trains on one of two labels every sample, but both target functions are built from the same base function (let y = x^2, then f1 = sin(y), f2 = exp(y))
    // which label each sample uses is determined randomly
    fn partial_multiregression_identical_base_random_split() {
        fn true_function(x: f64) -> Vec<f64> {
            let new_x = x.powi(2);
            vec![new_x.sin(), new_x.exp()]
        }
        let input_range = 0.0..2.5;
        let rng = &mut thread_rng();
        let training_data: Vec<Sample> = (0..1000)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let mut labels = true_function(x);
                let mut label_mask = vec![true, true];
                let use_first = rng.gen_bool(0.5);
                if use_first {
                    labels[1] = 0.0;
                    label_mask[1] = false;
                } else {
                    labels[0] = 0.0;
                    label_mask[0] = false;
                }
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let validation_data: Vec<Sample> = (0..100)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let labels = true_function(x);
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 1,
            layer_sizes: vec![1, 2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 5,
                num_epochs: 100,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &trained_model);
        assert!(
            validation_loss < untrained_validation_loss,
            "Validation loss did not decrease after training. Before training: {}, After training: {}",
            untrained_validation_loss,
            validation_loss
            );
    }

    #[test]
    // tests a model that trains on one of two labels every sample, where each target function is a different function of x (let f1 = sin(x^2), f2 = exp(x))
    // which label each sample uses is determined by the value of x
    fn partial_multiregression_different_base_determined_split() {
        fn true_function(x: f64) -> Vec<f64> {
            vec![(x.powi(2)).sin(), x.exp()]
        }
        let input_range = 0.0..2.5;
        let rng = &mut thread_rng();
        let training_data: Vec<Sample> = (0..1000)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let mut labels = true_function(x);
                let mut label_mask = vec![true, true];
                if x > 3.0 {
                    labels[0] = 0.0;
                    label_mask[0] = false;
                } else {
                    labels[1] = 0.0;
                    label_mask[1] = false;
                }
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let validation_data: Vec<Sample> = (0..100)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let labels = true_function(x);
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 1,
            layer_sizes: vec![1, 2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 5,
                num_epochs: 100,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &trained_model);
        assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
        );
    }

    #[test]
    // tests a model that trains on one of two labels every sample, where each target function is a different function of x (let f1 = sin(x^2), f2 = exp(x))
    // which label each sample uses is determined randomly
    fn partial_multiregression_different_base_random_split() {
        fn true_function(x: f64) -> Vec<f64> {
            vec![(x.powi(2)).sin(), x.exp()]
        }
        let input_range = 0.0..2.5;
        let rng = &mut thread_rng();
        let training_data: Vec<Sample> = (0..1000)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let mut labels = true_function(x);
                let mut label_mask = vec![true, true];
                let use_first = rng.gen_bool(0.5);
                if use_first {
                    labels[1] = 0.0;
                    label_mask[1] = false;
                } else {
                    labels[0] = 0.0;
                    label_mask[0] = false;
                }
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let validation_data: Vec<Sample> = (0..100)
            .map(|_| {
                let x = rng.gen_range(input_range.clone());
                let labels = true_function(x);
                Sample::new_multiregression_sample(vec![x], labels, vec![true, false])
            })
            .collect();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 1,
            layer_sizes: vec![1, 2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 5,
                num_epochs: 100,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &trained_model);
        assert!(
            validation_loss < untrained_validation_loss,
            "Validation loss did not decrease after training. Before training: {}, After training: {}",
            untrained_validation_loss,
            validation_loss
            );
    }

    #[test]
    // tests a model that trains on two labels every sample, where each target function is a function of a different input (let f1 = sin(x), f2 = exp(y))
    fn full_multiregression_distinct_functions() {
        fn true_function(x: f64, y: f64) -> Vec<f64> {
            vec![x.sin(), y.exp()]
        }
        let input_range = 0.0..6.0;
        let training_data: Vec<Sample> = (0..1000)
            .map(|_| {
                let x = thread_rng().gen_range(input_range.clone());
                let y = thread_rng().gen_range(input_range.clone());
                let labels = true_function(x, y);
                let label_mask = vec![true, true];
                Sample::new_multiregression_sample(vec![x, y], labels, label_mask)
            })
            .collect();
        let validation_data: Vec<Sample> = (0..100)
            .map(|_| {
                let x = thread_rng().gen_range(input_range.clone());
                let y = thread_rng().gen_range(input_range.clone());
                let labels = true_function(x, y);
                let label_mask = vec![true, true];
                Sample::new_multiregression_sample(vec![x, y], labels, label_mask)
            })
            .collect();
        let mut untrained_model = Kan::new(&KanOptions {
            num_features: 2,
            layer_sizes: vec![2, 2],
            degree: 3,
            coef_size: 10,
            model_type: ModelType::Regression,
            class_map: None,
            embedding_options: None,
        });
        let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
        let trained_model = train_model(
            untrained_model,
            &training_data,
            TrainingOptions {
                num_threads: 5,
                num_epochs: 100,
                learning_rate: 0.01,
                each_epoch: fekan::training_options::EachEpoch::DoNotValidateModel,
                ..TrainingOptions::default()
            },
        )
        .unwrap();
        let validation_loss = validate_model(&validation_data, &trained_model);
        assert!(
            validation_loss < untrained_validation_loss,
            "Validation loss did not decrease after training. Before training: {}, After training: {}",
            untrained_validation_loss,
            validation_loss
        );
    }

    mod symbols {
        use super::*;
        use test_log::test;
        #[test]
        fn sin_x() {
            let input_range = 0.0..6.0;

            fn true_function(x: f64) -> f64 {
                4.0 * x.sin()
            }

            let training_data = (0..2400)
                .map(|_| {
                    let x: f64 = thread_rng().gen_range(input_range.clone());
                    let label = true_function(x);
                    Sample::new_regression_sample(vec![x], label)
                })
                .collect::<Vec<Sample>>();
            let validation_data = (0..100)
                .map(|_| {
                    let x: f64 = thread_rng().gen_range(input_range.clone());
                    let label = true_function(x);
                    Sample::new_regression_sample(vec![x], label)
                })
                .collect::<Vec<Sample>>();
            let mut untrained_model = Kan::new(&KanOptions {
                num_features: 1,
                layer_sizes: vec![1],
                degree: 3,
                coef_size: 10,
                model_type: ModelType::Regression,
                class_map: None,
                embedding_options: None,
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
                    batch_size: 150,
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
                symbolic_loss - validation_loss < 0.01,
                "Symbolification significantly degraded loss. Before {}, After {}",
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

            let training_data = (0..2400)
                .map(|_| {
                    let x: f64 = thread_rng().gen_range(input_range.clone());
                    let label = true_function(x);
                    Sample::new_regression_sample(vec![x], label)
                })
                .collect::<Vec<Sample>>();
            let validation_data = (0..100)
                .map(|_| {
                    let x: f64 = thread_rng().gen_range(input_range.clone());
                    let label = true_function(x);
                    Sample::new_regression_sample(vec![x], label)
                })
                .collect::<Vec<Sample>>();
            let mut untrained_model = Kan::new(&KanOptions {
                num_features: 1,
                layer_sizes: vec![1],
                degree: 3,
                coef_size: 10,
                model_type: ModelType::Regression,
                class_map: None,
                embedding_options: None,
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
                    batch_size: 150,
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
            trained_model.test_and_set_symbolic(0.98);
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

            let training_data = (0..2400)
                .map(|_| {
                    let x: f64 = thread_rng().gen_range(input_range.clone());
                    let label = true_function(x);
                    Sample::new_regression_sample(vec![x], label)
                })
                .collect::<Vec<Sample>>();
            let validation_data = (0..100)
                .map(|_| {
                    let x: f64 = thread_rng().gen_range(input_range.clone());
                    let label = true_function(x);
                    Sample::new_regression_sample(vec![x], label)
                })
                .collect::<Vec<Sample>>();
            let mut untrained_model = Kan::new(&KanOptions {
                num_features: 1,
                layer_sizes: vec![1],
                degree: 3,
                coef_size: 10,
                model_type: ModelType::Regression,
                class_map: None,
                embedding_options: None,
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
                    batch_size: 150,
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
            trained_model.test_and_set_symbolic(0.90);
            let symbolic_loss = validate_model(&validation_data, &mut trained_model);
            assert!(
                symbolic_loss < validation_loss,
                "Symbolification did not improve loss. Before {}, After {}",
                validation_loss,
                symbolic_loss,
            );
        }
    }
}
