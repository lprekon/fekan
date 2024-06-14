use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model, validate_model, Sample, TrainingOptions,
};
use rand::{thread_rng, Rng};

/// Build a model and train it on the function f(x, y, z) = x + y + z > 0. Tests that the trained validation loss is less than the untrained validation loss
#[test]
fn classifier_sum_greater_than_zero() {
    // select 10000 random x's, y's, and z's in the range -1000 to 1000 to train on, and 100 random x's, y's, and z's in the range -1000 to 1000 to validate on
    let training_data = (0..10000)
        .map(|_| {
            let x = thread_rng().gen_range(-1000.0..1000.0);
            let y = thread_rng().gen_range(-1000.0..1000.0);
            let z = thread_rng().gen_range(-1000.0..1000.0);
            let label = (x + y + z) > 0.0;
            Sample {
                features: vec![x, y, z],
                label: label as u32 as f32,
            }
        })
        .collect::<Vec<Sample>>();
    let validation_data = (0..100)
        .map(|_| {
            let x = thread_rng().gen_range(-1000.0..1000.0);
            let y = thread_rng().gen_range(-1000.0..1000.0);
            let z = thread_rng().gen_range(-1000.0..1000.0);
            let label = (x + y + z) > 0.0;
            Sample {
                features: vec![x, y, z],
                label: label as u32 as f32,
            }
        })
        .collect::<Vec<Sample>>();

    let mut untrained_model = Kan::new(&KanOptions {
        input_size: 3,
        layer_sizes: vec![2],
        degree: 3,
        coef_size: 4,
        model_type: ModelType::Classification,
    });
    let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
    let mut trained_model = train_model(
        untrained_model,
        training_data,
        Some(&validation_data),
        TrainingOptions::default(),
    )
    .unwrap();
    let validation_loss = validate_model(&validation_data, &mut trained_model);
    assert!(
        validation_loss < untrained_validation_loss,
        "Validation loss did not decrease after training. Before training: {}, After training: {}",
        untrained_validation_loss,
        validation_loss
    );
}

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
        training_data.push(Sample {
            features: vec![x, y],
            label: x * y,
        });
    }
    for _ in 0..100 {
        let x = rand.gen_range(-1000.0..1000.0);
        let y = rand.gen_range(-1000.0..1000.0);
        validation_data.push(Sample {
            features: vec![x, y],
            label: x * y,
        });
    }
    let mut untrained_model = Kan::new(&KanOptions {
        input_size: 2,
        layer_sizes: vec![3, 2, 1],
        degree: 3,
        coef_size: 4,
        model_type: ModelType::Regression,
    });
    let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
    let mut trained_model = train_model(
        untrained_model,
        training_data,
        Some(&validation_data), // this way if the test fails, we can see the validation loss over time
        TrainingOptions::default(),
    )
    .unwrap();
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
#[ignore]
fn exp_sin_pix_y_squared() {}
