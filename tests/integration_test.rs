use fekan::{kan::Kan, train_model, validate_model, Sample, TrainingOptions};
use rand::{thread_rng, Rng};

/// Build a model and train it on the function f(x, y, z) = x + y + z > 0
#[test]
fn sum_greater_than_zero() {
    // select 10000 random x's, y's, and z's in the range -1000 to 1000 to train on, and 100 random x's, y's, and z's in the range -1000 to 1000 to validate on
    let training_data = (0..10000)
        .map(|_| {
            let x = thread_rng().gen_range(-1000.0..1000.0);
            let y = thread_rng().gen_range(-1000.0..1000.0);
            let z = thread_rng().gen_range(-1000.0..1000.0);
            let label = (x + y + z) > 0.0;
            Sample {
                features: vec![x, y, z],
                label: label as u32,
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
                label: label as u32,
            }
        })
        .collect::<Vec<Sample>>();

    let input_dimension = 3;
    let layers = vec![2]; // two possible ouputs, true or false
    let k = 3;
    let coef_size = 4;
    let mut untrained_model = Kan::new(input_dimension, layers, k, coef_size);
    let untrained_validation_loss = validate_model(&validation_data, &mut untrained_model);
    let mut trained_model = train_model(
        untrained_model,
        training_data,
        Some(&validation_data),
        TrainingOptions {
            num_epochs: 100,
            knot_update_interval: 500,
            learning_rate: 0.01,
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
}

/// build a model and train it on the function f(x, y) = xy
#[test]
#[ignore]
fn xy() {
    // select 1000 random points in the range -1000 to 1000 to train on, and 100 random points in the range -1000 to 1000 to validate on
    // let training_data = {
    //     let mut rng = rand::thread_rng();
    //     let mut training_data = Vec::new();
    //     for _ in 0..1000 {
    //         let x = rng.gen_range(-1000.0..1000.0);
    //         let y = rng.gen_range(-1000.0..1000.0);
    //         training_data.push();
    //     }
    //     training_data
    // };
    // }
    todo!("Implement f(x,y) = xy test");
}

/// build a model and train it on the function f(x, y) = e^(sin(pi*x) * y^2)
#[test]
#[ignore]
fn exp_sin_pix_y_squared() {
    // I need to implement regression models for this
    todo!("Implement f(x,y) = e^(sin(pi*x) * y^2) test")
}
