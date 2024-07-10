mod util;
use util::TestObserver;

use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model, validate_model, Sample, TrainingOptions,
};
use rand::{thread_rng, Rng};

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
            num_epochs: 250,
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
