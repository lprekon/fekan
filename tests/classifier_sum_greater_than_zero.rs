use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model, validate_model, Sample, TrainingOptions,
};
use rand::{thread_rng, Rng};
mod util;
use util::TestObserver;

/// Build a model and train it on the function f(x, y, z) = x + y + z > 0. Tests that the trained validation loss is less than the untrained validation loss
#[test]
fn classifier_sum_greater_than_zero() {
    // select 10000 random x's, y's, and z's in the range -1000 to 1000 to train on, and 100 random x's, y's, and z's in the range -1000 to 1000 to validate on
    let training_data = (0..1000)
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
            max_knot_length: Some(100),
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
