mod util;
use util::TestObserver;

use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model, validate_model, Sample, TrainingOptions,
};
use rand::{thread_rng, Rng};
#[test]
fn regressor_xy() {
    // select 1000 random points in the range -1000 to 1000 to train on, and 100 random points in the range -1000 to 1000 to validate on
    let mut rand = thread_rng();
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
            num_epochs: 250,
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
