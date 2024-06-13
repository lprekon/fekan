pub mod kan;

use kan::Kan;
use std::error::Error;

#[derive(Clone)]
pub struct ClassificationSample {
    pub features: Vec<f32>,
    pub label: u32, // use a u32 so the size doesn't change between platforms
}

pub struct RegressionSample {
    pub features: Vec<f32>,
    pub label: f32,
}

pub enum SampleType {
    Classification(ClassificationSample),
    Regression(RegressionSample),
}

pub struct TrainingOptions {
    pub num_epochs: usize,
    pub knot_update_interval: usize,
    pub learning_rate: f32,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        TrainingOptions {
            num_epochs: 100,
            knot_update_interval: 100,
            learning_rate: 0.001,
        }
    }
}

/// Train the provided model with the provided data. This function will print a heartbeat with a timestamp, epoch count, and epoch loss after each epoch.
/// If validation data is provided, the model will be validated after each epoch, and the validation loss will also be printed.
pub fn train_model(
    mut model: Kan,
    training_data: Vec<ClassificationSample>,
    validation_data: Option<&Vec<ClassificationSample>>,
    options: TrainingOptions,
) -> Result<Kan, Box<dyn Error>> {
    // train the model
    for epoch in 0..options.num_epochs {
        let mut epoch_loss = 0.0;
        let mut samples_seen = 0;
        for sample in &training_data {
            samples_seen += 1;
            // run over each sample in the training data for each epoch
            let logits = model
                .forward(sample.features.iter().map(|&x| x as f32).collect())
                .unwrap();
            // calculate classification probability from logits
            let (loss, dlogits) = calculate_error(&logits, sample.label as usize);
            epoch_loss += loss;
            // pass the error back through the model
            let _ = model.backward(dlogits).unwrap();
            model.update(options.learning_rate); // TODO implement momentum
            model.zero_gradients();
            if samples_seen % options.knot_update_interval == 0 {
                let _ = model.update_knots_from_samples()?;
            }
        }
        epoch_loss /= training_data.len() as f32;

        let mut validation_loss = 0.0;
        if let Some(validation_data) = &validation_data {
            validation_loss = validate_model(&validation_data, &mut model);
        }
        // print stats
        print!(
            "[HEARTBEAT] {} epoch: {} epoch_loss: {}",
            chrono::Local::now(),
            epoch,
            epoch_loss
        );
        if validation_data.is_some() {
            print!(" validation_loss: {validation_loss}");
        }
        println!();
    }

    Ok(model)
}

pub fn validate_model(validation_data: &Vec<ClassificationSample>, model: &mut Kan) -> f32 {
    let mut validation_loss = 0.0;

    for sample in validation_data {
        let logits = model
            .forward(sample.features.iter().map(|&x| x as f32).collect())
            .unwrap();
        let (loss, _) = calculate_error(&logits, sample.label as usize);
        validation_loss += loss;
    }
    validation_loss /= validation_data.len() as f32;

    validation_loss
}

fn calculate_error(logits: &Vec<f32>, label: usize) -> (f32, Vec<f32>) {
    // calculate the classification probabilities
    let (logit_max, logit_max_index) = {
        let mut max = f32::NEG_INFINITY;
        let mut max_index = 0;
        for (i, &logit) in logits.iter().enumerate() {
            if logit > max {
                max = logit;
                max_index = i;
            }
        }
        (max, max_index)
    };
    let norm_logits = logits.iter().map(|&x| x - logit_max).collect::<Vec<f32>>(); // subtract the max logit to prevent overflow
    let counts = norm_logits.iter().map(|&x| x.exp()).collect::<Vec<f32>>();
    let count_sum = counts.iter().sum::<f32>();
    let probs = counts.iter().map(|&x| x / count_sum).collect::<Vec<f32>>();

    // calculate the loss
    let logprobs = probs.iter().map(|&x| x.ln()).collect::<Vec<f32>>();
    let loss = -logprobs[label];

    // calculate the error
    let dlogprobs = (0..probs.len())
        .map(|i| if i == label { -1.0 } else { 0.0 })
        .collect::<Vec<f32>>(); // dloss/dlogpobs. vector is 0 except for the correct class, where it's -1
    let dprobs = probs
        .iter()
        .zip(dlogprobs.iter())
        .map(|(&p, &dlp)| dlp / p)
        .collect::<Vec<f32>>(); // dloss/dprobs = dlogprobs/dprobs * dloss/dlogprobs. d/dx ln(x) = 1/x. dlogprobs/dprobs = 1/probs, `dlp` = dloss/dlogprobs
    let dcounts_sum: f32 = counts
        .iter()
        .zip(dprobs.iter())
        .map(|(&count, &dprob)| -count / (count_sum * count_sum) * dprob)
        .sum();
    let dcounts = dprobs
        .iter()
        .map(|&dprob| dcounts_sum + dprob / count_sum)
        .collect::<Vec<f32>>();
    let dnorm_logits = dcounts
        .iter()
        .zip(counts.iter())
        .map(|(&dcount, &e_norm_logit)| dcount * e_norm_logit)
        .collect::<Vec<f32>>(); // dloss/dnorm_logits = dloss/dcounts * dcounts/dnorm_logits, dcounts/dnorm_logits = d/dx exp(x) = exp(x), and exp(norm_logits) = counts, so we just use counts rather than recalculating
    let dlogit_max: f32 = -dnorm_logits.iter().sum::<f32>();
    let dlogits = dnorm_logits.iter().enumerate().map(|(i, &dnorm_logit)|{if i == logit_max_index {1.0} else {0.0}} * dlogit_max + dnorm_logit).collect::<Vec<f32>>();

    (loss, dlogits)
}
