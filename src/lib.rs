pub mod kan;
pub mod training_observer;

use kan::{Kan, KanError, ModelType};
use training_observer::TrainingObserver;

#[derive(Clone, PartialEq, Debug)]
pub struct Sample {
    pub features: Vec<f32>,
    pub label: f32, // use a f32 so the size doesn't change between platforms
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TrainingOptions {
    pub num_epochs: usize,
    pub knot_update_interval: usize,
    pub knot_adaptivity: f32,
    pub learning_rate: f32,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        TrainingOptions {
            num_epochs: 100,
            knot_update_interval: 100,
            knot_adaptivity: 0.1,
            learning_rate: 0.001,
        }
    }
}

/// Train the provided model with the provided data. This function will report status to the provided observer.
/// If validation data is provided, the model will be validated after each epoch, and provided to the observer. If validation is not provided, the validation loss will be `NaN` in the epoch-end report.
pub fn train_model<T: TrainingObserver>(
    mut model: Kan,
    training_data: Vec<Sample>,
    validation_data: Option<&Vec<Sample>>,
    training_observer: &T,
    options: TrainingOptions,
) -> Result<Kan, TrainingError> {
    // train the model
    for epoch in 0..options.num_epochs {
        let mut epoch_loss = 0.0;
        let mut samples_seen = 0;
        for sample in &training_data {
            samples_seen += 1;
            // run over each sample in the training data for each epoch
            let output = model
                .forward(sample.features.iter().map(|&x| x as f32).collect())
                .map_err(|e| TrainingError {
                    source: e,
                    epoch,
                    sample: samples_seen,
                })?;
            match model.model_type {
                ModelType::Classification => {
                    // calculate classification probability from logits
                    let (loss, dlogits) =
                        calculate_nll_loss_and_gradient(&output, sample.label as usize);
                    epoch_loss += loss;
                    // pass the error back through the model
                    let _ = model.backward(dlogits).map_err(|e| TrainingError {
                        source: e,
                        epoch,
                        sample: samples_seen,
                    })?;
                }
                ModelType::Regression => {
                    let (loss, dlogits) =
                        calculate_mse_and_gradient(output[0], sample.label as f32);
                    epoch_loss += loss;
                    let _ = model.backward(vec![dlogits]).map_err(|e| TrainingError {
                        source: e,
                        epoch,
                        sample: samples_seen,
                    })?;
                }
            }
            model.update(options.learning_rate); // TODO implement momentum
            model.zero_gradients();
            if samples_seen % options.knot_update_interval == 0 {
                let _ = model
                    .update_knots_from_samples(options.knot_adaptivity)
                    .map_err(|e| TrainingError {
                        source: e,
                        epoch,
                        sample: samples_seen,
                    })?;
                model.clear_samples();
            }

            training_observer.on_sample_end();
        }
        epoch_loss /= training_data.len() as f32;

        let validation_loss = match validation_data {
            Some(validation_data) => {
                validate_model(&validation_data, &mut model, training_observer)
            }
            None => f32::NAN,
        };

        training_observer.on_epoch_end(epoch, epoch_loss, validation_loss);
    }

    Ok(model)
}

/// Calculates the loss of the model on the provided validation data. If the model is a classification model, the cross entropy loss is calculated.
/// If the model is a regression model, the mean squared error is calculated.
pub fn validate_model<T: TrainingObserver>(
    validation_data: &Vec<Sample>,
    model: &mut Kan,
    observer: &T,
) -> f32 {
    let mut validation_loss = 0.0;

    for sample in validation_data {
        let output = model
            .forward(sample.features.iter().map(|&x| x as f32).collect())
            .unwrap();
        let loss = match model.model_type {
            ModelType::Classification => {
                let (loss, _) = calculate_nll_loss_and_gradient(&output, sample.label as usize);
                loss
            }
            ModelType::Regression => {
                let (loss, _) = calculate_mse_and_gradient(output[0], sample.label as f32);
                loss
            }
        };
        validation_loss += loss;
        observer.on_sample_end();
    }
    validation_loss /= validation_data.len() as f32;

    validation_loss
}

/// Returns the negative log liklihood loss and the gradient of the loss with respect to the logits,
fn calculate_nll_loss_and_gradient(logits: &Vec<f32>, label: usize) -> (f32, Vec<f32>) {
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
    // should I multiply the actual loss in here? intuitively it feels like I should, but the math doesn't tell me I must
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

/// Calculates the mean squared error loss and the gradient of the loss with respect to the actual value
fn calculate_mse_and_gradient(actual: f32, expected: f32) -> (f32, f32) {
    let loss = (actual - expected).powi(2);
    let gradient = 2.0 * (actual - expected);
    (loss, gradient)
}

// EmptyObserver is basically a singleton, so there's no point in implementing any other common traits
#[derive(Default)]
pub struct EmptyObserver {}
impl EmptyObserver {
    pub fn new() -> Self {
        EmptyObserver {}
    }
}
impl TrainingObserver for EmptyObserver {
    fn on_epoch_end(&self, _epoch: usize, _epoch_loss: f32, _validation_loss: f32) {
        // do nothing
    }
    fn on_sample_end(&self) {
        // do nothing
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TrainingError {
    pub source: KanError,
    pub epoch: usize,
    pub sample: usize,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "epoch {} sample {} encountered error {}",
            self.epoch, self.sample, self.source
        )
    }
}

impl std::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(test)]
mod test {
    // ! test the loss functions

    use super::*;

    #[test]
    fn test_nll_loss_and_gradient() {
        // values calculated using Karpathy's Makemore backpropogation example
        let logits = vec![
            0.0043, -0.2063, 0.0260, -0.1313, -0.2248, 0.0478, 0.1392, 0.1436, 0.0624, -0.1926,
            0.0551, -0.2938, 0.1467, -0.0836, -0.1743, -0.0238, -0.1242, -0.2127, -0.1016, 0.0549,
            -0.0582, -0.0845, 0.0619, -0.0104, -0.0895, 0.0112, -0.3106,
        ];
        let label = 1;
        let (loss, gradient) = calculate_nll_loss_and_gradient(&logits, label);
        let expected_loss = 3.4522;
        let expected_gradients = vec![
            0.0391, -0.9683, 0.0400, 0.0341, 0.0311, 0.0408, 0.0447, 0.0449, 0.0414, 0.0321,
            0.0411, 0.0290, 0.0451, 0.0358, 0.0327, 0.0380, 0.0344, 0.0315, 0.0352, 0.0411, 0.0367,
            0.0358, 0.0414, 0.0385, 0.0356, 0.0394, 0.0285,
        ];
        let rounded_loss = (loss * 10000.0).round() / 10000.0;
        let rounded_gradients = gradient
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect::<Vec<f32>>();
        assert_eq!(rounded_loss, expected_loss);
        assert_eq!(rounded_gradients, expected_gradients);
    }
}
