#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! A library to build and train Kolmogorov-Arnold neural networks.
//!
//! The `fekan` crate contains utilities to build and train Kolmogorov-Arnold Networks (KANs) in Rust. Of particular note:
//! * the [`Kan`] struct, which represents a full KAN model
//! * the [`train_model`] function, which trains a KAN model
//! * the [`KanLayer`](crate::kan_layer::KanLayer) struct, which represents a single layer of a KAN, and can be used to build full KANs or as a layer in other models
//!
//! ## What is a Kolmogorov-Arnold Network?
//! Rather than perform a weighted sum of the activations of the previous layer and passing the sum through a fixed non-linear function,
//! each node in a KAN passes each activation from the previous layer through a different, trainable non-linear function, then sums and outputs the result.
//! This allows the network to be more interpretable than,
//! and in some cases be significantly more accurate with a smaller memory footprint than, traditional neural networks.
//!
//! Because the activation of each KAN layer can not be calculated using matrix multiplication, training a KAN is currently much slower than training a traditional neural network of comparable size.
//! It is the author's hope, however, that the increased accuracy of KANs will allow smaller networks to be used in many cases, offsetting most increased training time;
//! and that the interpretability of KANs will more than justify whatever aditional training time remains.
//!
//! For more information on the theory behind this library and examples of problem-sets well suited to KANs, see the arXiv paper [KAN: Kolmogorov-Arnold Neural Networks](https://arxiv.org/abs/2404.19756)
//!
//! # Examples
//! Build, train and save a full KAN regression model with a 2-dimensional input, 1 hidden layer with 3 nodes, and 1 output node,
//! where each layer uses degree-4 [B-splines](https://en.wikipedia.org/wiki/B-spline) with 5 coefficients (AKA control points):
//! ```
//! use fekan::kan::{Kan, KanOptions, ModelType};
//! use fekan::{Sample, training_options::{TrainingOptions, EachEpoch}};
//! use tempfile::tempfile;
//!
//!
//!
//! // initialize the model
//! let model_options = KanOptions{
//!     input_size: 2,
//!     layer_sizes: vec![3, 1],
//!     degree: 3,
//!     coef_size: 7,
//!     model_type: ModelType::Regression,
//!     class_map: None};
//! let mut untrained_model = Kan::new(&model_options);
//!
//! // train the model
//! let training_data: Vec<Sample> = Vec::new();
//! /* Load training data */
//! # let sample_1 = Sample::new(vec![1.0, 2.0], 3.0);
//! # let sample_2 = Sample::new(vec![-1.0, 1.0], 0.0);
//! # let training_data = vec![sample_1, sample_2];
//!
//! let trained_model = fekan::train_model(untrained_model, &training_data, TrainingOptions::default())?;
//!
//! // save the model
//! // both Kan and KanLayer implement the serde Serialize trait, so they can be saved to a file using any serde-compatible format
//! // here we use the ciborium crate to save the model in the CBOR format
//! let mut file = tempfile().unwrap();
//! ciborium::into_writer(&trained_model, &mut file)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Contains the main struct of the library - [`Kan`] - which represents a full Kolmogorov-Arnold Network.
pub mod kan;
/// Contains the struct [`KanLayer`](crate::kan_layer::KanLayer), which represents a single layer of a Kolmogorov-Arnold Network.
pub mod kan_layer;
/// Contains the struct [`TrainingError`] representing an error encountered during training.
pub mod training_error;
/// Options for training a model with [`crate::train_model`].
pub mod training_options;

use std::thread;

use kan::{Kan, ModelType};
use log::{debug, info};
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use shuffle::{fy, shuffler::Shuffler};
use training_error::TrainingError;
use training_options::{EachEpoch, TrainingOptions};

/// A sample of data to be used in training a model.
///
/// Used for both [training](train_model) and [validation](validate_model) data.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Sample {
    /// The input data for the model
    features: Vec<f64>,
    /// The expected output of the model
    label: f64, // use a f64 so the size doesn't change between platforms
}

impl Sample {
    /// Create a new Sample
    pub fn new(features: Vec<f64>, label: f64) -> Self {
        Sample { features, label }
    }

    /// Get the features of the sample
    pub fn features(&self) -> &Vec<f64> {
        &self.features
    }
    /// Get the label of the sample
    pub fn label(&self) -> f64 {
        self.label
    }
}

/// Train the provided model with the provided data.
///
/// if `validation_data` is not `None`, the model will be validated after each epoch, and the validation loss will be reported to the observer, otherwise the reported validation loss will be NaN.
///
/// This function will report log status.
///
/// This function uses the [`ModelType`] of the model to determine how to calculate the loss and gradient of the model.
///
/// Returns the trained model if no errors are thrown.
///
/// # Notes
/// * if training_data.len() % knot_update_interval == 0, the knot update interval will be increased by 1 until the modulo is no longer 0, to ensure the knots are not updated at the very end of an epoch, which prevents grid extension after the epoch
///
/// # Errors
/// returns a [TrainingError] if the model reports an error at any point during training.
///
/// # Example
/// train a model with some training data:
/// ```
/// use fekan::{train_model, Sample, training_options::TrainingOptions};
/// use fekan::kan::{Kan, KanOptions, ModelType};
/// # use fekan::training_error::TrainingError;
///
/// # let some_model_options = KanOptions{ input_size: 2, layer_sizes: vec![3, 1], degree: 3, coef_size: 7, model_type: ModelType::Regression, class_map: None};
/// let untrained_model = Kan::new(&some_model_options);
/// let mut training_data: Vec<Sample> = Vec::new();
/// /* Load training data */
/// # training_data.push(Sample::new(vec![1.0, 2.0], 3.0));
///
/// let trained_model = train_model(
///     untrained_model,
///     &training_data,
///     TrainingOptions::default())?;
/// # Ok::<(), TrainingError>(())
/// ```
///
/// Train a model, testing it against the validation data after each epoch:
/// ```
/// use fekan::{train_model, Sample, training_options::{TrainingOptions, EachEpoch}};
/// use fekan::kan::{Kan, KanOptions, ModelType};
/// # use fekan::training_error::TrainingError;
/// # let some_model_options = KanOptions{ input_size: 2, layer_sizes: vec![3, 1], degree: 3, coef_size: 7, model_type: ModelType::Regression, class_map: None};
///
/// let untrained_model = Kan::new(&some_model_options);
///
/// let mut training_data: Vec<Sample> = Vec::new();
/// let mut validation_data: Vec<Sample> = Vec::new();
/// /* Load training and validation data */
/// # training_data.push(Sample::new(vec![1.0, 2.0], 3.0));
/// # validation_data.push(Sample::new(vec![1.0, 2.0], 3.0));
///
/// let trained_model = train_model(
///     untrained_model,
///     &training_data,
///     TrainingOptions::default())?;
/// // loss is reported each epoch to the training observer
/// # Ok::<(), TrainingError>(())
/// ```
///
// TODO implement training multi-variate regression models. I'll need to calculate the loss w.r.t each output node and run backward on each,
// then call update after all those gradients have been accumulated
pub fn train_model(
    mut model: Kan,
    training_data: &[Sample],
    options: TrainingOptions,
) -> Result<Kan, TrainingError> {
    // TRAINING

    let mut randomness = thread_rng();
    let mut fys = fy::FisherYates::default();
    let mut knot_extensions_completed = 0;
    let knot_extension_targets = options.knot_extension_targets.unwrap_or_default();
    let knot_extension_times = options.knot_extension_times.unwrap_or_default();

    // do several "dummy" passes so we can udpate the knots to span the proper ranges before we start training
    // we need to do one round of pre-setting per layer, since the knot ranges of layer n depend on the output of layer n-1
    preset_knot_ranges(&mut model, training_data)?;
    // now start the actual training loop
    // if the number of threads is <= 1, run the training loop in a single thread

    for epoch in 1..=options.num_epochs {
        let mut epoch_loss = 0.0;
        let mut samples_seen = 0;
        // shuffle the training data
        let mut shuffled_data = training_data.to_vec();
        fys.shuffle(&mut shuffled_data, &mut randomness)
            .expect("Shuffling can't fail");
        // multi-threaded training
        if options.num_threads > 1 {
            let chunk_size =
                f32::ceil(shuffled_data.len() as f32 / options.num_threads as f32) as usize;
            let multithreaded_training: Result<Vec<(Kan, f64)>, TrainingError> = // I love that Result implements FromIterator, so Vec<Result<T,E>> gets automatically converted to Result<Vec<T>, E>
            thread::scope(|s| {
                let handles: Vec<_> = training_data
                    .chunks(chunk_size)
                    .map(|training_data_chunk| {
                        let cloned_model = model.clone();
                        s.spawn(move || {
                            let mut model = cloned_model;
                            let mut chunk_loss = 0.0;
                            let mut chunk_samples_seen = 0;
                            for sample in training_data_chunk {
                                chunk_samples_seen += 1;
                                // forward
                                let logits =
                                    model.forward(sample.clone().features).map_err(|e| {
                                        TrainingError {
                                            source: e,
                                            epoch,
                                            sample: chunk_samples_seen,
                                        }
                                    })?;
                                // backward
                                let (loss, gradient) = match model.model_type() {
                                    ModelType::Classification => calculate_nll_loss_and_gradient(
                                        &logits,
                                        sample.label as usize,
                                    ),
                                    ModelType::Regression => {
                                        let (loss, dlogit) = calculate_huber_loss_and_gradient(
                                            logits[0],
                                            sample.label,
                                        );
                                        (loss, vec![dlogit])
                                    }
                                };
                                chunk_loss += loss;
                                model.backward(gradient).map_err(|e| TrainingError {
                                    source: e,
                                    epoch,
                                    sample: chunk_samples_seen,
                                })?;
                                // update
                                model.update(options.learning_rate);
                                model.zero_gradients();
                                // update the knots if necessary
                                if chunk_samples_seen % options.knot_update_interval == 0 {
                                    model
                                        .update_knots_from_samples(options.knot_adaptivity)
                                        .map_err(|e| TrainingError {
                                            source: e,
                                            epoch,
                                            sample: chunk_samples_seen,
                                        })?;
                                    model.clear_samples();
                                }
                            }
                            Ok((model, chunk_loss))
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect()
            });
            // recombine the models and losses
            let multithreaded_training_result = multithreaded_training?;
            let (partially_trained_models, chunk_losses): (Vec<Kan>, Vec<f64>) =
                multithreaded_training_result.into_iter().unzip();
            model = Kan::merge_models(partially_trained_models).map_err(|e| TrainingError {
                source: e,
                epoch,
                sample: 0,
            })?;

            epoch_loss = chunk_losses.iter().sum::<f64>() / training_data.len() as f64;
        }
        // end multi-threaded code
        // single-thread-specific training
        else {
            for sample in shuffled_data.iter() {
                samples_seen += 1;
                // forward
                let logits = model
                    .forward(sample.clone().features)
                    .map_err(|e| TrainingError {
                        source: e,
                        epoch,
                        sample: samples_seen,
                    })?;
                // backward
                let (loss, gradient) = match model.model_type() {
                    ModelType::Classification => {
                        calculate_nll_loss_and_gradient(&logits, sample.label as usize)
                    }
                    ModelType::Regression => {
                        let (loss, dlogit) =
                            calculate_huber_loss_and_gradient(logits[0], sample.label);
                        (loss, vec![dlogit])
                    }
                };
                epoch_loss += loss;
                model.backward(gradient).map_err(|e| TrainingError {
                    source: e,
                    epoch,
                    sample: samples_seen,
                })?;
                // update
                model.update(options.learning_rate);
                model.zero_gradients();
                // update the knots if necessary
                if samples_seen % options.knot_update_interval == 0 {
                    model
                        .update_knots_from_samples(options.knot_adaptivity)
                        .map_err(|e| TrainingError {
                            source: e,
                            epoch,
                            sample: samples_seen,
                        })?;
                    model.clear_samples();
                }
            }
        } // end single-thread-specific code
        match options.each_epoch {
            EachEpoch::ValidateModel(validation_data) => {
                let validation_lostt = validate_model(validation_data, &mut model);
                info!(
                    "Epoch: {}, Epoch Loss: {}, Validation Loss: {}",
                    epoch, epoch_loss, validation_lostt
                );
            }
            EachEpoch::DoNotValidateModel => info!("Epoch: {}, Epoch Loss: {}", epoch, epoch_loss),
        };
        // notify the observer that the epoch has ended

        // update the knots if necessary
        if knot_extensions_completed < knot_extension_targets.len()
            && epoch == knot_extension_times[knot_extensions_completed]
        {
            let target_length = knot_extension_targets[knot_extensions_completed];
            let old_length = model.knot_length();
            // get some samples for knot extension
            for i in 0..(2 * old_length) {
                model.forward(shuffled_data[i].features.clone()).unwrap();
            }
            model
                .set_knot_length(target_length)
                .map_err(|e| TrainingError {
                    source: e,
                    epoch,
                    sample: samples_seen,
                })?;
            info!("Knot extension: {} -> {}", old_length, target_length);
            knot_extensions_completed += 1;
        }
    }

    Ok(model)
}

/// Scan over the training data and adjust model knot ranges. This is equivalent to calling [`Kan::forward`] on each sample in the training data, then calling [`Kan::update_knots_from_samples`] with a `knot_adaptivity` of 0.0.
/// This presetting helps avoid large amounts of training inputs falling outside the knot ranges, which can cause the model to fail to converge.
pub fn preset_knot_ranges(model: &mut Kan, preset_data: &[Sample]) -> Result<(), TrainingError> {
    debug!("Presetting knot ranges");
    if log::log_enabled!(log::Level::Debug) {
        let mut ranges: Vec<(f64, f64)> = vec![(0.0, 0.0); preset_data[0].features.len()];
        for sample in preset_data {
            for idx in 0..sample.features.len() {
                ranges[idx].0 = ranges[idx].0.min(sample.features[idx]);
                ranges[idx].1 = ranges[idx].1.max(sample.features[idx]);
            }
        }
        debug!("Layer 0 input ranges: {:#?}", ranges);
    }
    for set_layer in 0..model.layers.len() {
        for i in 0..preset_data.len() {
            model
                .forward(preset_data[i].features().clone())
                .map_err(|e| TrainingError {
                    source: e,
                    epoch: 0,
                    sample: i,
                })?;
        }

        model
            .update_knots_from_samples(0.0)
            .map_err(|e| TrainingError {
                source: e,
                epoch: 0,
                sample: set_layer * preset_data.len(),
            })?; // we only want to get the proper input ranges - we're not worried about high-frequency resolution

        debug!("Layer {} knot ranges set.", set_layer);
        if log::log_enabled!(log::Level::Debug) && set_layer < model.layers.len() - 1 {
            let mut output_ranges: Vec<(f64, f64)> =
                vec![(0.0, 0.0); model.layers[set_layer].output_dimension()];
            for sample in preset_data {
                let mut outputs = sample.features.clone();
                for layer_idx in 0..=set_layer {
                    outputs = model.layers[layer_idx].forward(&outputs).unwrap();
                }
                for output_idx in 0..outputs.len() {
                    output_ranges[output_idx].0 =
                        output_ranges[output_idx].0.min(outputs[output_idx]);
                    output_ranges[output_idx].1 =
                        output_ranges[output_idx].1.max(outputs[output_idx]);
                }
            }
            debug!("Layer {} input ranges: {:#?}", set_layer + 1, output_ranges);
        }
    }
    Ok(())
}

/// Calculates the loss of the model on the provided validation data. If the model is a classification model, the cross entropy loss is calculated.
/// If the model is a regression model, the mean squared error is calculated.
///
pub fn validate_model(validation_data: &[Sample], model: &mut Kan) -> f64 {
    let mut validation_loss = 0.0;

    for sample in validation_data {
        let output = model.infer(sample.features().clone()).unwrap();
        let loss = match model.model_type() {
            ModelType::Classification => {
                let (loss, _) = calculate_nll_loss_and_gradient(&output, sample.label as usize);
                loss
            }
            ModelType::Regression => {
                let (loss, _) = calculate_huber_loss_and_gradient(output[0], sample.label as f64); //TODO allow different delta values
                loss
            }
        };
        validation_loss += loss;
    }
    validation_loss /= validation_data.len() as f64;

    validation_loss
}

/// Returns the negative log liklihood loss and the gradient of the loss with respect to the logits,
fn calculate_nll_loss_and_gradient(logits: &Vec<f64>, label: usize) -> (f64, Vec<f64>) {
    // calculate the classification probabilities
    let logit_max = {
        let mut max = f64::NEG_INFINITY;
        for &logit in logits.iter() {
            if logit > max {
                max = logit;
            }
        }
        max
    };
    let norm_logits = logits.iter().map(|&x| x - logit_max).collect::<Vec<f64>>(); // subtract the max logit to prevent overflow
    let counts = norm_logits.iter().map(|&x| x.exp()).collect::<Vec<f64>>();

    let count_sum = counts.iter().sum::<f64>();
    let probs = counts.iter().map(|&x| x / count_sum).collect::<Vec<f64>>();

    let logprobs = probs
        .iter()
        .map(|&x| (x + f64::MIN_POSITIVE).ln()) // make sure we don't take the log of 0
        .collect::<Vec<f64>>();
    let loss = -logprobs[label];

    let mut dlogits = probs;
    dlogits[label] -= 1.;

    (loss, dlogits)
}

/// Calculates the mean squared error loss and the gradient of the loss with respect to the actual value
// fn calculate_mse_and_gradient(actual: f64, expected: f64) -> (f64, f64) {
//     let loss = (actual - expected).powi(2);
//     let gradient = 2.0 * (actual - expected);
//     (loss, gradient)
// }

const HUBER_DELTA: f64 = 1.3407807929942596e154 - 1.0; // f64::MAX ^ 0.5 - 1.0. Chosen so the loss is equivalent to the MSE loss until the error would be greater than f64::MAX, and then it becomes linear

/// Calculates the huber loss and the gradient of the loss with respect to the actual value
fn calculate_huber_loss_and_gradient(actual: f64, expected: f64) -> (f64, f64) {
    let diff = actual - expected;
    if diff.abs() <= HUBER_DELTA {
        (0.5 * diff.powi(2), diff)
    } else {
        (
            HUBER_DELTA * (diff.abs() - 0.5 * HUBER_DELTA),
            HUBER_DELTA * diff.signum(),
        )
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
            .collect::<Vec<f64>>();
        assert_eq!(rounded_loss, expected_loss);
        assert_eq!(rounded_gradients, expected_gradients);
    }

    #[test]
    fn test_nll_loss_and_gradient_2() {
        let logits = vec![50.4043, -42.404835];
        let label = 1;
        let (loss, gradient) = calculate_nll_loss_and_gradient(&logits, label);
        println!("loss: {}, gradient: {:?}", loss, gradient);
        assert!(gradient.iter().all(|x| x.is_normal()));
    }

    // // #[test]
    // fn test_build_knot_extension_plan_update_by_1() {
    //     let starting_knots = 10;
    //     let ideal_knots = 50;
    //     let num_epochs = 100;
    //     let (interval, targets) =
    //         build_knot_extension_plan(starting_knots, ideal_knots, num_epochs);
    //     assert_eq!(interval, 2, "knot update interval");
    //     assert_eq!(
    //         targets.len(),
    //         ideal_knots - starting_knots,
    //         "knot update count"
    //     );
    //     assert_eq!(targets.last().unwrap(), &ideal_knots, "last knot count");
    // }

    // // #[test]
    // fn test_build_knot_extension_plan_update_every_epoch() {
    //     let starting_knots = 10;
    //     let ideal_knots = 100;
    //     let num_epochs = 9;
    //     let (interval, targets) =
    //         build_knot_extension_plan(starting_knots, ideal_knots, num_epochs);
    //     assert_eq!(interval, 1, "knot update interval");
    //     assert_eq!(targets.len(), num_epochs - 1, "knot update count");
    //     assert_eq!(targets.last().unwrap(), &ideal_knots, "last knot count");
    // }

    #[test]
    fn test_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TrainingError>();
    }

    #[test]
    fn test_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TrainingError>();
    }
}
