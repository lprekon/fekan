#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

//! A library to build and train Kolmogorov-Arnold neural networks.
//!
//! The `fekan` crate contains utilities to build and train Kolmogorov-Arnold Networks (KANs) in Rust.
//!
//! The [kan_layer] module contains the [`kan_layer::KanLayer`] struct, representing a single layer of a KAN,
//! which can be used to build full KANs or as a layer in other models.
//!
//! The crate also contains the [`Kan`] struct, which represents a full KAN model.
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
//! use fekan::{Sample, TrainingOptions, EachEpoch};
//! use tempfile::tempfile;
//!
//!
//!
//! // initialize the model
//! let model_options = KanOptions{
//!     input_size: 2,
//!     layer_sizes: vec![3, 1],
//!     degree: 4,
//!     coef_size: 5,
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
//! let trained_model = fekan::train_model(untrained_model, &training_data, EachEpoch::DoNotValidateModel, &fekan::EmptyObserver::new(), TrainingOptions::default())?;
//!
//! // save the model
//! // both Kan and KanLayer implement the serde Serialize trait, so they can be saved to a file using any serde-compatible format
//! // here we use the ciborium crate to save the model in the CBOR format
//! let mut file = tempfile().unwrap();
//! ciborium::into_writer(&trained_model, &mut file)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Contains the main struct of the library, the [`Kan`] struct, which represents a full Kolmogorov-Arnold Network.
pub mod kan;
/// Contains the struct [`KanLayer`](crate::kan_layer::KanLayer), which represents a single layer of a Kolmogorov-Arnold Network.
pub mod kan_layer;
/// Provides a trait for observing the training process during [`crate::train_model`].
pub mod training_observer;

use std::cmp::min;

use kan::{Kan, KanError, ModelType};
use rand::thread_rng;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use shuffle::{fy, shuffler::Shuffler};
use training_observer::TrainingObserver;

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

/// Used by the [`train_model`] function to determine how the model should be trained.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TrainingOptions {
    /// number of epochs for which to train, where an epoch is one pass through the training data
    pub num_epochs: usize,
    /// number of samples to pass through the model before updating the knots. See [kan_layer::KanLayer::update_knots_from_samples] for more information about this process.
    pub knot_update_interval: usize,
    /// the adaptivity of the knots when updating them. See [kan_layer::KanLayer::update_knots_from_samples] for more information about this process.
    pub knot_adaptivity: f64,
    /// the learning rate of the model
    pub learning_rate: f64,
    /// If set, the maximum length to which the knot vectors can be extended during training.
    /// Liu et al. 2024 conjecture that the ideal model has a total knot count ~= ||training_data||, but that could become obscene for small models and large datasets
    pub max_knot_length: Option<usize>,
    /// the number of threads to use when training the model.
    pub num_threads: usize,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        TrainingOptions {
            num_epochs: 100,
            knot_update_interval: 100,
            knot_adaptivity: 0.1,
            learning_rate: 0.001,
            max_knot_length: Some(1000),
            num_threads: 1,
        }
    }
}

/// Train the provided model with the provided data.
///
/// if `validation_data` is not `None`, the model will be validated after each epoch, and the validation loss will be reported to the observer, otherwise the reported validation loss will be NaN.
///
/// This function will report status to the provided [`training_observer`](TrainingObserver).
///
/// This function uses the [ModelType] of the model to determine how to calculate the loss and gradient of the model.
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
/// train a model, using the provided [EmptyObserver] to ignore all training events:
/// ```
/// use fekan::{train_model, Sample, TrainingOptions, EmptyObserver, EachEpoch};
/// use fekan::kan::{Kan, KanOptions, ModelType};
/// # use fekan::TrainingError;
///
/// # let some_model_options = KanOptions{ input_size: 2, layer_sizes: vec![3, 1], degree: 4, coef_size: 5, model_type: ModelType::Regression, class_map: None};
/// let untrained_model = Kan::new(&some_model_options);
/// let training_data: Vec<Sample> = Vec::new();
///
/// /* Load training data */
///
/// let trained_model = train_model(
///     untrained_model,
///     &training_data,
///     EachEpoch::DoNotValidateModel,
///     &EmptyObserver::new(),
///     TrainingOptions::default())?;
/// # Ok::<(), TrainingError>(())
/// ```
///
/// Train a model, testing it against the validation data after each epoch, and catching the results with a custom struct that implements [TrainingObserver]:
/// ```
/// use fekan::{train_model, Sample, TrainingOptions, EachEpoch};
/// use fekan::kan::{Kan, KanOptions, ModelType};
/// # use fekan::TrainingError;
/// # use fekan::training_observer::TrainingObserver;
/// # let some_model_options = KanOptions{ input_size: 2, layer_sizes: vec![3, 1], degree: 4, coef_size: 5, model_type: ModelType::Regression, class_map: None};
/// # struct MyCustomObserver {}
/// # impl MyCustomObserver {
/// #     fn new() -> Self { MyCustomObserver{} }
/// # }
/// # impl TrainingObserver for MyCustomObserver {
/// #     fn on_epoch_end(&self, epoch: usize, epoch_loss: f64, validation_loss: f64) {}
/// #     fn on_sample_end(&self) {}
/// # }
///
/// let untrained_model = Kan::new(&some_model_options);
///
/// let my_observer = MyCustomObserver::new(); // custom type that implements TrainingObserver
///
/// let training_data: Vec<Sample> = Vec::new();
/// let validation_data: Vec<Sample> = Vec::new();
///
/// /* Load training and validation data */
///
/// let trained_model = train_model(
///     untrained_model,
///     &training_data,
///     EachEpoch::ValidateModel(&validation_data),
///     &my_observer,
///     TrainingOptions::default())?;
/// # Ok::<(), TrainingError>(())
/// ```
// TODO implement training multi-variate regression models. I'll need to calculate the loss w.r.t each output node and run backward on each,
// then call update after all those gradients have been accumulated
pub fn train_model<T: TrainingObserver>(
    mut model: Kan,
    training_data: &[Sample],
    validate: EachEpoch,
    training_observer: &T,
    options: TrainingOptions,
) -> Result<Kan, TrainingError> {
    // TRAINING

    // build the thread pool
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(options.num_threads)
        .build()
        .expect("Failed to build thread pool");

    // calculate the ideal knot length, using the conjectur from Liu et al. 2024 that the ideal total knot count ~= ||training_data||
    let num_splines = model
        .layers
        .iter()
        .fold(0, |acc, layer| acc + layer.total_edges());
    let ideal_knot_length = (training_data.len() as f64 / num_splines as f64) as usize;
    let (knot_extension_interval, knot_extension_targets) = build_knot_extension_plan(
        model.knot_length(),
        min(
            ideal_knot_length,
            options.max_knot_length.unwrap_or(usize::MAX),
        ),
        options.num_epochs,
    );
    // eprintln!(
    //     "ideal knot length: {}, knot extension interval: {}, knot extension targets: {:?}",
    //     ideal_knot_length, knot_extension_interval, knot_extension_targets
    // );
    let mut next_knot_target_idx = 0;
    let mut randomness = thread_rng();
    let mut fys = fy::FisherYates::default();
    for epoch in 0..options.num_epochs {
        let mut epoch_loss = 0.0;
        let mut samples_seen = 0;
        // shuffle the training data
        let mut shuffled_pointers: Vec<&Sample> = training_data.iter().collect();
        fys.shuffle(&mut shuffled_pointers, &mut randomness)
            .expect("Shuffling can't fail");
        for sample in shuffled_pointers {
            samples_seen += 1;
            // run over each sample in the training data for each epoch
            let output = model
                .forward_concurrent(
                    sample.features.iter().map(|&x| x as f64).collect(),
                    &thread_pool,
                )
                .map_err(|e| TrainingError {
                    source: e,
                    epoch,
                    sample: samples_seen,
                })?;
            let (loss, dlogits) = match model.model_type() {
                ModelType::Classification => {
                    calculate_nll_loss_and_gradient(&output, sample.label as usize)
                }
                ModelType::Regression => {
                    let (loss, dlogit) =
                        calculate_huber_loss_and_gradient(output[0], sample.label as f64); //TODO allow different delta values
                    (loss, vec![dlogit])
                }
            };
            epoch_loss += loss;
            // pass the error back through the model
            let _ = model
                .backward_concurrent(dlogits, &thread_pool)
                .map_err(|e| TrainingError {
                    source: e,
                    epoch,
                    sample: samples_seen,
                })?;
            model.update(options.learning_rate); // TODO implement momentum
            model.zero_gradients();
            if samples_seen % options.knot_update_interval == 0
                && samples_seen < training_data.len() - model.knot_length()
            // don't update knots at the end of an epoch, as it prevents grid extension
            {
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
        if epoch % knot_extension_interval == 0
            && next_knot_target_idx < knot_extension_targets.len()
        {
            // eprintln!(
            //     "extending knots from {} to {}",
            //     model.knot_length(),
            //     knot_extension_targets[next_knot_target_idx]
            // );
            let new_knot_target = knot_extension_targets[next_knot_target_idx];
            next_knot_target_idx += 1;
            model.set_knot_length(new_knot_target).unwrap();
        }
        epoch_loss /= training_data.len() as f64;

        let validation_loss = match validate {
            EachEpoch::ValidateModel(validation_data) => {
                validate_model(validation_data, &mut model, training_observer)
            }
            EachEpoch::DoNotValidateModel => f64::NAN,
        };

        training_observer.on_epoch_end(epoch, epoch_loss, validation_loss);
    }

    Ok(model)
}

/// Calculates the loss of the model on the provided validation data. If the model is a classification model, the cross entropy loss is calculated.
/// If the model is a regression model, the mean squared error is calculated.
///
/// Calls the [`TrainingObserver::on_sample_end`] method of the provided observer after each sample is processed.
pub fn validate_model<T: TrainingObserver>(
    validation_data: &[Sample],
    model: &mut Kan,
    observer: &T,
) -> f64 {
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
        observer.on_sample_end();
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

/// Builds a plan for extending the knots of the model over the course of training.
fn build_knot_extension_plan(
    starting_knot_length: usize,
    ideal_knot_length: usize,
    num_epochs: usize,
) -> (usize, Vec<usize>) {
    // Determine how many epochs to wait before extending the knots, and to what length the knots should be extended each time, assuming:
    // 1. we update no more frequently than once per epoch (this is negotiable, but it's what I'm going with for now)
    // 2. update by at least 1 knot each time
    // 3. we want to reach the ideal knot length before the last epoch
    // 4. we want to extend the knots by the same amount each time

    const MAX_KNOT_EXTENSIONS: usize = 5; // testing something

    if starting_knot_length >= ideal_knot_length {
        (num_epochs, vec![])
    } else {
        let knot_gap = ideal_knot_length - starting_knot_length;
        // let knots_added_per_epoch = knot_gap as f64 / (num_epochs - 1) as f64; // -1 to make sure we have max knots by the last epoch
        // if knots_added_per_epoch >= 1.0 {
        //     // we need to add at least one knot per epoch
        //     let knot_targets: Vec<usize> = (1..num_epochs)
        //         .map(|epoch_count| {
        //             starting_knot_length
        //                 + (epoch_count as f64 * knots_added_per_epoch).round() as usize
        //         })
        //         .collect();
        //     (1, knot_targets)
        // } else {
        //     // we need to wait several epochs before adding a knot
        //     let epochs_between_extensions = num_epochs / knot_gap;
        //     let knot_targets: Vec<usize> = (starting_knot_length + 1..=ideal_knot_length).collect();
        //     (epochs_between_extensions, knot_targets)
        // }
        let knot_step_size = knot_gap / MAX_KNOT_EXTENSIONS;
        let knot_update_interval = num_epochs / MAX_KNOT_EXTENSIONS;
        let knot_targets: Vec<usize> = (1..=MAX_KNOT_EXTENSIONS)
            .map(|i| starting_knot_length + i * knot_step_size)
            .collect();
        (knot_update_interval, knot_targets)
    }
}

/// Indicates whether the model should be tested against the validation data set after each epoch
pub enum EachEpoch<'a> {
    /// Test the model against the validation data set after each epoch, and report the validation loss through the [TrainingObserver]
    ValidateModel(&'a [Sample]),
    /// Do not test the model against the validation data set after each epoch
    DoNotValidateModel,
}

// EmptyObserver is basically a singleton, so there's no point in implementing any other common traits
/// An observer that does nothing when called.
/// Used for ignoring training events in the [train_model] function.
#[derive(Default)]
pub struct EmptyObserver {}
impl EmptyObserver {
    /// Create a new instance of the EmptyObserver
    pub fn new() -> Self {
        EmptyObserver {}
    }
}
impl TrainingObserver for EmptyObserver {
    fn on_epoch_end(&self, _epoch: usize, _epoch_loss: f64, _validation_loss: f64) {
        // do nothing
    }
    fn on_sample_end(&self) {
        // do nothing
    }
}

/// Indicates that an error was encountered during training
///
/// If displayed, this error will show the epoch and sample at which the error was encountered, as well as the [KanError] that caused the error.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TrainingError {
    /// The error that caused the training error
    pub source: KanError,
    /// The epoch at which the error was encountered
    pub epoch: usize,
    /// The sample within the epoch at which the error was encountered
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
