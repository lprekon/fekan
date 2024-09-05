use std::fmt;

use crate::Sample;

/// Used by the [`train_model`](crate::train_model) function to determine how the model should be trained.
#[derive(Clone, PartialEq, Debug)]
pub struct TrainingOptions<'a> {
    /// number of epochs for which to train, where an epoch is one complete pass through the training data
    pub num_epochs: usize,
    /// number of samples to pass through the model per batch. After each batch, the model's weights are updated, and the knots are adapted (see [`KanLayer::update_knots_from_samples`](crate::kan_layer::KanLayer::update_knots_from_samples) for more information about this process)
    ///
    /// for best results when the training data is small, let |training_data| % (batch_size * num_threads) = 0
    pub batch_size: usize,
    /// the adaptivity of the knots when updating them. See [`KanLayer::update_knots_from_samples`](crate::kan_layer::KanLayer::update_knots_from_samples) for more information about this process. Knots are updated after each batch
    pub knot_adaptivity: f64,
    /// the overall learning factor applied to the gradients when updating the model. This factor is applied to prediction, L1, and entropy penalties.
    pub learning_rate: f64,
    /// the amount by which to scale the L1 penalty, relative to the prediction penalty, when updating the model. The L1 penalty affects the rate at which weights are pushed to zero.
    pub l1_penalty: f64,
    /// the amount by which to scale the entropy penalty, relative to the prediction penalty, when updating the model. The entropy penalty affects the rate at which layers are pushed to favor a few edges over others
    pub entropy_penalty: f64,
    /// The lengths to which the knot vectors should be extended. Extension will happen after the epochs specified in `knot_extension_times`
    pub knot_extension_targets: Option<Vec<usize>>,
    /// The epochs (one-indexed) after which to extend the knots. Must be sorted in ascending order and equal in length to `knot_extension_targets`
    pub knot_extension_times: Option<Vec<usize>>,
    /// the epochs (one-indexed) after which to perform symbolification. Must be sorted in ascending order. See [`KanLayer::test_and_set_symbolic`](crate::kan_layer::KanLayer::test_and_set_symbolic) for more information about this process. If not set, no symbolification will occur
    pub symbolification_times: Option<Vec<usize>>,
    /// the R2 threshold for symbolification. See [`KanLayer::test_and_set_symbolic`](crate::kan_layer::KanLayer::test_and_set_symbolic) for more information about this process.
    pub symbolification_threshold: f64,
    /// The epochs (one-indexed) after which to check edges for pruning. Must be sorted in ascending order. If not set, no pruning will occur. Which edges are pruned is determined by the `pruning_threshold`
    pub pruning_times: Option<Vec<usize>>,
    /// Any edges who's average absolute output was less than this threshold during the last batch will be pruned
    pub pruning_threshold: f64,
    /// the number of threads to use when training the model. If <= 1, training will be single-threaded.
    pub num_threads: usize,
    /// whether to test the model against the validation data set after each epoch
    pub each_epoch: EachEpoch<'a>,
}

#[derive(Clone, PartialEq, Debug)]
/// Indicates whether the model should be tested against the validation data set after each epoch
pub enum EachEpoch<'a> {
    /// Test the model against the validation data set after each epoch, and log the validation loss
    ValidateModel(&'a [Sample]),
    /// Do not test the model against the validation data set after each epoch
    DoNotValidateModel,
}

impl<'a> TrainingOptions<'_> {
    /// Create a new TrainingOptions struct with the given parameters.
    /// # Errors
    /// Returns [`TrainingOptionsError`] error if...
    /// * `knot_extension_targets` is Some and `knot_extension_times` is None, or vice versa,
    /// * the lengths of `knot_extension_targets` and `knot_extension_times` are not equal.
    pub fn new(
        num_epochs: usize,
        batch_size: usize,
        knot_adaptivity: f64,
        learning_rate: f64,
        l1_penalty: f64,
        entropy_penalty: f64,
        knot_extension_targets: Option<Vec<usize>>,
        knot_extension_times: Option<Vec<usize>>,
        symbolification_times: Option<Vec<usize>>,
        symbolification_threshold: Option<f64>,
        pruning_times: Option<Vec<usize>>,
        pruning_threshold: Option<f64>,
        num_threads: usize,
        each_epoch: EachEpoch<'a>,
    ) -> Result<TrainingOptions<'a>, TrainingOptionsError> {
        if knot_extension_targets.is_some() && knot_extension_times.is_none() {
            return Err(TrainingOptionsError::MissingKnotExtensionTimes);
        }
        if knot_extension_targets.is_none() && knot_extension_times.is_some() {
            return Err(TrainingOptionsError::MissingKnotExtensionTargets);
        }
        if knot_extension_targets.is_some()
            && knot_extension_times.is_some()
            && knot_extension_targets.as_ref().unwrap().len()
                != knot_extension_times.as_ref().unwrap().len()
        {
            return Err(TrainingOptionsError::MismatchedKnotExtensionLengths {
                knot_extension_targets_length: knot_extension_targets.as_ref().unwrap().len(),
                knot_extension_times_length: knot_extension_times.as_ref().unwrap().len(),
            });
        }
        let extension_times: Option<Vec<usize>> = match knot_extension_times {
            Some(times) => {
                let mut times = times;
                times.sort();
                Some(times)
            }
            None => None,
        };
        if symbolification_times.is_some() && symbolification_threshold.is_none() {
            return Err(TrainingOptionsError::MissingSymbolificationThreshold);
        }
        if symbolification_times.is_none() && symbolification_threshold.is_some() {
            return Err(TrainingOptionsError::MissingSymbolificationTimes);
        }
        let symbol_times: Option<Vec<usize>> = match symbolification_times {
            Some(times) => {
                let mut times = times;
                times.sort();
                Some(times)
            }
            None => None,
        };
        if pruning_times.is_some() && pruning_threshold.is_none() {
            return Err(TrainingOptionsError::MissingPruningThreshold);
        }
        if pruning_times.is_none() && pruning_threshold.is_some() {
            return Err(TrainingOptionsError::MissingPruningTimes);
        }
        if pruning_threshold.is_some() && pruning_threshold.unwrap() < 0.0 {
            return Err(TrainingOptionsError::NegativePruningThrehsold);
        }
        let pruning_times: Option<Vec<usize>> = match pruning_times {
            Some(times) => {
                let mut times = times;
                times.sort();
                Some(times)
            }
            None => None,
        };
        Ok(TrainingOptions {
            num_epochs,
            batch_size,
            knot_adaptivity,
            learning_rate,
            l1_penalty,
            entropy_penalty,
            knot_extension_targets,
            knot_extension_times: extension_times,
            symbolification_times: symbol_times,
            symbolification_threshold: symbolification_threshold.unwrap_or(0.0),
            pruning_times,
            pruning_threshold: pruning_threshold.unwrap_or(0.0),
            num_threads,
            each_epoch,
        })
    }
}

impl Default for TrainingOptions<'_> {
    /// Returns a TrainingOptions struct with the following default values:
    /// * `num_epochs`: 100
    /// * `batch_size`: 100
    /// * `knot_adaptivity`: 0.1
    /// * `learning_rate`: 0.001
    /// * `l1_penalty`: 1.0
    /// * `entropy_penalty`: 1.0
    /// * `knot_extension_targets`: None
    /// * `knot_extension_times`: None
    /// * `symbolification_times`: None
    /// * `symbolification_threshold`: 0.0
    /// * `num_threads`: 1
    /// * `each_epoch`: EachEpoch::DoNotValidateModel
    fn default() -> Self {
        TrainingOptions {
            num_epochs: 100,
            batch_size: 100,
            knot_adaptivity: 0.1,
            learning_rate: 0.001,
            l1_penalty: 1.0,
            entropy_penalty: 1.0,
            knot_extension_targets: None,
            knot_extension_times: None,
            symbolification_times: None,
            symbolification_threshold: 0.0,
            pruning_times: None,
            pruning_threshold: 0.0,
            num_threads: 1,
            each_epoch: EachEpoch::DoNotValidateModel,
        }
    }
}

/// Errors that can occur when creating a new TrainingOptions struct
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingOptionsError {
    /// Knot extension targets were provided, but knot extension times were not
    MissingKnotExtensionTimes,
    /// Knot extension times were provided, but knot extension targets were not
    MissingKnotExtensionTargets,
    /// The lengths of knot extension targets and knot extension times are not equal
    MismatchedKnotExtensionLengths {
        /// The length of the knot extension targets received
        knot_extension_targets_length: usize,
        /// The length of the knot extension times received
        knot_extension_times_length: usize,
    },
    /// Symbolification times were provided, but the symbolification threshold was not
    MissingSymbolificationThreshold,
    /// Symbolification threshold was provided, but the symbolification times were not
    MissingSymbolificationTimes,
    /// Pruning times were provided, but the pruning threshold was not
    MissingPruningThreshold,
    /// Pruning threshold was provided, but the pruning times were not
    MissingPruningTimes,
    /// Pruning threshold was negative (the pruning threshold is checked against edge absolute output values, so it must be non-negative)
    NegativePruningThrehsold,
}

impl fmt::Display for TrainingOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingOptionsError::MissingKnotExtensionTimes => write!(f, "Missing knot extension times"),
            TrainingOptionsError::MissingKnotExtensionTargets => write!(f, "Missing knot extension targets"),
            TrainingOptionsError::MismatchedKnotExtensionLengths { knot_extension_targets_length, knot_extension_times_length } => write!(f, "Mismatched knot extension lengths: knot extension targets length is {}, knot extension times length is {}", knot_extension_targets_length, knot_extension_times_length),
            TrainingOptionsError::MissingSymbolificationThreshold => write!(f, "Missing symbolification threshold"),
            TrainingOptionsError::MissingSymbolificationTimes => write!(f, "Missing symbolification times"),
            TrainingOptionsError::MissingPruningThreshold => write!(f, "Missing pruning threshold"),
            TrainingOptionsError::MissingPruningTimes => write!(f, "Missing pruning times"),
            TrainingOptionsError::NegativePruningThrehsold => write!(f, "Pruning threshold must be non-negative"),
        }
    }
}

impl std::error::Error for TrainingOptionsError {}
