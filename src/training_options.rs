use std::fmt;

/// Used by the [`train_model`] function to determine how the model should be trained. See [`TrainingOptions::new`] for more information about the parameters.
#[derive(Clone, PartialEq, Debug)]
pub struct TrainingOptions {
    /// number of epochs for which to train, where an epoch is one pass through the training data
    pub num_epochs: usize,
    /// number of samples to pass through the model before updating the knots. See [kan_layer::KanLayer::update_knots_from_samples] for more information about this process.
    pub knot_update_interval: usize,
    /// the adaptivity of the knots when updating them. See [kan_layer::KanLayer::update_knots_from_samples] for more information about this process.
    pub knot_adaptivity: f64,
    /// the learning rate of the model
    pub learning_rate: f64,
    /// The lengths to which the knot vectors should be extended. Extension will happen before the epochs specified in `knot_extension_times`
    pub knot_extension_targets: Option<Vec<usize>>,
    /// The epochs (zero-indexed) after which to extend the knots. Must be sorted in ascending order and equal in length to `knot_extension_targets`
    pub knot_extension_times: Option<Vec<usize>>,
    /// the number of threads to use when training the model. If <= 1, training will be single-threaded.
    pub num_threads: usize,
}

impl TrainingOptions {
    /// Create a new TrainingOptions struct with the given parameters.
    /// # Arguments
    /// * `num_epochs` - number of epochs for which to train, where an epoch is one pass through the training data
    /// * `knot_update_interval` - number of samples to pass through the model before updating the knots. See [`crate::kan_layer::KanLayer::update_knots_from_samples`] for more information about this process.
    /// * `knot_adaptivity` - the adaptivity of the knots when updating them. See [`crate::kan_layer::KanLayer::update_knots_from_samples`] for more information about this process.
    /// * `learning_rate` - the learning rate of the model
    /// * `knot_extension_targets` - The lengths to which the knot vectors should be extended. Extension will happen before the epochs specified in `knot_extension_times`. If both `knot_extension_targets` and `knot_extension_times` are None, no extension will occur. If Some, must be equal in length to `knot_extension_times`. See [`crate::kan_layer::KanLayer::set_knot_length`] for more information about this process.
    /// * `knot_extension_times` - The epochs (zero-indexed) before which to extend the knots. Must be equal in length to `knot_extension_targets`
    /// * `num_threads` - the number of threads to use when training the model.
    /// # Errors
    /// Returns an error if `knot_extension_targets` is Some and `knot_extension_times` is None, or vice versa, or if the lengths of `knot_extension_targets` and `knot_extension_times` are not equal.
    pub fn new(
        num_epochs: usize,
        knot_update_interval: usize,
        knot_adaptivity: f64,
        learning_rate: f64,
        knot_extension_targets: Option<Vec<usize>>,
        knot_extension_times: Option<Vec<usize>>,
        num_threads: usize,
    ) -> Result<Self, TrainingOptionsError> {
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
        Ok(TrainingOptions {
            num_epochs,
            knot_update_interval,
            knot_adaptivity,
            learning_rate,
            knot_extension_targets,
            knot_extension_times: extension_times,
            num_threads,
        })
    }

    /// get the number of epochs for which to train, where an epoch is one pass through the training data
    pub fn num_epochs(&self) -> usize {
        self.num_epochs
    }

    /// get the number of samples to pass through the model before updating the knots. See [`crate::kan_layer::KanLayer::update_knots_from_samples`] for more information about this process.
    pub fn knot_update_interval(&self) -> usize {
        self.knot_update_interval
    }

    /// get the adaptivity of the knots when updating them. See [`crate::kan_layer::KanLayer::update_knots_from_samples`] for more information about this process.
    pub fn knot_adaptivity(&self) -> f64 {
        self.knot_adaptivity
    }

    /// get the learning rate of the model
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// set a knot extension plan. See [`crate::kan_layer::KanLayer::set_knot_length`] for more information about this process.
    /// # Arguments
    /// * `knot_extension_targets` - The lengths to which the knot vectors should be extended. Extension will happen before the epochs specified in `knot_extension_times`. If both `knot_extension_targets` and `knot_extension_times` are None, no extension will occur. If Some, must be equal in length to `knot_extension_times`. See [`crate::kan_layer::KanLayer::set_knot_length`] for more information about this process.
    /// * `knot_extension_times` - The epochs (zero-indexed) before which to extend the knots. Must be equal in length to `knot_extension_targets`
    pub fn set_knot_extension_plan(
        &mut self,
        knot_extension_targets: Vec<usize>,
        knot_extension_times: Vec<usize>,
    ) -> Result<(), TrainingOptionsError> {
        if knot_extension_targets.len() != knot_extension_times.len() {
            return Err(TrainingOptionsError::MismatchedKnotExtensionLengths {
                knot_extension_targets_length: knot_extension_targets.len(),
                knot_extension_times_length: knot_extension_times.len(),
            });
        }
        self.knot_extension_targets = Some(knot_extension_targets);
        let mut times = knot_extension_times;
        times.sort();
        self.knot_extension_times = Some(times);
        Ok(())
    }

    /// get the lengths to which the knot vectors should be extended, if set. Extension will happen after the epochs specified in `knot_extension_times`
    pub fn knot_extension_targets(&self) -> Option<&[usize]> {
        self.knot_extension_targets.as_deref()
    }

    /// get the epochs (zero-indexed) after which to extend the knots, if set. Must be equal in length to `knot_extension_targets`
    pub fn knot_extension_times(&self) -> Option<&[usize]> {
        self.knot_extension_times.as_deref()
    }

    /// get the number of threads to use when training the model.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for TrainingOptions {
    fn default() -> Self {
        TrainingOptions {
            num_epochs: 100,
            knot_update_interval: 100,
            knot_adaptivity: 0.1,
            learning_rate: 0.001,
            knot_extension_targets: None,
            knot_extension_times: None,
            num_threads: 1,
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
}

impl fmt::Display for TrainingOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingOptionsError::MissingKnotExtensionTimes => write!(f, "Missing knot extension times"),
            TrainingOptionsError::MissingKnotExtensionTargets => write!(f, "Missing knot extension targets"),
            TrainingOptionsError::MismatchedKnotExtensionLengths { knot_extension_targets_length, knot_extension_times_length } => write!(f, "Mismatched knot extension lengths: knot extension targets length is {}, knot extension times length is {}", knot_extension_targets_length, knot_extension_times_length),
        }
    }
}

impl std::error::Error for TrainingOptionsError {}
