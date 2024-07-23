//! Error types relating to the creation and manipulation of [`KanLayer`](crate::kan_layer::KanLayer)s

use super::spline::spline_errors::*;
use std::fmt::{self, Formatter};

#[allow(missing_docs)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum ForwardLayerError {
    /// the length of the preactivation vector passed to [`KanLayer::forward`] was not equal to the input dimension of the layer
    MissizedPreactsError { actual: usize, expected: usize },
    /// the call to [`KanLayer::forward`] resulted in NaNs in the function's output. This is usually caused by too many duplicate knots in the spline.
    /// Internal controls should prevent this situation from occuring, but this error check and type are left in for safety. If this error occurs,
    /// the only course of action is to reinitalize the layer and, as a precaution, increase the number of forward-passes between calls to [`KanLayer::update_knots_from_samples`]
    NaNsError,
}

impl std::fmt::Display for ForwardLayerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ForwardLayerError::MissizedPreactsError { actual, expected } => {
                write!(
                    f,
                    "Bad preactivation length. Expected {}, got {}",
                    expected, actual
                )
            }
            ForwardLayerError::NaNsError => {
                write!(f, "NaNs in activations")
            }
        }
    }
}

impl std::error::Error for ForwardLayerError {}

#[allow(missing_docs)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum BackwardLayerError {
    /// the length of the error vector passed to [`KanLayer::backward`] was not equal to the output dimension of the layer
    MissizedGradientError {
        actual: usize,
        expected: usize,
    },
    /// [`KanLayer::backward`] was called before [`KanLayer::forward`]. The backward pass relies on internal state set by the forward pass, so the forward pass must be called first
    BackwardBeforeForwardError,
    ReceivedNanError,
}

#[allow(unreachable_patterns)]
impl From<BackwardSplineError> for BackwardLayerError {
    fn from(e: BackwardSplineError) -> Self {
        match e {
            BackwardSplineError::BackwardBeforeForwardError => {
                BackwardLayerError::BackwardBeforeForwardError
            }
            BackwardSplineError::ReceivedNanError => BackwardLayerError::ReceivedNanError,
            _ => panic!(
                "KanLayer::backward received an unexpected error from a spline: {:?}",
                e
            ),
        }
    }
}

impl std::fmt::Display for BackwardLayerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BackwardLayerError::MissizedGradientError { actual, expected } => {
                write!(
                    f,
                    "received error vector of length {} but required vector of length {}",
                    actual, expected
                )
            }
            BackwardLayerError::BackwardBeforeForwardError => {
                write!(f, "backward called before forward")
            }
            BackwardLayerError::ReceivedNanError => {
                write!(f, "received NaNs in error vector")
            }
        }
    }
}

impl std::error::Error for BackwardLayerError {}

#[allow(missing_docs)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum UpdateLayerKnotsError {
    /// the layer has no samples to update the knot vectors with. This error is usually caused by calling [`KanLayer::update_knots_from_samples`] or [`KanLayer::set_knot_length`] - which both consume and clear the internal cache - before calling [`KanLayer::forward`] - which populates the internal cache
    NoSamplesError,
}

impl From<UpdateSplineKnotsError> for UpdateLayerKnotsError {
    fn from(e: UpdateSplineKnotsError) -> Self {
        match e {
            UpdateSplineKnotsError::ActivationsEmptyError => {
                UpdateLayerKnotsError::NoSamplesError
            }
            _ => panic!(
                "KanLayer::update_knots_from_samples received an unexpected error from a spline: {:?}",
                e
            ),
        }
    }
}

impl std::fmt::Display for UpdateLayerKnotsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            UpdateLayerKnotsError::NoSamplesError => {
                write!(f, "called an internal-cache-consuming function without first populating the cache with calls to `forward()`")
            }
        }
    }
}

impl std::error::Error for UpdateLayerKnotsError {}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum KanLayerMergeError {
    NoLayersError,
    MismatchedInputDimensionError {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MismatchedOutputDimensionError {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeSplineError {
        pos: usize,
        source: MergeSplinesError,
    },
}

impl std::fmt::Display for KanLayerMergeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            KanLayerMergeError::NoLayersError => {
                write!(f, "no layers to merge")
            }
            KanLayerMergeError::MismatchedInputDimensionError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched input dimension at {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            KanLayerMergeError::MismatchedOutputDimensionError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched output dimension at pos {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            KanLayerMergeError::MergeSplineError { pos, source } => {
                write!(f, "error merging splines at {}. {}", pos, source)
            }
        }
    }
}

impl std::error::Error for KanLayerMergeError {}

#[cfg(test)]
mod test {
    use super::super::KanLayer;
    use super::*;
    #[test]
    fn test_layer_send() {
        fn assert_send<T: Send>() {}
        assert_send::<KanLayer>();
    }

    #[test]
    fn test_layer_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KanLayer>();
    }

    #[test]
    fn test_forward_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ForwardLayerError>();
    }

    #[test]
    fn test_forward_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ForwardLayerError>();
    }

    #[test]
    fn test_backward_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BackwardLayerError>();
    }

    #[test]
    fn test_backward_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BackwardLayerError>();
    }

    #[test]
    fn test_knot_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<UpdateLayerKnotsError>();
    }

    #[test]
    fn test_knot_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<UpdateLayerKnotsError>();
    }
}
