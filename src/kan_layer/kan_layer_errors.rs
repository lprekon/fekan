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

/// Errors that can occur when merging multiple layers into a single layer
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum MergeLayerError {
    /// no layers were provided to merge
    NoLayersError,
    /// the input dimension of the layer at the given position did not match the input dimension of the first layer
    MismatchedInputDimensionError {
        /// the position of the layer whose input dimension differed
        pos: usize,
        /// the expected input dimension - i.e the input dimension of the first layer
        expected: usize,
        /// the actual input dimension of the layer at the given position
        actual: usize,
    },
    /// the output dimension of the layer at the given position did not match the output dimension of the first layer
    MismatchedOutputDimensionError {
        /// the position of the layer whose output dimension differed
        pos: usize,
        /// the expected output dimension - i.e the output dimension of the first layer
        expected: usize,
        /// the actual output dimension of the layer at the given position
        actual: usize,
    },
    /// The degrees of the splines in the layers differed
    MismatchedDegreeError {
        /// the position of the layer whose splines had a different degree
        pos: usize,
        /// the expected degree - i.e the degree of the first layer
        expected: usize,
        /// the actual degree of the splines in the layer at the given position
        actual: usize,
    },
    /// The number of control points in the splines in the layers differed
    MismatchedControlPointCountError {
        /// the position of the layer whose splines had a different number of control points
        pos: usize,
        /// the expected number of control points - i.e the number of control points in the splines of the first layer
        expected: usize,
        /// the actual number of control points in the splines of the layer at the given position
        actual: usize,
    },
    /// The number of knots in the splines in the layers differed
    MismatchedKnotCountError {
        /// the position of the layer whose splines had a different number of knots
        pos: usize,
        /// the expected number of knots - i.e the number of knots in the splines of the first layer
        expected: usize,
        /// the actual number of knots in the splines of the layer at the given position
        actual: usize,
    },
    /// The layers provided to merge had no splines. equivalent to input_dimension == 0 or output_dimension == 0
    EmptyLayerError,
}

impl std::fmt::Display for MergeLayerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MergeLayerError::NoLayersError => {
                write!(f, "no layers to merge")
            }
            MergeLayerError::MismatchedInputDimensionError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched input dimension at layer {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            MergeLayerError::MismatchedOutputDimensionError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched output dimension at layer {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            MergeLayerError::MismatchedControlPointCountError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched spline control point count at layer {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            MergeLayerError::MismatchedDegreeError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched spline degree at layer {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            MergeLayerError::MismatchedKnotCountError {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "mismatched spline knot count at layer {}. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            MergeLayerError::EmptyLayerError => {
                write!(f, "no splines to merge")
            }
        }
    }
}

impl std::error::Error for MergeLayerError {}

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
