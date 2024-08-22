//! Error types relating to the creation and manipulation of [`KanLayer`](crate::kan_layer::KanLayer)s

use super::edge::{edge_errors::EdgeError, Edge};
use std::fmt::{self, Formatter};

/// Represents any error returned from a KanLayer method or static function
#[derive(Debug, PartialEq, Clone)]
pub struct KanLayerError {
    error_kind: KanLayerErrorType,
    source: Option<EdgeError>,
    spline_idx: Option<usize>,
}

#[derive(Debug, PartialEq, Clone)]
enum KanLayerErrorType {
    MissizedPreacts {
        actual: usize,
        expected: usize,
    },
    NaNsInActivations {
        preacts: Vec<f64>,
        offending_spline: Edge,
    },
    MissizedGradient {
        actual: usize,
        expected: usize,
    },
    BackwardBeforeForward,
    NaNsInGradient,
    NoSamples,
    SetKnotLength,
    MergeNoLayers,
    MergeMismatchedInputDimension {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeMismatchedOutputDimension {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeUnmergableSplines,
}

impl KanLayerError {
    // Existing function for MissizedPreacts
    pub(super) fn missized_preacts(actual: usize, expected: usize) -> Self {
        Self {
            error_kind: KanLayerErrorType::MissizedPreacts { actual, expected },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for NaNsInActivations
    pub(super) fn nans_in_activations(
        spline_idx: usize,
        preacts: Vec<f64>,
        offending_spline: Edge,
    ) -> Self {
        Self {
            error_kind: KanLayerErrorType::NaNsInActivations {
                preacts,
                offending_spline,
            },
            source: None,
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for MissizedGradient
    pub(super) fn missized_gradient(actual: usize, expected: usize) -> Self {
        Self {
            error_kind: KanLayerErrorType::MissizedGradient { actual, expected },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for BackwardBeforeForward
    pub(super) fn backward_before_forward(
        spline_error: Option<EdgeError>,
        spline_idx: usize,
    ) -> Self {
        Self {
            error_kind: KanLayerErrorType::BackwardBeforeForward,
            source: spline_error,
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for NaNsInGradient
    pub(super) fn nans_in_gradient() -> Self {
        Self {
            error_kind: KanLayerErrorType::NaNsInGradient,
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for NoSamples
    pub(super) fn no_samples() -> Self {
        Self {
            error_kind: KanLayerErrorType::NoSamples,
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for SetKnotLength
    pub(super) fn set_knot_length(spline_idx: usize, spline_error: EdgeError) -> Self {
        Self {
            error_kind: KanLayerErrorType::SetKnotLength,
            source: Some(spline_error),
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for MergeNoLayers
    pub(super) fn merge_no_layers() -> Self {
        Self {
            error_kind: KanLayerErrorType::MergeNoLayers,
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for MergeMismatchedInputDimension
    pub(super) fn merge_mismatched_input_dimension(
        pos: usize,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self {
            error_kind: KanLayerErrorType::MergeMismatchedInputDimension {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for MergeMismatchedOutputDimension
    pub(super) fn merge_mismatched_output_dimension(
        pos: usize,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self {
            error_kind: KanLayerErrorType::MergeMismatchedOutputDimension {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for SplineMerge
    pub(super) fn spline_merge(spline_idx: usize, spline_error: EdgeError) -> Self {
        Self {
            error_kind: KanLayerErrorType::MergeUnmergableSplines,
            source: Some(spline_error),
            spline_idx: Some(spline_idx),
        }
    }
}

impl fmt::Display for KanLayerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.error_kind {
            KanLayerErrorType::MissizedPreacts { actual, expected } => {
                write!(
                    f,
                    "Bad preactivation length. Expected {}, got {}",
                    expected, actual
                )
            }
            KanLayerErrorType::NaNsInActivations {
                preacts,
                offending_spline,
            } => {
                write!(
                    f,
                    "NaNs in activations for spline {} - actications: {:?} - bad knots: {:?} - bad control points: {:?}",
                    self.spline_idx
                        .expect("NaNsInActivations error must have a spline index"),
                    preacts,
                    offending_spline.knots().collect::<Vec<&f64>>(), offending_spline.control_points().collect::<Vec<&f64>>()
                )
            }
            KanLayerErrorType::MissizedGradient { actual, expected } => {
                write!(
                    f,
                    "received error vector of length {} but required vector of length {}",
                    actual, expected
                )
            }
            KanLayerErrorType::BackwardBeforeForward => {
                write!(f, "backward called before forward")
            }
            KanLayerErrorType::NaNsInGradient => {
                write!(f, "received NaNs in gradient vector during backpropogation")
            }
            KanLayerErrorType::SetKnotLength => {
                write!(
                    f,
                    "setting layer knot length resulted in error at spline {} - {}",
                    self.spline_idx
                        .expect("SetKnotLength error must have a spline index"),
                    self.source
                        .as_ref()
                        .expect("SetKnotLength error must have a source spline error")
                )
            }
            KanLayerErrorType::NoSamples => {
                write!(f, "called an internal-cache-consuming function without first populating the cache with calls to `forward()`")
            }
            KanLayerErrorType::MergeNoLayers => {
                write!(f, "no layers to merge")
            }
            KanLayerErrorType::MergeMismatchedInputDimension {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging layers, layer {} had a different input dimension than the first layer. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            KanLayerErrorType::MergeMismatchedOutputDimension {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging layers, layer {} had a different output dimension than the first layer. Expected {}, got {}",
                    pos, expected, actual
                )
            }

            KanLayerErrorType::MergeUnmergableSplines {} => {
                write!(
                    f,
                    "error merging splines at index {} - {:?}",
                    self.spline_idx
                        .expect("SplineMerge error must have a spline index"),
                    self.source
                        .as_ref()
                        .expect("SplineMerge error must have a source spline error")
                )
            }
        }
    }
}

impl std::error::Error for KanLayerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.source {
            Some(source) => Some(source),
            None => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_layer_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<KanLayerError>();
    }

    #[test]
    fn test_layer_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KanLayerError>();
    }
}
