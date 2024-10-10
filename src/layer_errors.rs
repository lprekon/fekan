//! Error types relating to the creation and manipulation of [`KanLayer`](crate::kan_layer::KanLayer)s

use bitvec::vec::BitVec;

use crate::kan_layer::edge::{edge_errors::EdgeError, Edge};
use std::fmt::{self, Formatter};

/// Represents any error returned from a KanLayer method or static function
#[derive(Debug, PartialEq, Clone)]
pub struct LayerError {
    error_kind: LayerErrorType,
    source: Option<EdgeError>,
    spline_idx: Option<usize>,
}

#[derive(Debug, PartialEq, Clone)]
enum LayerErrorType {
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
    MergeMismatchedEmbeddingDimension {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeMismatchedEmbeddingVocabSize {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeMismatchedEmbeddingFeatures {
        pos: usize,
        expected: BitVec,
        actual: BitVec,
    },
    EmbeddingFloat {
        embedded_features: BitVec,
        input_vec: Vec<f64>,
        problem_index: usize,
        problem_value: f64,
    },
}

impl LayerError {
    // Existing function for MissizedPreacts
    pub(crate) fn missized_preacts(actual: usize, expected: usize) -> Self {
        Self {
            error_kind: LayerErrorType::MissizedPreacts { actual, expected },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for NaNsInActivations
    pub(crate) fn nans_in_activations(
        spline_idx: usize,
        preacts: Vec<f64>,
        offending_spline: Edge,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::NaNsInActivations {
                preacts,
                offending_spline,
            },
            source: None,
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for MissizedGradient
    pub(crate) fn missized_gradient(actual: usize, expected: usize) -> Self {
        Self {
            error_kind: LayerErrorType::MissizedGradient { actual, expected },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for BackwardBeforeForward
    pub(crate) fn backward_before_forward(
        spline_error: Option<EdgeError>,
        spline_idx: usize,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::BackwardBeforeForward,
            source: spline_error,
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for NaNsInGradient
    pub(crate) fn nans_in_gradient() -> Self {
        Self {
            error_kind: LayerErrorType::NaNsInGradient,
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for NoSamples
    pub(crate) fn no_samples() -> Self {
        Self {
            error_kind: LayerErrorType::NoSamples,
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for SetKnotLength
    pub(crate) fn set_knot_length(spline_idx: usize, spline_error: EdgeError) -> Self {
        Self {
            error_kind: LayerErrorType::SetKnotLength,
            source: Some(spline_error),
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for MergeNoLayers
    pub(crate) fn merge_no_layers() -> Self {
        Self {
            error_kind: LayerErrorType::MergeNoLayers,
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for MergeMismatchedInputDimension
    pub(crate) fn merge_mismatched_input_dimension(
        pos: usize,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::MergeMismatchedInputDimension {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for MergeMismatchedOutputDimension
    pub(crate) fn merge_mismatched_output_dimension(
        pos: usize,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::MergeMismatchedOutputDimension {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for SplineMerge
    pub(crate) fn spline_merge(spline_idx: usize, spline_error: EdgeError) -> Self {
        Self {
            error_kind: LayerErrorType::MergeUnmergableSplines,
            source: Some(spline_error),
            spline_idx: Some(spline_idx),
        }
    }

    // Initialization function for MergeMismatchedEmbeddingDimension
    pub(crate) fn merge_mismatched_embedding_dimension(
        pos: usize,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::MergeMismatchedEmbeddingDimension {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for MergeMismatchedEmbeddingVocabSize
    pub(crate) fn merge_mismatched_vocab_size(pos: usize, expected: usize, actual: usize) -> Self {
        Self {
            error_kind: LayerErrorType::MergeMismatchedEmbeddingVocabSize {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for MergeMismatchedEmbeddingFeatures
    pub(crate) fn merge_mismatched_embedded_features(
        pos: usize,
        expected: BitVec,
        actual: BitVec,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::MergeMismatchedEmbeddingFeatures {
                pos,
                expected,
                actual,
            },
            source: None,
            spline_idx: None,
        }
    }

    // Initialization function for EmbeddingFloat
    pub(crate) fn embedding_float(
        embedded_features: BitVec,
        input_vec: Vec<f64>,
        problem_index: usize,
        problem_value: f64,
    ) -> Self {
        Self {
            error_kind: LayerErrorType::EmbeddingFloat {
                embedded_features,
                input_vec,
                problem_index,
                problem_value,
            },
            source: None,
            spline_idx: None,
        }
    }
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.error_kind {
            LayerErrorType::MissizedPreacts { actual, expected } => {
                write!(
                    f,
                    "Bad preactivation length. Expected {}, got {}",
                    expected, actual
                )
            }
            LayerErrorType::NaNsInActivations {
                preacts,
                offending_spline,
            } => {
                write!(
                    f,
                    "NaNs in activations for spline {} - preacts: {:?} - bad knots: {:?}",
                    self.spline_idx
                        .expect("NaNsInActivations error must have a spline index"),
                    preacts,
                    offending_spline.knots()
                )
            }
            LayerErrorType::MissizedGradient { actual, expected } => {
                write!(
                    f,
                    "received error vector of length {} but required vector of length {}",
                    actual, expected
                )
            }
            LayerErrorType::BackwardBeforeForward => {
                write!(f, "backward called before forward")
            }
            LayerErrorType::NaNsInGradient => {
                write!(f, "received NaNs in gradient vector during backpropogation")
            }
            LayerErrorType::SetKnotLength => {
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
            LayerErrorType::NoSamples => {
                write!(f, "called an internal-cache-consuming function without first populating the cache with calls to `forward()`")
            }
            LayerErrorType::MergeNoLayers => {
                write!(f, "no layers to merge")
            }
            LayerErrorType::MergeMismatchedInputDimension {
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
            LayerErrorType::MergeMismatchedOutputDimension {
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

            LayerErrorType::MergeUnmergableSplines {} => {
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

            LayerErrorType::MergeMismatchedEmbeddingDimension {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging layers, layer {} had a different embedding dimension than the first layer. Expected {}, got {}",
                    pos, expected, actual
                )
            }

            LayerErrorType::MergeMismatchedEmbeddingVocabSize {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging layers, layer {} had a different embedding vocab size than the first layer. Expected {}, got {}",
                    pos, expected, actual
                )
            }

            LayerErrorType::MergeMismatchedEmbeddingFeatures {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging layers, layer {} had different embedding features than the first layer. Expected {:?}, got {:?}",
                    pos, expected, actual
                )
            }
            LayerErrorType::EmbeddingFloat {
                embedded_features,
                input_vec,
                problem_index,
                problem_value,
            } => {
                write!(
                    f,
                    "embedding layer had a float in the input vector at index {} - embedded features: {:?} - input vector: {:?} - problem value: {}",
                    problem_index, embedded_features, input_vec, problem_value
                )
            }
        }
    }
}

impl std::error::Error for LayerError {
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
        assert_send::<LayerError>();
    }

    #[test]
    fn test_layer_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<LayerError>();
    }
}
