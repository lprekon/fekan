use crate::kan_layer::kan_layer_errors::KanLayerError;

use super::ModelType;

/// An error ocurring during the operation of a Kan model
///
/// Displaying the error will show the index of the layer that encountered the error, and the error itself
#[derive(Debug, Clone, PartialEq)]
pub struct KanError {
    error_kind: KanErrorType,
    /// the error that occurred
    source: Option<KanLayerError>,
    /// the index of the layer that encountered the error
    layer_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KanErrorType {
    Forward,
    Backward,
    UpdateKnots,
    SetKnotLength,
    MergeMismatchedModelType {
        pos: usize,
        expected: ModelType,
        actual: ModelType,
    },
    MergeMismatchedClassMap {
        pos: usize,
        expected: Option<Vec<String>>,
        actual: Option<Vec<String>>,
    },
    MergeMismatchedDepthModel {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeUnmergableLayers,
}

impl KanError {
    pub fn forward(source: KanLayerError, layer_index: usize) -> Self {
        Self {
            error_kind: KanErrorType::Forward,
            source: Some(source),
            layer_index: Some(layer_index),
        }
    }

    pub fn backward(source: KanLayerError, layer_index: usize) -> Self {
        Self {
            error_kind: KanErrorType::Backward,
            source: Some(source),
            layer_index: Some(layer_index),
        }
    }

    pub fn update_knots(source: KanLayerError, layer_index: usize) -> Self {
        Self {
            error_kind: KanErrorType::UpdateKnots,
            source: Some(source),
            layer_index: Some(layer_index),
        }
    }

    pub fn set_knot_length(source: KanLayerError, layer_index: usize) -> Self {
        Self {
            error_kind: KanErrorType::SetKnotLength,
            source: Some(source),
            layer_index: Some(layer_index),
        }
    }

    pub fn merge_mismatched_model_type(pos: usize, expected: ModelType, actual: ModelType) -> Self {
        Self {
            error_kind: KanErrorType::MergeMismatchedModelType {
                pos,
                expected,
                actual,
            },
            source: None,
            layer_index: None,
        }
    }

    pub fn merge_unmergable_layers(source: KanLayerError, layer_index: usize) -> Self {
        Self {
            error_kind: KanErrorType::MergeUnmergableLayers,
            source: Some(source),
            layer_index: Some(layer_index),
        }
    }

    pub fn merge_mismatched_class_map(
        pos: usize,
        expected: Option<Vec<String>>,
        actual: Option<Vec<String>>,
    ) -> Self {
        Self {
            error_kind: KanErrorType::MergeMismatchedClassMap {
                pos,
                expected,
                actual,
            },
            source: None,
            layer_index: None,
        }
    }

    pub fn merge_mismatched_depth_model(pos: usize, expected: usize, actual: usize) -> Self {
        Self {
            error_kind: KanErrorType::MergeMismatchedDepthModel {
                pos,
                expected,
                actual,
            },
            source: None,
            layer_index: None,
        }
    }
}

impl std::fmt::Display for KanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.error_kind {
            KanErrorType::Forward => {
                write!(
                    f,
                    "Error in forward pass at layer {}: {}",
                    self.layer_index.unwrap(),
                    self.source.as_ref().unwrap()
                )
            }
            KanErrorType::Backward => {
                write!(
                    f,
                    "Error in backward pass at layer {}: {}",
                    self.layer_index.unwrap(),
                    self.source.as_ref().unwrap()
                )
            }
            KanErrorType::UpdateKnots => {
                write!(
                    f,
                    "Error updating knots from samples in layer {}: {}",
                    self.layer_index.unwrap(),
                    self.source.as_ref().unwrap()
                )
            }
            KanErrorType::SetKnotLength => {
                write!(
                    f,
                    "Error setting knot length in layer {}: {}",
                    self.layer_index.unwrap(),
                    self.source.as_ref().unwrap()
                )
            }
            KanErrorType::MergeMismatchedModelType {
                pos,
                actual,
                expected,
            } => {
                write!(
                    f,
                    "while merging models, model {} had a different model type than the first model. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            KanErrorType::MergeMismatchedClassMap {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging models, model {} had a different class map than the first model. Expected {:?}, got {:?}", pos, expected, actual
                )
            }
            KanErrorType::MergeMismatchedDepthModel {
                pos,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "while merging models, model {} had a different number of layers than the first model. Expected {}, got {}",
                    pos, expected, actual
                )
            }
            KanErrorType::MergeUnmergableLayers => {
                write!(
                    f,
                    "while merging models, unable to merge layer {}: {}",
                    self.layer_index
                        .expect("MergeUnmergableLayers error must have a layer index"),
                    self.source
                        .as_ref()
                        .expect("MergeUnmergableLayers error must have a source layer error")
                )
            }
        }
    }
}

impl std::error::Error for KanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.source {
            Some(source) => Some(source),
            None => None,
        }
    }
}
