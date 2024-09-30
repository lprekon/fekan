use bitvec::vec::BitVec;
use rand::{distributions::Distribution, thread_rng};
use serde::{Deserialize, Serialize};
use statrs::distribution::Normal;

use crate::layer_errors::LayerError;

/// A layer that embeds features into a higher-dimensional space. This is useful for categorical features that need to be represented as continuous values. All embedded features must be unsigned integer values (not necessarily unsigned integer type)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    embedding_table: Vec<Vec<f64>>,
    /// Feature index that should be mapped with the embedding table have their bit set to one
    embedded_features: BitVec,
    input_dimension: usize,
    output_dimension: usize,
    embedding_dimension: usize,

    #[serde(skip)]
    embedding_gradients: Vec<Vec<f64>>,
    #[serde(skip)]
    past_inputs: Vec<Vec<f64>>,
}

/// Hyperparameters for the embedding layer.
///
/// #Example
/// see [`EmbeddingLayer::new`]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingOptions {
    /// The number of unique features that can be embedded
    pub vocab_size: usize,
    /// The number of dimensions into which each feature will be embedded. In other words, the length of the embedding vector for each feature
    pub embedding_dimension: usize,
    /// The indices of the features that should be embedded. All other features will be passed through unchanged
    pub embedded_features: Vec<usize>,
    /// The total number of features in the input vector. This is used to determine the output dimension of the embedding layer and for error checking
    pub full_input_dimension: usize,
}

impl EmbeddingLayer {
    /// Create a new embedding layer for use in a neural network
    /// # Example
    /// Create an embedding layer to handle categorical features.
    /// ```
    /// # fn get_person_age(person_id: usize) -> f64 { 0.0 }
    /// # fn get_person_gender(person_id: usize) -> f64 { 0.0 }
    /// # let person_id = 0;
    /// use fekan::embedding_layer::{EmbeddingLayer, EmbeddingOptions};
    /// # let vocab_size = 2;
    /// # let embedding_dimension = 4;
    /// # let embedded_features = vec![1];
    /// # let full_input_dimension = 2;
    /// let embedding_options = EmbeddingOptions {
    ///    vocab_size,
    ///    embedding_dimension,
    ///    embedded_features: vec![1], // embed the second feature in the input vector (i.e index 1)
    ///    full_input_dimension,
    /// };
    /// let mut embedding_layer = EmbeddingLayer::new(&embedding_options);
    /// let mut features = vec![0.0; full_input_dimension];
    /// features[0] = get_person_age(person_id); // age is a continuous feature, and can be passed directly to a neural network
    /// features[1] = get_person_gender(person_id); // gender is a categorical feature, and should be embedded before being passed to a neural network
    /// let expanded_features = embedding_layer.forward(vec![features])?;
    /// assert_eq!(expanded_features.len(), 1); // like other layers in this crate, the embedding layer operates on a batch of samples. We only passed one sample, so we expect one sample back
    /// assert_eq!(expanded_features[0].len(), full_input_dimension + (embedding_dimension - 1)); // the second element of the vector was replaced with `embedding_dimension` elements, so the output dimension should be `full_input_dimension - 1 + embedding_dimension`
    ///
    /// /* now pass expanded features to a network for training */
    /// # Ok::<(), fekan::layer_errors::LayerError>(())
    /// ```
    pub fn new(options: &EmbeddingOptions) -> Self {
        let mut embedded_features_bitvec = BitVec::repeat(false, options.full_input_dimension); // full input dimension because we need a go/no-go for every feature. You'd think this was obvious, but I screwed it up the first time
        for feature in options.embedded_features.iter() {
            embedded_features_bitvec.set(*feature, true);
        }

        let mut embedding_table = vec![vec![0.0; options.embedding_dimension]; options.vocab_size];
        let normal_distribution =
            Normal::new(0.0, 0.001).expect("Unable to create normal distribution"); // Empirically best way to initialize the embedding table, per arXiv:1711.09160
        let mut randomness = thread_rng();
        for i in 0..options.vocab_size {
            for j in 0..options.embedding_dimension {
                embedding_table[i][j] = normal_distribution.sample(&mut randomness);
            }
        }

        let embedding_gradients = vec![vec![0.0; options.embedding_dimension]; options.vocab_size];
        let output_dimension = options.full_input_dimension
            + (options.embedding_dimension - 1) * embedded_features_bitvec.count_ones() as usize;
        Self {
            embedding_table,
            embedded_features: embedded_features_bitvec,
            input_dimension: options.full_input_dimension,
            output_dimension: output_dimension,
            embedding_dimension: options.embedding_dimension,
            embedding_gradients,
            past_inputs: Vec::new(),
        }
    }

    /// get the output dimension of the embedding layer
    ///
    /// `output_dimension = full_input_dimension + (embedding_dimension - 1) * number_of_embedded_features`
    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    /// Create an expanded input vector by replacing values at the indices specified in `embedded_features` with the corresponding row in the embedding table
    ///
    /// # Example
    /// see [`EmbeddingLayer::new`] for an example creating and forwarding through an embedding layer
    ///
    /// # Errors
    /// Returns a `LayerError` if
    /// * the input vector is not the correct size
    /// * A feature value at a to-be-embedded index is not an integer (i.e has a fractional part)
    pub fn forward(&mut self, preacts: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, LayerError> {
        if preacts.iter().any(|x| x.len() != self.input_dimension) {
            return Err(LayerError::missized_preacts(
                preacts[0].len(),
                self.input_dimension,
            ));
        }
        // Store the preacts for backpropagation
        self.past_inputs = preacts;
        let mut expanded_input =
            vec![Vec::with_capacity(self.output_dimension); self.past_inputs.len()];
        for sample_idx in 0..self.past_inputs.len() {
            let sample = &self.past_inputs[sample_idx];
            for i in 0..sample.len() {
                if self.embedded_features[i] {
                    if sample[i] as usize as f64 != sample[i] {
                        return Err(LayerError::embedding_float(
                            self.embedded_features.clone(),
                            sample.clone(),
                            i,
                            sample[i],
                        ));
                    }
                    expanded_input[sample_idx]
                        .extend_from_slice(&self.embedding_table[sample[i] as usize]);
                } else {
                    expanded_input[sample_idx].push(sample[i]);
                }
            }
            assert_eq!(
                expanded_input[sample_idx].len(),
                self.output_dimension,
                "Sample {} is length {} but should have expanded to {}",
                sample_idx,
                expanded_input[sample_idx].len(),
                self.output_dimension
            );
        }
        Ok(expanded_input)
    }

    /// As [EmbeddingLayer::forward], but does not store the preacts for backpropagation.
    pub fn infer(&self, preacts: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LayerError> {
        if preacts.iter().any(|x| x.len() != self.input_dimension) {
            return Err(LayerError::missized_preacts(
                preacts[0].len(),
                self.input_dimension,
            ));
        }
        let mut expanded_input = Vec::with_capacity(preacts.len());
        for sample in preacts {
            let mut expanded_sample = Vec::with_capacity(self.output_dimension);
            for i in 0..sample.len() {
                if self.embedded_features[i] {
                    expanded_sample.extend_from_slice(&self.embedding_table[sample[i] as usize]);
                } else {
                    expanded_sample.push(sample[i]);
                }
            }
            assert_eq!(
                expanded_sample.len(),
                self.output_dimension,
                "Sample is length {} but should have expanded to {}",
                expanded_sample.len(),
                self.output_dimension
            );
            expanded_input.push(expanded_sample);
        }
        Ok(expanded_input)
    }

    /// Backpropagate the error through the embedding layer, accumulating gradients for those embedding rows that were used in the corresponding forward pass.
    ///
    /// # Example
    /// ```
    /// use fekan::embedding_layer::{EmbeddingLayer, EmbeddingOptions};
    /// let vocab_size = 3;
    /// let embedding_dimension = 4;
    /// let full_input_dimension = 2;
    /// let embedding_options = EmbeddingOptions {
    ///     vocab_size,
    ///     embedding_dimension,
    ///     embedded_features: vec![1],
    ///     full_input_dimension,
    /// };
    /// let mut embedding = EmbeddingLayer::new(&embedding_options);
    /// let input = vec![vec![0.0, 1.0]];
    /// let _ = embedding.forward(input)?;
    /// let error = vec![vec![1.0; embedding.output_dimension()]];
    /// let _ = embedding.backward(error)?; // only row 1 of the embedding table will have a nonzero gradient, because that's the only row that was used in the forward pass
    ///  
    ///
    /// /* now update the embedding table with the accumulated gradients */
    /// let learning_rate = 0.01;
    /// embedding.update(learning_rate);
    /// embedding.zero_gradients(); // clear the gradients for the next batch
    ///
    /// /* now continue training */
    /// # Ok::<(), fekan::layer_errors::LayerError>(())
    /// ```
    /// # Errors
    /// Returns a [`LayerError`] if
    /// * The error vector is not the correct size
    /// * The number of samples passed to [`EmbeddingLayer::forward`] is not the same as the number of gradients passed to `backward`
    pub fn backward(&mut self, error: Vec<Vec<f64>>) -> Result<(), LayerError> {
        // return an empty Ok because there's no need for gradients earlier than this
        if error.iter().any(|x| x.len() != self.output_dimension) {
            return Err(LayerError::missized_gradient(
                error[0].len(),
                self.output_dimension,
            ));
        }
        assert_eq!(
            error.len(),
            self.past_inputs.len(),
            "Sorry - for now, please don't call Kan::forward() more than once before clearing state. We have {} input samples and {} gradients",
            self.past_inputs.len(),
            error.len()
        );

        // for each sample in the batch...
        for (input_vec, gradient_vec) in self.past_inputs.iter().zip(error.iter()) {
            let mut gradient_idx = 0;
            // for each feature in the sample...
            for input_idx in 0..input_vec.len() {
                // check if it's an embedded feature
                if self.embedded_features[input_idx] {
                    // this feature was embedded, therefore we need to accumulate the gradients for the next `embedding_dimension` indices in the gradient vector for row `input_vec[input_idx]` in the embedding table
                    let embedding_table_row =
                        &mut self.embedding_gradients[input_vec[input_idx] as usize];
                    for i in 0..self.embedding_dimension {
                        embedding_table_row[i] += gradient_vec[gradient_idx];
                        gradient_idx += 1;
                    }
                } else {
                    // this feature was not embedded, therefore we ignore the gradient value at this index and continue
                    gradient_idx += 1;
                }
            }
        }

        Ok(())
    }

    /// Update the embeddings using the accumulated gradients
    ///
    /// This function relies on the gradients accumulated during [EmbeddingLayer::backward]
    /// # Example
    /// see [`EmbeddingLayer::backward`]
    pub fn update(&mut self, learning_rate: f64) {
        for (embedding_row, gradient_row) in self
            .embedding_table
            .iter_mut()
            .zip(self.embedding_gradients.iter())
        {
            for (embedding_val, gradient_val) in embedding_row.iter_mut().zip(gradient_row.iter()) {
                *embedding_val -= learning_rate * *gradient_val;
            }
        }
    }

    /// Zero out the gradients for the embedding table
    ///
    /// # Example
    /// see [`EmbeddingLayer::backward`]
    pub fn zero_gradients(&mut self) {
        for embedding_row in self.embedding_gradients.iter_mut() {
            for gradient_val in embedding_row.iter_mut() {
                *gradient_val = 0.0;
            }
        }
    }

    /// Clear the samples stored during the forward pass.
    pub fn clear_samples(&mut self) {
        self.past_inputs.clear();
    }

    /// Merge multiple embedding layers into a single layer. All layers to be merged must be identical in terms of input dimension, output dimension, embedding dimension, vocab size, and embedded features. The only difference should be the values of the embeddings themselves
    ///
    /// Usage is identical to [`KanLayer::forward`](crate::kan_layer::KanLayer::merge_layers)
    ///
    /// # Example
    /// see [`KanLayer::merge_layers`](crate::kan_layer::KanLayer::merge_layers)
    ///
    /// # Errors
    /// Returns a [`LayerError`] if
    /// * The input dimensions of the layers to be merged are not the same
    /// * The output dimensions of the layers to be merged are not the same
    /// * The embedding dimensions of the layers to be merged are not the same
    /// * The vocab sizes of the layers to be merged are not the same
    /// * The embedded features of the layers to be merged are not the same
    pub fn merge_layers(layers_to_merge: &[&EmbeddingLayer]) -> Result<EmbeddingLayer, LayerError> {
        let expected_true_input_size = layers_to_merge[0].input_dimension;
        let expected_true_output_size = layers_to_merge[0].output_dimension;
        let expected_embedding_dimension = layers_to_merge[0].embedding_dimension;
        let expected_vocab_size = layers_to_merge[0].embedding_table.len();
        let expected_embedded_features = layers_to_merge[0].embedded_features.clone();

        let mut merged_embedding_table =
            vec![vec![0.0; expected_embedding_dimension]; expected_vocab_size];
        for i in 1..layers_to_merge.len() {
            let layer = &layers_to_merge[i];
            if layer.input_dimension != expected_true_input_size {
                return Err(LayerError::merge_mismatched_input_dimension(
                    i,
                    expected_true_input_size,
                    layer.input_dimension,
                ));
            }
            if layer.output_dimension != expected_true_output_size {
                return Err(LayerError::merge_mismatched_output_dimension(
                    i,
                    expected_true_output_size,
                    layer.output_dimension,
                ));
            }
            if layer.embedding_dimension != expected_embedding_dimension {
                return Err(LayerError::merge_mismatched_embedding_dimension(
                    i,
                    expected_embedding_dimension,
                    layer.embedding_dimension,
                ));
            }
            if layer.embedding_table.len() != expected_vocab_size {
                return Err(LayerError::merge_mismatched_vocab_size(
                    i,
                    expected_vocab_size,
                    layer.embedding_table.len(),
                ));
            }
            if layer.embedded_features != expected_embedded_features {
                return Err(LayerError::merge_mismatched_embedded_features(
                    i,
                    expected_embedded_features.clone(),
                    layer.embedded_features.clone(),
                ));
            }
            for (merged_row, layer_row) in merged_embedding_table
                .iter_mut()
                .zip(layer.embedding_table.iter())
            {
                for (merged_val, layer_val) in merged_row.iter_mut().zip(layer_row.iter()) {
                    *merged_val += *layer_val;
                }
            }
        }
        for merged_row in merged_embedding_table.iter_mut() {
            for merged_val in merged_row.iter_mut() {
                *merged_val /= layers_to_merge.len() as f64;
            }
        }

        Ok(EmbeddingLayer {
            embedding_table: merged_embedding_table,
            embedded_features: expected_embedded_features,
            input_dimension: expected_true_input_size,
            output_dimension: expected_true_output_size,
            embedding_dimension: expected_embedding_dimension,
            embedding_gradients: vec![vec![0.0; expected_embedding_dimension]; expected_vocab_size],
            past_inputs: Vec::new(),
        })
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_build_embedding_table() {
        let vocab_size = 3;
        let embedding_dimension = 4;
        let embedding_options = EmbeddingOptions {
            vocab_size,
            embedding_dimension,
            embedded_features: vec![1],
            full_input_dimension: 3,
        };
        let embed = EmbeddingLayer::new(&embedding_options);
        assert_eq!(embed.embedding_table.len(), vocab_size);
        assert_eq!(embed.embedding_table[0].len(), embedding_dimension);
        assert_eq!(embed.embedded_features.len(), 2); // the highest index in the embedded features + 1
        assert_eq!(embed.embedded_features.count_ones(), 1);
        assert_eq!(embed.output_dimension, 6);
    }

    #[test]
    fn test_embedding_forward() {
        let vocab_size = 3;
        let embedding_dimension = 4;
        let full_input_dimension = 2;
        let embedding_options = EmbeddingOptions {
            vocab_size,
            embedding_dimension,
            embedded_features: vec![1],
            full_input_dimension: full_input_dimension,
        };
        let mut embedding = EmbeddingLayer::new(&embedding_options);
        let input = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 0.0]];
        let expanded_input = embedding.forward(input.clone()).unwrap();
        assert_eq!(expanded_input.len(), input.len());
        assert_eq!(
            expanded_input[0].len(),
            full_input_dimension + embedding_dimension - 1
        );
    }

    #[test]
    fn test_embedding_forward_and_backward() {
        let vocab_size = 3;
        let embedding_dimension = 4;
        let full_input_dimension = 2;
        let embedding_options = EmbeddingOptions {
            vocab_size,
            embedding_dimension,
            embedded_features: vec![1],
            full_input_dimension: full_input_dimension,
        };
        let mut embedding = EmbeddingLayer::new(&embedding_options);
        let input = vec![vec![0.0, 1.0]];
        let _ = embedding.forward(input.clone()).unwrap();
        let error = vec![vec![1.0, 1.0, 1.0, 1.0, 1.0]];
        let _ = embedding.backward(error.clone()).unwrap();
        assert_eq!(
            embedding.embedding_gradients[0],
            vec![0.0; embedding_dimension]
        ); // we didn't use the 0th vocab index, so the gradient should be zero
        assert_ne!(
            embedding.embedding_gradients[1],
            vec![0.0; embedding_dimension]
        ); // we used the 1st vocab index, so the gradient should be nonzero
        assert_eq!(
            embedding.embedding_gradients[2],
            vec![0.0; embedding_dimension]
        ); // we didn't use the 2nd vocab index, so the gradient should be zero
    }

    #[test]
    fn test_embed_float() {
        let vocab_size = 3;
        let embedding_dimension = 4;
        let full_input_dimension = 2;
        let embedding_options = EmbeddingOptions {
            vocab_size,
            embedding_dimension,
            embedded_features: vec![1],
            full_input_dimension: full_input_dimension,
        };
        let mut embedding = EmbeddingLayer::new(&embedding_options);
        let input = vec![vec![0.5, 0.5]];
        let result = embedding.forward(input.clone());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("embedding layer had a float"));
    }
}
