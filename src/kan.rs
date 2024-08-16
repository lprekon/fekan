pub mod kan_error;

use std::collections::VecDeque;

use kan_error::KanError;
use log::{debug, trace};

use crate::kan_layer::{KanLayer, KanLayerOptions};

use serde::{Deserialize, Serialize};

/// A full neural network model, consisting of multiple Kolmogorov-Arnold layers
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Kan {
    /// the layers of the model
    pub layers: Vec<KanLayer>,
    /// the type of model. This field is metadata and does not affect the operation of the model, though it is used elsewhere in the crate. See [`fekan::train_model()`](crate::train_model) for an example
    model_type: ModelType, // determined how the output is interpreted, and what the loss function ought to be
    /// A map of class names to node indices. Only used if the model is a classification model or multi-output regression model.
    class_map: Option<Vec<String>>,
}

/// Hyperparameters for a Kan model
///
/// # Example
/// see [Kan::new]
///
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct KanOptions {
    /// the number of input features the model should accept
    pub input_size: usize,
    /// the sizes of the layers to use in the model, including the output layer
    pub layer_sizes: Vec<usize>,
    /// the degree of the b-splines to use in each layer
    pub degree: usize,
    /// the number of coefficients to use in the b-splines in each layer
    pub coef_size: usize,
    /// the type of model to create. This field is metadata and does not affect the operation of the model, though it is used by [`fekan::train_model()`](crate::train_model) to determine the proper loss function
    pub model_type: ModelType,
    /// A list of human-readable names for the output nodes.
    /// The length of this vector should be equal to the number of output nodes in the model (the last number in `layer_sizes`), or else behavior is undefined
    pub class_map: Option<Vec<String>>,
}

/// Metadata suggesting how the model's output ought to be interpreted
///
/// For information on how model type can affect training, see [`train_model()`](crate::train_model)
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// For models designed to assign a discreet class to an input. For example, determining if an image contains a cat or a dog
    Classification,
    /// For models design to predict a continuous value. For example, predicting the price of a house
    Regression,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ModelType::Classification => write!(f, "Classification"),
            ModelType::Regression => write!(f, "Regression"),
        }
    }
}

impl Kan {
    /// creates a new Kan model with the given hyperparameters
    ///
    /// # Example
    /// Create a regression model with 5 input features, 2 hidden layers of size 4 and 3, and 1 output feature, using degree 3 b-splines with 6 coefficients per spline
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType};
    ///
    /// let options = KanOptions {
    ///     input_size: 5,
    ///     layer_sizes: vec![4, 3, 1],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Regression,
    ///     class_map: None,
    /// };
    /// let mut model = Kan::new(&options);
    ///```
    pub fn new(options: &KanOptions) -> Self {
        let mut layers = Vec::with_capacity(options.layer_sizes.len());
        let mut prev_size = options.input_size;
        for &size in options.layer_sizes.iter() {
            layers.push(KanLayer::new(&KanLayerOptions {
                input_dimension: prev_size,
                output_dimension: size,
                degree: options.degree,
                coef_size: options.coef_size,
            }));
            prev_size = size;
        }
        Kan {
            layers,
            model_type: options.model_type,
            class_map: options.class_map.clone(),
        }
    }

    /// returns the type of the model
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// returns the class map of the model, if it has one
    pub fn class_map(&self) -> Option<&Vec<String>> {
        self.class_map.as_ref()
    }

    /// Returns the index of the output node that corresponds to the given label.
    ///
    /// Returns None if the label is not found in the model's class map, or if the model does not have a class map
    ///
    /// # Example
    /// creating a model with a class map
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType};
    /// let my_class_map = vec!["cat".to_string(), "dog".to_string()];
    /// let options = KanOptions {
    ///     input_size: 5,
    ///     layer_sizes: vec![4, 2],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Regression,
    ///     class_map: Some(my_class_map),
    /// };
    /// let model = Kan::new(&options);
    /// assert_eq!(model.label_to_node("cat"), Some(0));
    /// assert_eq!(model.label_to_node("dog"), Some(1));
    /// assert_eq!(model.label_to_node("fish"), None);
    /// ```
    /// Using a model's class map during training to determine the index of node that should have had the highest value
    /// ```
    /// # use fekan::kan::{Kan, KanOptions, ModelType};
    /// # let my_class_map = vec!["cat".to_string(), "dog".to_string()];
    /// # let options = KanOptions {
    /// #    input_size: 5,
    /// #    layer_sizes: vec![4, 2],
    /// #    degree: 3,
    /// #    coef_size: 6,
    /// #    model_type: ModelType::Regression,
    /// #    class_map: Some(my_class_map),
    /// # };
    /// # let mut model = Kan::new(&options);
    /// # let feature_data = vec![0.5, 0.4, 0.5, 0.5, 0.4];
    /// # let label = "cat";
    /// # fn cross_entropy_loss(output: Vec<f64>, expected_highest_node: usize) -> f64 {0.0}
    /// /* within your custom training function */
    /// let logits: Vec<f64> = model.forward(feature_data)?;
    /// let expected_highest_node: usize = model.label_to_node(label).unwrap();
    /// let loss: f64 = cross_entropy_loss(logits, expected_highest_node);
    /// # Ok::<(), fekan::kan::kan_error::KanError>(())
    /// ```
    pub fn label_to_node(&self, label: &str) -> Option<usize> {
        if let Some(class_map) = &self.class_map {
            class_map.iter().position(|x| x == label)
        } else {
            None
        }
    }

    /// Returns the label for the output node at the given index.
    ///
    /// Returns None if the index is out of bounds, or if the model does not have a class map
    ///
    /// # Example
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType};
    /// let class_map = vec!["cat".to_string(), "dog".to_string()];
    /// let options = KanOptions {
    ///     input_size: 5,
    ///     layer_sizes: vec![4, 2],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Regression,
    ///     class_map: Some(class_map),
    /// };
    /// let model = Kan::new(&options);
    /// assert_eq!(model.node_to_label(0), Some("cat"));
    /// assert_eq!(model.node_to_label(1), Some("dog"));
    /// assert_eq!(model.node_to_label(2), None);
    /// ```
    /// Using a model's class map during inference to interpret the output of a classifier
    /// ```
    /// # use fekan::kan::{Kan, KanOptions, ModelType};
    /// # let my_class_map = vec!["cat".to_string(), "dog".to_string()];
    /// # let options = KanOptions {
    /// #    input_size: 5,
    /// #    layer_sizes: vec![4, 2],
    /// #    degree: 3,
    /// #    coef_size: 6,
    /// #    model_type: ModelType::Regression,
    /// #    class_map: Some(my_class_map),
    /// # };
    /// # let model = Kan::new(&options);
    /// # let feature_data = vec![0.5, 0.4, 0.5, 0.5, 0.4];
    /// /* using an already trained model... */
    /// let logits: Vec<f64> = model.infer(feature_data)?;
    /// let highest_node: usize = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    /// let label: &str = model.node_to_label(highest_node).unwrap();
    /// println!("The model predicts the input is a {}", label);
    /// Ok::<(), fekan::kan::kan_error::KanError>(())
    /// ```
    pub fn node_to_label(&self, node: usize) -> Option<&str> {
        if let Some(class_map) = &self.class_map {
            class_map.get(node).map(|x| x.as_str())
        } else {
            None
        }
    }

    /// Forward-propogate the input through the model, by calling [`KanLayer::forward`] method of the first layer on the input,
    /// then calling the `forward` method of each subsequent layer with the output of the previous layer,
    /// returning the output of the final layer.
    ///
    /// This method accumulates internal state in the model needed for training. For inference or validation, use [`Kan::infer`], which does not accumulate state and is more efficient
    ///
    /// # Errors
    /// returns a [`KanError`] if any layer returns an error.
    /// See [`KanLayer::forward`] for more information
    ///
    /// # Example
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType, kan_error::KanError};
    /// let input_size = 5;
    /// let output_size = 3;
    /// let options = KanOptions {
    ///     input_size: input_size,
    ///     layer_sizes: vec![4, output_size],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Classification,
    ///     class_map: None,
    /// };
    /// let mut model = Kan::new(&options);
    /// let input = vec![0.5, 0.4, 0.5, 0.5, 0.4];
    /// assert_eq!(input.len(), input_size);
    /// let output = model.forward(input)?;
    /// assert_eq!(output.len(), output_size);
    /// /* interpret the output as you like, for example as logits in a classifier, or as predicted value in a regressor */
    /// # Ok::<(), fekan::kan::kan_error::KanError>(())
    /// ```
    pub fn forward(&mut self, input: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, KanError> {
        debug!("Forwarding {} samples through model", input.len());
        let mut preacts = input;
        trace!("Preactivations: {:?}", preacts);
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            debug!("Forwarding through layer {}", idx);
            preacts = layer
                .forward(preacts)
                .map_err(|e| KanError::forward(e, idx))?;
        }
        Ok(preacts)
    }

    // /// as [Kan::forward], but uses a thread pool to multi-thread the forward pass
    // pub fn forward_concurrent(
    //     &mut self,
    //     input: Vec<f64>,
    //     thread_pool: &ThreadPool,
    // ) -> Result<Vec<f64>, KanError> {
    //     let mut preacts = input;
    //     for (idx, layer) in self.layers.iter_mut().enumerate() {
    //         let result = layer.forward_concurrent(&preacts, thread_pool);
    //         if let Err(e) = result {
    //             return Err(KanError {
    //                 source: ErrorOperation::Forward(e),
    //                 index: idx,
    //             });
    //         }
    //         let output = result.unwrap();
    //         preacts = output;
    //     }
    //     Ok(preacts)
    // }

    /// as [`Kan::forward`], but does not accumulate any internal state
    ///
    /// This method should be used when the model is not being trained, for example during inference or validation: when you won't be backpropogating, this method is faster uses less memory than [`Kan::forward`]
    ///
    /// # Errors
    /// returns a [KanError] if any layer returns an error.
    /// # Example
    /// see [`Kan::forward`] for an example
    pub fn infer(&self, input: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, KanError> {
        let mut preacts = input;
        for (idx, layer) in self.layers.iter().enumerate() {
            preacts = layer
                .infer(&preacts)
                .map_err(|e| KanError::forward(e, idx))?;
        }
        Ok(preacts)
    }

    /// Back-propogate the gradient through the model, internally accumulating the gradients of the model's parameters, to be applied later with [`Kan::update`]
    ///
    /// # Errors
    /// returns an error if any layer returns an error.
    /// See [`KanLayer::backward`] for more information
    ///
    /// # Example
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType, kan_error::KanError};
    ///
    /// # let options = KanOptions {
    /// #    input_size: 5,
    /// #    layer_sizes: vec![4, 1],
    /// #    degree: 3,
    /// #    coef_size: 6,
    /// #    model_type: ModelType::Regression,
    /// #    class_map: None,
    /// # };
    /// let mut model = Kan::new(&options);
    ///
    /// # fn calculate_gradient(output: Vec<f64>, label: f64) -> Vec<f64> {vec![1.0; output.len()]}
    /// # let learning_rate = 0.1;
    /// # let features = vec![0.5, 0.4, 0.5, 0.5, 0.4];
    /// # let label = 0;
    /// let output = model.forward(features)?;
    /// let gradient = calculate_gradient(output, label as f64);
    /// let _ = model.backward(gradient)?; // the input gradient can be disregarded here.
    ///
    /// /*
    /// * The model has stored the gradients for it's parameters internally.
    /// * We can add conduct as many forward/backward pass-pairs as we like to accumulate gradient,
    /// * until we're ready to update the paramaters.
    /// */
    ///
    /// model.update(learning_rate); // update the parameters of the model based on the accumulated gradients here
    /// model.zero_gradients(); // zero the gradients for the next batch of training data
    /// # Ok::<(), fekan::kan::kan_error::KanError>(())
    /// ```
    pub fn backward(
        &mut self,
        gradients: Vec<Vec<f64>>,
        l1_penalty: f64,
        entropy_penalty: f64,
    ) -> Result<Vec<Vec<f64>>, KanError> {
        debug!("Backwarding {} gradients through model", gradients.len());
        let mut gradients = gradients;
        for (idx, layer) in self.layers.iter_mut().enumerate().rev() {
            gradients = layer
                .backward(&gradients, l1_penalty, entropy_penalty)
                .map_err(|e| KanError::backward(e, idx))?;
        }
        Ok(gradients)
    }

    // /// as [Kan::backward], but uses a thread pool to multi-thread the backward pass
    // pub fn backward_concurrent(
    //     &mut self,
    //     error: Vec<f64>,
    //     thread_pool: &ThreadPool,
    // ) -> Result<Vec<f64>, KanError> {
    //     let mut error = error;
    //     for (idx, layer) in self.layers.iter_mut().enumerate().rev() {
    //         let backward_result = layer.backward_concurrent(&error, thread_pool);
    //         match backward_result {
    //             Ok(result) => {
    //                 error = result;
    //             }
    //             Err(e) => {
    //                 return Err(KanError {
    //                     source: ErrorOperation::Backward(e),
    //                     index: idx,
    //                 });
    //             }
    //         }
    //     }
    //     Ok(error)
    // }

    /// Update the model's parameters based on the gradients that have been accumulated with [`Kan::backward`].
    /// # Example
    /// see [`Kan::backward`]
    pub fn update(&mut self, learning_rate: f64) {
        for layer in self.layers.iter_mut() {
            layer.update(learning_rate);
        }
    }

    /// Zero the internal gradients of the model's parameters
    /// # Example
    /// see [`Kan::backward`]
    pub fn zero_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_gradients();
        }
    }

    /// get the total number of parameters in the model, inlcuding untrained parameters. See [`KanLayer::parameter_count`] for more information
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    /// get the total number of trainable parameters in the model. See [`KanLayer::trainable_parameter_count`] for more information
    pub fn trainable_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.trainable_parameter_count())
            .sum()
    }

    /// Update the ranges spanned by the B-spline knots in the model, using samples accumulated by recent [`Kan::forward`] calls.
    ///
    /// see [`KanLayer::update_knots_from_samples`] for more information
    /// # Errors
    /// returns a [KanError] if any layer returns an error.
    ///
    /// # Example
    /// see [`KanLayer::update_knots_from_samples`] for examples
    pub fn update_knots_from_samples(&mut self, knot_adaptivity: f64) -> Result<(), KanError> {
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            debug!("Updating knots for layer {}", idx);
            if let Err(e) = layer.update_knots_from_samples(knot_adaptivity) {
                return Err(KanError::update_knots(e, idx));
            }
        }
        return Ok(());
    }

    /// Clear the cached samples used by [`Kan::update_knots_from_samples`]
    ///
    /// see [`KanLayer::clear_samples`] for more information
    pub fn clear_samples(&mut self) {
        debug!("Clearing samples from model");
        for layer_idx in 0..self.layers.len() {
            debug!("Clearing samples from layer {}", layer_idx);
            self.layers[layer_idx].clear_samples();
        }
    }

    /// Set the size of the knot vector used in all splines in this model
    /// see [`KanLayer::set_knot_length`] for more information
    pub fn set_knot_length(&mut self, knot_length: usize) -> Result<(), KanError> {
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            if let Err(e) = layer.set_knot_length(knot_length) {
                return Err(KanError::set_knot_length(e, idx));
            }
        }
        Ok(())
    }

    /// Get the size of the knot vector used in all splines in this model
    ///
    /// ## Note
    /// if different layers have different knot lengths, this method will return the knot length of the first layer
    pub fn knot_length(&self) -> usize {
        self.layers[0].knot_length()
    }

    /// Create a new model by merging multiple models together. Models must be of the same type and have the same number of layers, and all layers must be mergable (see [`KanLayer::merge_layers`])
    /// # Errors
    /// Returns a [`KanError`] if:
    /// * the models are not mergable. See [`Kan::models_mergable`] for more information
    /// * any layer encounters an error during the merge. See [`KanLayer::merge_layers`] for more information
    /// # Example
    /// Train multiple copies of the model on different data in different threads, then merge the trained models together
    /// ```
    /// use fekan::{kan::{Kan, KanOptions, ModelType, kan_error::KanError}, Sample};
    /// use std::thread;
    /// # let model_options = KanOptions {
    /// #    input_size: 5,
    /// #    layer_sizes: vec![4, 3],
    /// #    degree: 3,
    /// #    coef_size: 6,
    /// #    model_type: ModelType::Regression,
    /// #    class_map: None,
    /// };
    /// # let num_training_threads = 1;
    /// # let training_data = vec![Sample::new(vec![], 0.0)];
    /// # fn my_train_model_function(model: Kan, data: &[Sample]) -> Kan {model}
    /// let mut my_model = Kan::new(&model_options);
    /// let partially_trained_models: Vec<Kan> = thread::scope(|s|{
    ///     let chunk_size = f32::ceil(training_data.len() as f32 / num_training_threads as f32) as usize; // round up, since .chunks() gives up-to chunk_size chunks. This way to don't leave any data on the cutting room floor
    ///     let handles: Vec<_> = training_data.chunks(chunk_size).map(|training_data_chunk|{
    ///         let clone_model = my_model.clone();
    ///         s.spawn(move ||{
    ///             my_train_model_function(clone_model, training_data_chunk) // `my_train_model_function` is a stand-in for whatever function you're using to train the model - not actually defined in this crate
    ///         })
    ///     }).collect();
    ///     handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    /// });
    /// let fully_trained_model = Kan::merge_models(partially_trained_models)?;
    /// # Ok::<(), fekan::kan::kan_error::KanError>(())
    /// ```
    ///
    pub fn merge_models(models: Vec<Kan>) -> Result<Kan, KanError> {
        Self::models_mergable(&models)?; // check if the models are mergable
        let layer_count = models[0].layers.len();
        let model_type = models[0].model_type;
        let class_map = models[0].class_map.clone();
        let mut all_layers: Vec<VecDeque<KanLayer>> = models
            .into_iter()
            .map(|model| model.layers.into())
            .collect();
        // aggregate layers by index
        let mut merged_layers = Vec::new();
        for layer_idx in 0..layer_count {
            let layers_to_merge: Vec<KanLayer> = all_layers
                .iter_mut()
                .map(|layers| {
                    layers
                        .pop_front()
                        .expect("iterated past end of dequeue while merging models")
                })
                .collect();
            let merged_layer = KanLayer::merge_layers(layers_to_merge)
                .map_err(|e| KanError::merge_unmergable_layers(e, layer_idx))?;
            merged_layers.push(merged_layer);
        }

        let merged_model = Kan {
            layers: merged_layers,
            model_type,
            class_map,
        };
        Ok(merged_model)
    }

    /// Check if the given models can be merged using [`Kan::merge_models`]. Returns Ok(()) if the models are mergable, an error otherwise
    /// # Errors
    /// Returns a [`KanError`] if any of the models:
    /// * have different model types (e.g. classification vs regression)
    /// * have different numbers of layers
    /// * have different class maps (if the models are classification models)
    /// or if the input slice is empty
    pub fn models_mergable(models: &[Kan]) -> Result<(), KanError> {
        // this will be handled by the merge_layers method as a NoLayers error
        // if models.is_empty() {
        //     return Err(KanError {
        //         source: KanLayerError {
        //             error_kind: KanLayerErrorType::MergeNoLayers,
        //             source: None,
        //             spline_idx: None,
        //         },
        //         index: 0,
        //     });
        // }
        let expected_model_type = models[0].model_type;
        let expected_class_map = models[0].class_map.clone();
        let expected_layer_count = models[0].layers.len();
        for idx in 1..models.len() {
            if models[idx].model_type != expected_model_type {
                return Err(KanError::merge_mismatched_model_type(
                    idx,
                    expected_model_type,
                    models[idx].model_type,
                ));
            }
            if models[idx].class_map != expected_class_map {
                return Err(KanError::merge_mismatched_class_map(
                    idx,
                    expected_class_map.clone(),
                    models[idx].class_map.clone(),
                ));
            }
            if models[idx].layers.len() != expected_layer_count {
                return Err(KanError::merge_mismatched_depth_model(
                    idx,
                    expected_layer_count,
                    models[idx].layers.len(),
                ));
            }
        }
        Ok(())
    }

    /// Test and set the symbolic status of the model, using the given R^2 threshold. See [`KanLayer::test_and_set_symbolic`] for more information
    pub fn test_and_set_symbolic(&mut self, r2_threshold: f64) -> Vec<(usize, usize)> {
        debug!("Testing and setting symbolic for the model");
        let mut symbolified_edges = Vec::new();
        for i in 0..self.layers.len() {
            debug!("Symbolifying layer {}", i);
            let clamped_edges = self.layers[i].test_and_set_symbolic(r2_threshold);
            symbolified_edges.extend(clamped_edges.into_iter().map(|j| (i, j)));
        }
        symbolified_edges
    }

    /// Prune the model, removing any edges that have low average output. See [`KanLayer::prune`] for more information
    /// Returns a list of the indices of pruned edges (i,j), where i is the index of the layer, and j is the index of the edge in that layer that was pruned
    pub fn prune(&mut self, threshold: f64) -> Vec<(usize, usize)> {
        let mut pruned_edges = Vec::new();
        for i in 0..self.layers.len() {
            debug!("Pruning layer {}", i);
            let layer_prunings = self.layers[i].prune(threshold);
            pruned_edges.extend(layer_prunings.into_iter().map(|j| (i, j)));
        }
        pruned_edges
    }
}

impl PartialEq for Kan {
    fn eq(&self, other: &Self) -> bool {
        self.layers == other.layers && self.model_type == other.model_type
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_forward() {
        let kan_config = KanOptions {
            input_size: 3,
            layer_sizes: vec![4, 2, 3],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
            class_map: None,
        };
        let mut first_kan = Kan::new(&kan_config);
        let second_kan_config = KanOptions {
            layer_sizes: vec![2, 4, 3],
            ..kan_config
        };
        let mut second_kan = Kan::new(&second_kan_config);
        let input = vec![vec![0.5, 0.4, 0.5]];
        let result = first_kan.forward(input.clone()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
        let result = second_kan.forward(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_forward_then_backward() {
        let options = &KanOptions {
            input_size: 5,
            layer_sizes: vec![4, 2, 3],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
            class_map: None,
        };
        let mut first_kan = Kan::new(options);
        let input = vec![vec![0.5, 0.4, 0.5, 0.5, 0.4]];
        let result = first_kan.forward(input.clone()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), options.layer_sizes.last().unwrap().clone());
        let error = vec![vec![0.5, 0.4, 0.5]];
        let result = first_kan.backward(error, 1.0, 1.0).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), options.input_size);
    }

    #[test]
    fn test_merge_identical_models_yields_identical_output() {
        let kan_config = KanOptions {
            input_size: 3,
            layer_sizes: vec![4, 2, 3],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
            class_map: None,
        };
        let first_kan = Kan::new(&kan_config);
        let second_kan = first_kan.clone();
        let input = vec![vec![0.5, 0.4, 0.5]];
        let first_result = first_kan.infer(input.clone()).unwrap();
        let second_result = second_kan.infer(input.clone()).unwrap();
        assert_eq!(first_result, second_result);
        let merged_kan = Kan::merge_models(vec![first_kan, second_kan]).unwrap();
        let merged_result = merged_kan.infer(input).unwrap();
        assert_eq!(first_result, merged_result);
    }

    #[test]
    fn test_model_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Kan>();
    }

    #[test]
    fn test_model_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Kan>();
    }

    #[test]
    fn test_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<KanError>();
    }

    #[test]
    fn test_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KanError>();
    }
}
