use crate::kan_layer::{KanLayer, KanLayerOptions, LayerError};
use serde::{Deserialize, Serialize};

/// A full neural network model, consisting of multiple Kolmogorov-Arnold layers
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Kan {
    /// the layers of the model
    pub layers: Vec<KanLayer>,
    /// the type of model. This field is metadata and does not affect the operation of the model, though it is used elsewhere in the crate. See [`fekan::train_model()`](crate::train_model) for an example
    pub model_type: ModelType, // determined how the output is interpreted, and what the loss function ought to be
}

/// Hyperparameters for a Kan model
///
/// # Example
/// see [Kan::new]
///
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct KanOptions {
    /// the number of input features the model should except
    pub input_size: usize,
    /// the sizes of the layers to use in the model, including the output layer
    pub layer_sizes: Vec<usize>,
    /// the degree of the b-splines to use in each layer
    pub degree: usize,
    /// the number of coefficients to use in each layer
    pub coef_size: usize,
    /// the type of model to create. This field is metadata and does not affect the operation of the model, though it is used elsewhere in the crate. See [`fekan::train_model()`](crate::train_model) for an example
    pub model_type: ModelType,
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

impl Kan {
    /// creates a new Kan model with the given hyperparameters
    ///
    /// # Example
    /// Create a regression model with 5 input features, 2 hidden layers of size 4 and 3, and 1 output feature, using degree 3 b-splines with 6 coefficients
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType};
    ///
    /// let options = KanOptions {
    ///     input_size: 5,
    ///     layer_sizes: vec![4, 3],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Regression,
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
        }
    }

    /// passes the input to the [`crate::kan_layer::KanLayer::forward`] method of the first layer,
    /// then calls the `forward` method of each subsequent layer with the output of the previous layer,
    /// returning the output of the final layer
    ///
    /// # Errors
    /// returns a [KanError] if any layer returns an error.
    /// See [crate::kan_layer::KanLayer::forward] and [crate::kan_layer::LayerError] for more information
    ///
    /// # Example
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType};
    /// let input_size = 5;
    /// let output_size = 3;
    /// let options = KanOptions {
    ///     input_size: input_size,
    ///     layer_sizes: vec![4, output_size],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Classification,
    /// };
    /// let mut model = Kan::new(&options);
    /// let input = vec![0.5, 0.4, 0.5, 0.5, 0.4];
    /// assert_eq!(input.len(), input_size);
    /// let output = model.forward(input)?;
    /// assert_eq!(output.len(), output_size);
    /// /* interpret the output as you like, for example as logits in a classifier, or as predicted value in a regressor */
    /// # Ok::<(), fekan::kan::KanError>(())
    /// ```
    pub fn forward(&mut self, input: Vec<f32>) -> Result<Vec<f32>, KanError> {
        let mut preacts = input;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let result = layer.forward(&preacts);
            if let Err(e) = result {
                return Err(KanError {
                    source: e,
                    index: idx,
                });
            }
            let output = result.unwrap();
            preacts = output;
        }
        Ok(preacts)
    }

    /// passes the error to the [crate::kan_layer::KanLayer::backward] method of the last layer,
    /// then calls the `backward` method of each subsequent layer with the output of the previous layer,
    /// returning the error returned by first layer
    ///
    /// # Errors
    /// returns an error if any layer returns an error.
    /// See [crate::kan_layer::KanLayer::backward] and [crate::kan_layer::LayerError] for more information
    ///
    /// # Example
    /// ```
    /// use fekan::kan::{Kan, KanOptions, ModelType};
    ///
    /// let options = KanOptions {
    ///     input_size: 5,
    ///     layer_sizes: vec![4, 3],
    ///     degree: 3,
    ///     coef_size: 6,
    ///     model_type: ModelType::Regression,
    /// };
    /// let mut model = Kan::new(&options);
    ///
    /// let input = vec![0.5, 0.4, 0.5, 0.5, 0.4];
    /// let output = model.forward(input)?;
    /// /* interpret the output as you like, for example as logits */
    /// # Ok::<(), fekan::kan::KanError>(())
    /// ```
    pub fn backward(&mut self, error: Vec<f32>) -> Result<Vec<f32>, KanError> {
        let mut error = error;
        for (idx, layer) in self.layers.iter_mut().enumerate().rev() {
            let backward_result = layer.backward(&error);
            match backward_result {
                Ok(result) => {
                    error = result;
                }
                Err(e) => {
                    return Err(KanError {
                        source: e,
                        index: idx,
                    });
                }
            }
        }
        Ok(error)
    }

    /// calls each layer's [`crate::kan_layer::KanLayer::update`] method with the given learning rate
    pub fn update(&mut self, learning_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(learning_rate);
        }
    }

    /// calls each layer's [`crate::kan_layer::KanLayer::zero_gradients`] method
    pub fn zero_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_gradients();
        }
    }

    /// returns the total number of parameters in the model, inlcuding untrained parameters
    ///
    /// see [`crate::kan_layer::KanLayer::parameter_count`] for more information
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    /// returns the total number of trainable parameters in the model
    ///
    /// see [`crate::kan_layer::KanLayer::trainable_parameter_count`] for more information
    pub fn trainable_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.trainable_parameter_count())
            .sum()
    }

    /// Update the knots in each layer based on the samples that have been passed through the model
    ///
    /// # Errors
    /// returns a [KanError] if any layer returns an error. see [`crate::kan_layer::KanLayer::update_knots_from_samples`] for more information
    ///
    pub fn update_knots_from_samples(&mut self, knot_adaptivity: f32) -> Result<(), KanError> {
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            if let Err(e) = layer.update_knots_from_samples(knot_adaptivity) {
                return Err(KanError {
                    source: e,
                    index: idx,
                });
            }
        }
        return Ok(());
    }

    /// Clear the samples from each layer
    ///
    /// see [`crate::kan_layer::KanLayer::clear_samples`] for more information
    pub fn clear_samples(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_samples();
        }
    }
}

impl PartialEq for Kan {
    fn eq(&self, other: &Self) -> bool {
        self.layers == other.layers && self.model_type == other.model_type
    }
}

/// An error that occurs when a Kan model encounters an error in one of its layers
///
/// Displaying the error will show the index of the layer that encountered the error, and the error itself
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KanError {
    /// the error that occurred
    source: LayerError,
    /// the index of the layer that encountered the error
    index: usize,
}

impl std::fmt::Display for KanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "layer {} encountered error {}", self.index, self.source)
    }
}

impl std::error::Error for KanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
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
        };
        let mut first_kan = Kan::new(&kan_config);
        let second_kan_config = KanOptions {
            layer_sizes: vec![2, 4, 3],
            ..kan_config
        };
        let mut second_kan = Kan::new(&second_kan_config);
        let input = vec![0.5, 0.4, 0.5];
        let result = first_kan.forward(input.clone()).unwrap();
        assert_eq!(result.len(), 3);
        let result = second_kan.forward(input).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_forward_then_backward() {
        let options = &KanOptions {
            input_size: 5,
            layer_sizes: vec![4, 2, 3],
            degree: 3,
            coef_size: 4,
            model_type: ModelType::Classification,
        };
        let mut first_kan = Kan::new(options);
        let input = vec![0.5, 0.4, 0.5, 0.5, 0.4];
        let result = first_kan.forward(input.clone()).unwrap();
        assert_eq!(result.len(), options.layer_sizes.last().unwrap().clone());
        let error = vec![0.5, 0.4, 0.5];
        let result = first_kan.backward(error).unwrap();
        assert_eq!(result.len(), options.input_size);
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
