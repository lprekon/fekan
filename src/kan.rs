use kan_layer::LayerError;
use serde::{Deserialize, Serialize};

pub mod kan_layer;

#[derive(Debug, Serialize, Deserialize)]
pub struct Kan {
    pub layers: Vec<kan_layer::KanLayer>,
    pub model_type: ModelType, // determined how the output is interpreted, and what the loss function ought to be
}

pub struct KanOptions {
    pub input_size: usize,
    pub layer_sizes: Vec<usize>,
    pub degree: usize,
    pub coef_size: usize,
    pub model_type: ModelType,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ModelType {
    Classification,
    Regression,
}

impl Kan {
    pub fn new(options: &KanOptions) -> Self {
        let mut layers = Vec::with_capacity(options.layer_sizes.len());
        let mut prev_size = options.input_size;
        for &size in options.layer_sizes.iter() {
            layers.push(kan_layer::KanLayer::new(
                prev_size,
                size,
                options.degree,
                options.coef_size,
            ));
            prev_size = size;
        }
        Kan {
            layers,
            model_type: options.model_type,
        }
    }

    /// passes the input to the [kan_layer::KanLayer::forward] method of the first layer,
    /// then calls the `forward` method of each subsequent layer with the output of the previous layer,
    /// returning the output of the final layer
    ///
    /// # Errors
    /// returns an error if any layer returns an error.
    /// See [kan_layer::KanLayer::forward] and [kan_layer::LayerError] for more information
    pub fn forward(&mut self, input: Vec<f32>) -> Result<Vec<f32>, KanError> {
        let mut preacts = input.clone();
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

    /// passes the error to the [kan_layer::KanLayer::backward] method of the last layer,
    /// then calls the `backward` method of each subsequent layer with the output of the previous layer,
    /// returning the error returned by first layer
    ///
    /// # Errors
    /// returns an error if any layer returns an error.
    /// See [kan_layer::KanLayer::backward] and [kan_layer::LayerError] for more information
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

    /// updates the weights of each layer based on the stored gradients that have been calculated
    pub fn update(&mut self, learning_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(learning_rate);
        }
    }

    /// sets the gradients of each layer to zero
    pub fn zero_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_gradients();
        }
    }

    /// returns the total number of parameters in the model, inlcuding untrained parameters
    pub fn get_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.get_parameter_count())
            .sum()
    }

    /// returns the total number of trainable parameters in the model
    pub fn get_trainable_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.get_trainable_parameter_count())
            .sum()
    }

    /// Update the knots in each layer based on the samples that have been passed through the model
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
    pub fn clear_samples(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_samples();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KanError {
    source: LayerError,
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
}
