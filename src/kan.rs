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

    pub fn forward(&mut self, input: Vec<f32>) -> Result<Vec<f32>, String> {
        let mut preacts = input;
        for layer in self.layers.iter_mut() {
            preacts = layer.forward(preacts)?;
        }
        Ok(preacts)
    }

    pub fn backward(&mut self, error: Vec<f32>) -> Result<Vec<f32>, String> {
        let mut error = error;
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(error)?;
        }
        Ok(error)
    }
    pub fn update(&mut self, learning_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(learning_rate);
        }
    }

    pub fn zero_gradients(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_gradients();
        }
    }

    pub fn get_parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.get_parameter_count())
            .sum()
    }

    pub fn update_knots_from_samples(&mut self) -> Result<(), String> {
        for layer in self.layers.iter_mut() {
            if let Err(e) = layer.update_knots_from_samples() {
                return Err(e);
            }
        }
        return Ok(());
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
