pub mod kan_layer;

#[derive(Debug)]
pub struct Kan {
    pub layers: Vec<kan_layer::KanLayer>,
}

impl Kan {
    pub fn new(input_size: usize, layer_sizes: Vec<usize>, k: usize, coef_size: usize) -> Self {
        let mut layers = Vec::with_capacity(layer_sizes.len());
        let mut prev_size = input_size;
        for &size in layer_sizes.iter() {
            layers.push(kan_layer::KanLayer::new(prev_size, size, k, coef_size));
            prev_size = size;
        }
        Kan { layers }
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_forward() {
        let input_dimension = 3;
        let k = 3;
        let coef_size = 4;
        let layer_sizes = vec![4, 2, 3];
        let mut first_kan = Kan::new(input_dimension, layer_sizes, k, coef_size);
        let layer_sizes = vec![2, 4, 3];
        let mut second_kan = Kan::new(input_dimension, layer_sizes, k, coef_size);
        let input = vec![0.5, 0.4, 0.5];
        let result = first_kan.forward(input.clone()).unwrap();
        assert_eq!(result.len(), 3);
        let result = second_kan.forward(input).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_forward_then_backward() {
        let input_dimension = 5;
        let k = 3;
        let coef_size = 4;
        let layer_sizes = vec![4, 2, 3];
        let mut first_kan = Kan::new(input_dimension, layer_sizes.clone(), k, coef_size);
        let input = vec![0.5, 0.4, 0.5, 0.5, 0.4];
        let result = first_kan.forward(input.clone()).unwrap();
        assert_eq!(result.len(), layer_sizes.last().unwrap().clone());
        let error = vec![0.5, 0.4, 0.5];
        let result = first_kan.backward(error).unwrap();
        assert_eq!(result.len(), input_dimension);
    }
}
