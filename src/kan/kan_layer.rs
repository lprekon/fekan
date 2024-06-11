// #![allow(dead_code)]

mod node;
mod spline;

use node::Node;
use serde::{Deserialize, Serialize};

use std::vec;

/// A layer in a Kolmogorov-Arnold neural network
///
/// because the interesting work in a KAN is done on the edges between nodes, the layer needs to know how many nodes are in the previous layer as well in itself,
/// so it can calculate the total number of incoming edges and initialize parameters for each edge
///
/// The layer keeps a vector of nodes, each of which has a vector of incoming edges. Each incoming edge has a knot vector and a set of control points.
/// The control points are the parameters that the network learns.
/// The knot vector is a set of values that define the b-spline.
///
/// the size of the node vector is equal to the output dimension of the layer
/// the size of the incoming edge vector for each node is equal to the input dimension of the layer

#[derive(Debug, Serialize, Deserialize)]
pub struct KanLayer {
    // I think it will make sense to have each KanLayer be a vector of splines, plus the input and output dimension.
    // the first `out_dim` splines will read from the first input, the second `out_dim` splines will read from the second input, etc., with `in_dim` such chunks
    // to caluclate the output of the layer, the first element is the sum of the output of splines 0, out_dim, 2*out_dim, etc., the second element is the sum of splines 1, out_dim+1, 2*out_dim+1, etc.
    /// the nodes in this layer. This nested vector contains the b-spline coefficients (control points) for the layer. This is the main parameter that the network learns.
    pub(crate) nodes: Vec<Node>,
    /// a vector of previous inputs to the layer, used to update the knot vectors for each incoming edge.
    samples: Vec<Vec<f32>>,
}

impl KanLayer {
    /// create a new layer with the given number of nodes in the previous layer and the given number of nodes in this layer
    /// # Examples
    /// ```
    /// use fekan::kan::kan_layer::KanLayer;
    ///
    /// let input_dimension = 3;
    /// let output_dimension = 4;
    /// let k = 5;
    /// let coef_size = 6;
    /// let my_layer = KanLayer::new(input_dimension, output_dimension, k, coef_size);
    /// assert_eq!(my_layer.len(), output_dimension);
    /// assert_eq!(my_layer.total_edges(), output_dimension * input_dimension);
    /// ```
    pub fn new(
        input_dimension: usize,
        output_dimension: usize,
        k: usize,
        coef_size: usize,
    ) -> Self {
        // let nodes: Vec<Node> = vec![Node::new(input_dimension, k, coef_size); output_dimension]; this is cloning the exact same node, which is not what we want
        let mut nodes = Vec::with_capacity(output_dimension);
        for _ in 0..output_dimension {
            nodes.push(Node::new(input_dimension, k, coef_size));
        }
        KanLayer {
            nodes,
            samples: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    // pub fn total_edges(&self) -> usize {
    //     self.nodes.len() * self.nodes[0].0.len()
    // }

    /// calculate the activations of the nodes in this layer given the preactivations. This operation mutates internal state, which will be read in [`KanLayer::backward()`].
    ///
    /// `preactivation.len()` must be equal to `input_dimension` provided when the layer was created
    pub fn forward(&mut self, preactivation: Vec<f32>) -> Result<Vec<f32>, String> {
        //  check the length here since it's the same check for the entire layer, even though the node is technically the part that cares
        if preactivation.len() != self.nodes[0].num_incoming_edges() {
            return Err(format!(
                "preactivation vector has length {}, but expected length {}",
                preactivation.len(),
                self.nodes[0].num_incoming_edges()
            ));
        }
        self.samples.push(preactivation.clone()); // save a copy of the preactivation for updating the knot vectors later
        let activations: Vec<f32> = self
            .nodes
            .iter_mut()
            .map(|node| node.activate(&preactivation))
            .collect();

        Ok(activations)
    }

    /// update the knot vectors for each incoming edge in this layer using the memoized samples
    ///
    pub fn update_knots_from_samples(&mut self) -> Result<(), String> {
        // this should never fire, but I'm leaving it here for now just to be safe.
        if !self
            .samples
            .iter()
            .all(|sample| sample.len() == self.nodes[0].num_incoming_edges())
        {
            return Err(format!("samples must have the same length as the input dimension of the layer! Expected {}, got {}", self.nodes[0].num_incoming_edges(), self.samples.len()));
        }
        for i in 0..self.nodes.len() {
            self.nodes[i].update_knots_from_samples(&self.samples);
        }

        // clear the samples after updating the knots
        self.samples.clear();

        Ok(())
    }

    /// given `error`, containing an error value for each node, calculate the gradients for the control points on each incoming edge,
    /// and return the error for the previous layer.
    ///
    /// This function relies on mutated inner state and should be called after [`KanLayer::forward`].
    ///
    /// This function mutates inner state, which will be used in [`KanLayer::update`]`
    ///
    /// # Errors
    /// Returns an error if the length of `error` is not equal to the number of nodes in this layer, or if `backward` is called before `forward`
    pub fn backward(&mut self, error: Vec<f32>) -> Result<Vec<f32>, String> {
        if error.len() != self.nodes.len() {
            return Err(format!(
                "error vector has length {}, but expected length {}",
                error.len(),
                self.nodes.len()
            ));
        }

        // first, calculate the gradients for the control points on each incoming edge
        let mut input_error = vec![0.0; self.nodes[0].num_incoming_edges()];
        for i in 0..self.nodes.len() {
            let node_input_error = self.nodes[i].backward(&error[i])?;
            for j in 0..input_error.len() {
                input_error[j] += node_input_error[j];
            }
        }
        Ok(input_error)
    }

    /// update the control points for each incoming edge in this layer given the learning rate
    ///
    /// this function relies on mutated inner state and should be called after [`KanLayer::backward()`]
    pub fn update(&mut self, learning_rate: f32) {
        for i in 0..self.nodes.len() {
            self.nodes[i].update(learning_rate);
        }
    }

    /// zero out the gradients for each incoming edge in this layer
    pub fn zero_gradients(&mut self) {
        for i in 0..self.nodes.len() {
            self.nodes[i].zero_gradients();
        }
    }

    pub fn get_parameter_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.get_parameter_count())
            .sum()
    }

    pub fn total_edges(&self) -> usize {
        self.nodes.iter().map(|node| node.total_edges()).sum()
    }
}

#[cfg(test)]
mod test {
    use spline::Spline;

    use super::*;

    /// returns a new layer with input and output dimension = 2, k = 3, and coef_size = 4
    fn build_test_layer() -> KanLayer {
        // this doesn't work because I need to specify the exact coefficients for each node
        let k = 3;
        let coef_size = 4;
        let knot_size = coef_size + k + 1;
        let mut knots = vec![0.0; knot_size];
        knots[0] = -1.0;
        for i in 1..knots.len() {
            knots[i] = -1.0 + (i as f32 / (knot_size - 1) as f32 * 2.0);
        }
        let spline1 = Spline::new(k, vec![1.0; coef_size], knots.clone()).unwrap();
        let spline2 = Spline::new(k, vec![-1.0; coef_size], knots.clone()).unwrap();
        let node1 = Node::new_from_splines(vec![spline1.clone(), spline2.clone()]);
        let node2 = Node::new_from_splines(vec![spline2, spline1]);
        KanLayer {
            nodes: vec![node1, node2],
            samples: vec![],
        }
    }

    #[test]
    fn test_new() {
        let input_dimension = 3;
        let output_dimension = 4;
        let k = 5;
        let coef_size = 6;
        let my_layer = KanLayer::new(input_dimension, output_dimension, k, coef_size);
        assert_eq!(my_layer.len(), output_dimension);
        assert_eq!(my_layer.nodes[0].num_incoming_edges(), input_dimension);
        // assert_eq!(my_layer.total_edges(), output_dimension * input_dimension);
    }

    #[test]
    fn test_forward() {
        // to properly test layer forward, I need a layer with output and input dim = 2, which means 4 total edges
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let acts = layer.forward(preacts).unwrap();
        let expected_activations = vec![0.3177, -0.3177];
        let rounded_activations: Vec<f32> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations);
    }

    #[test]
    fn test_forward_bad_activations() {
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5, 0.5];
        let acts = layer.forward(preacts);
        assert_eq!(
            acts,
            Err("preactivation vector has length 3, but expected length 2".to_string())
        );
    }

    #[test]
    fn test_forward_then_backward() {
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let acts = layer.forward(preacts).unwrap();
        let expected_activations = vec![0.3177, -0.3177];
        let rounded_activations: Vec<f32> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations);

        let error = vec![1.0, 0.5];
        let input_error = layer.backward(error).unwrap();
        let expected_input_error = vec![0.0, 0.60156];
        let rounded_input_error: Vec<f32> = input_error
            .iter()
            .map(|f| (f * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(rounded_input_error, expected_input_error);
    }

    #[test]
    fn test_backward_before_forward() {
        let mut layer = build_test_layer();
        let error = vec![1.0, 0.5];
        let input_error = layer.backward(error);
        assert!(input_error.is_err());
    }

    #[test]
    fn test_backward_bad_error_length() {
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let _ = layer.forward(preacts).unwrap();
        let error = vec![1.0, 0.5, 0.5];
        let input_error = layer.backward(error);
        assert!(input_error.is_err());
    }

    // it doesn't make sense to have this test anymore
    // #[test]
    // fn test_update_samples_bad_sample_length() {
    //     let mut layer = build_test_layer();
    //     let samples = vec![vec![0.0, 0.5, 0.5], vec![0.0, 0.5, 0.5]];
    //     let update = layer.update_knots_from_samples(&samples);
    //     assert!(update.is_err());
    // }
}
