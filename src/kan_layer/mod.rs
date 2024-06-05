// #![allow(dead_code)]

mod node;
mod spline;

use node::Node;

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

#[derive(Debug)]
pub struct KanLayer {
    /// the nodes in this layer. This nested vector contains the b-spline coefficients (control points) for the layer. This is the main parameter that the network learns.
    pub(crate) nodes: Vec<Node>,
}

impl KanLayer {
    /// create a new layer with the given number of nodes in the previous layer and the given number of nodes in this layer
    /// # Examples
    /// ```
    /// use fekan::kan_layer::KanLayer;
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
        KanLayer { nodes }
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
        let activations: Vec<f32> = self
            .nodes
            .iter_mut()
            .map(|node| node.activate(&preactivation))
            .collect();

        Ok(activations)
    }

    /// update the knot vectors for each incoming edge in this layer given the samples
    ///
    /// samples is a vector of vectors, where each inner vector is a sample of the input to the layer, and the outer vector contains all the samples
    ///
    /// # Errors
    /// returns and error if the length of the inner vectors in `samples` is not equal to the input dimension of the layer
    pub fn update_knots_from_samples(&mut self, samples: &Vec<Vec<f32>>) -> Result<(), String> {
        if !samples
            .iter()
            .all(|sample| sample.len() == self.nodes[0].num_incoming_edges())
        {
            return Err(format!("samples must have the same length as the input dimension of the layer! Expected {}, got {}", self.nodes[0].num_incoming_edges(), samples.len()));
        }
        for i in 0..self.nodes.len() {
            self.nodes[i].update_knots_from_samples(samples);
        }

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
}
