// #![allow(dead_code)]

mod spline;

use rand::distributions::Distribution; // apparently the statrs distributions use the rand Distribution trait
use rand::prelude::*;
use spline::Spline;
use statrs::distribution::Normal;
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

    pub fn total_edges(&self) -> usize {
        self.nodes.len() * self.nodes[0].0.len()
    }

    /// calculate the activations of the nodes in this layer given the preactivations. This operation mutates internal state, which will be read in [`KanLayer::backward()`].
    ///
    /// `preactivation.len()` must be equal to `input_dimension` provided when the layer was created
    pub fn forward(&mut self, preactivation: Vec<f32>) -> Result<Vec<f32>, String> {
        //  check the length here since it's the same check for the entire layer, even though the node is technically the part that cares
        if preactivation.len() != self.nodes[0].0.len() {
            return Err(format!(
                "preactivation vector has length {}, but expected length {}",
                preactivation.len(),
                self.nodes[0].0.len()
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
            .all(|sample| sample.len() == self.nodes[0].0.len())
        {
            return Err(format!("samples must have the same length as the input dimension of the layer! Expected {}, got {}", self.nodes[0].0.len(), samples.len()));
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
    /// Returns an error if the length of `error` is not equal to the number of nodes in this layer.
    pub fn backward(&mut self, error: Vec<f32>) -> Result<Vec<f32>, String> {
        if error.len() != self.nodes.len() {
            return Err(format!(
                "error vector has length {}, but expected length {}",
                error.len(),
                self.nodes.len()
            ));
        }

        // first, calculate the gradients for the control points on each incoming edge
        for i in 0..self.nodes.len() {
            self.nodes[i].backward(&error[i]);
        }
        // next, calculate the error for the previous layer

        todo!("implement the backward pass")
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

/// a list of sets of control points, one for each incoming edge
#[derive(Debug)]
pub(crate) struct Node(Vec<Spline>);
impl Node {
    /// create a new node with `input_dimension` incoming edges, each with a spline of degree `k` and `coef_size` control points
    ///
    /// All knots for all incoming edges are initialized to a linearly spaced set of values from -1 to 1.
    /// All control points for all incoming edges are initialized to random values from a standard normal distribution with mean 0 and standard deviation 1
    fn new(input_dimension: usize, k: usize, coef_size: usize) -> Self {
        // let incoming_edges: Vec<IncomingEdge> = vec![IncomingEdge::new(k, coef_size); input_dimension]; this is cloning the exact same edge, which is not what we want
        let mut incoming_edges = Vec::with_capacity(input_dimension);
        let knot_size = coef_size + k + 1;
        for _ in 0..input_dimension {
            let mut knots = vec![0.0; knot_size];
            knots[0] = -1.0;
            for i in 1..knots.len() {
                knots[i] = -1.0 + (i as f32 / (knot_size - 1) as f32 * 2.0);
            }

            let mut control_points = vec![0.0; coef_size];
            let norm_dist = Normal::new(0.0, 1.0).unwrap();
            let mut rng = thread_rng();
            for i in 0..coef_size {
                control_points[i] = norm_dist.sample(&mut rng) as f32;
            }

            let incoming_edge = Spline::new(k, control_points, knots).unwrap();
            incoming_edges.push(incoming_edge);
        }
        Node(incoming_edges)
    }

    /// calculate the activation of the node given the preactivation
    ///
    /// the values of `preactivation` are the output values of the nodes in the previous layer
    //the preactivation length is checked in the layer, so we don't need to check it here
    fn activate(&mut self, preactivation: &Vec<f32>) -> f32 {
        self.0
            .iter_mut()
            .zip(preactivation)
            .map(|(edge, &preact)| edge.forward(preact))
            .sum()
    }

    // sample size is checked at the layer level, so we don't need to check it here
    // samples is a vector of vectors, where each inner vector is a sample of the input to the layer, and the outer vector contains all the samples
    // we have to take the nth element from each inner vector, repack it into a vector, and pass that to the edge to use for updating.
    // TODO move the repackaging into the layer, so it's done once for all nodes
    fn update_knots_from_samples(&mut self, samples: &Vec<Vec<f32>>) {
        for i in 0..self.0.len() {
            let samples_for_edge: Vec<f32> = samples.iter().map(|sample| sample[i]).collect();
            self.0[i].update_knots_from_samples(samples_for_edge);
        }
    }

    /// apply the backward propogation step to each incoming edge to this node
    ///
    /// returns the input error for each incoming edge
    fn backward(&mut self, error: &f32) -> Result<Vec<f32>, String> {
        let num_incoming_edges = self.0.len();
        let partial_error = error / num_incoming_edges as f32; // divide the error evenly among the incoming edges
        let mut input_error = Vec::with_capacity(num_incoming_edges);
        for i in 0..num_incoming_edges {
            let edge_input_error = self.0[i].backward(partial_error)?;
            input_error.push(edge_input_error);
        }
        Ok(input_error)
    }

    /// update the control points for each incoming edge to this node
    /// this function relies on mutated inner state and should be called after [`Node::backward()`]
    fn update(&mut self, learning_rate: f32) {
        for i in 0..self.0.len() {
            self.0[i].update(learning_rate);
        }
    }

    /// zero out the gradients for each incoming edge to this node
    fn zero_gradients(&mut self) {
        for i in 0..self.0.len() {
            self.0[i].zero_gradients();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // new layer tested with doc test

    fn build_test_node() -> Node {
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
        Node(vec![spline1, spline2])
    }

    #[test]
    fn test_new_node() {
        let input_dimension = 2;
        let k = 3;
        let coef_size = 5;
        let my_node = Node::new(input_dimension, k, coef_size);
        assert_eq!(my_node.0.len(), input_dimension);
        assert_eq!(my_node.0[0].knots().len(), coef_size + k + 1);
        let expected_knots: Vec<f32> = vec![-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(
            my_node.0[0].knots().cloned().collect::<Vec<f32>>(),
            expected_knots
        );
    }

    #[test]
    fn test_activate() {
        let mut node = build_test_node();

        let preactivation = vec![0.0, 0.5];
        let activation = node.activate(&preactivation);
        let expected_activation = 1.0 - 0.68229163; // the first spline should get 0 as an input and output 1, the second spline should get 0.5 as an input and output -0.68229163
        assert_eq!(activation, expected_activation);
    }

    #[test]
    fn test_activate_and_backward() {
        let mut node = build_test_node();
        // println!("to start: {:#?}\n", node);
        let preactivation = vec![0.0, 0.5];
        let _ = node.activate(&preactivation);
        // println!("post activation: {:#?}\n", node);
        let error = 1.0;
        let input_error = node.backward(&error).unwrap();
        // println!("post backprop: {:#?}\n", node);
        let expected_input_drts = vec![0.0, 2.40626];
        let expected_input_error = expected_input_drts
            .iter()
            .map(|f| f * error / node.0.len() as f32) // the error gets divided amongst the incoming edges
            .collect::<Vec<f32>>();
        let rounded_input_error: Vec<f32> = input_error
            .iter()
            .map(|f| (f * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(rounded_input_error, expected_input_error);
    }
}
