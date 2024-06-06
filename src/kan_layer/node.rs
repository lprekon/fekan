use rand::distributions::Distribution; // apparently the statrs distributions use the rand Distribution trait
use rand::prelude::*;
use statrs::distribution::Normal;

use super::spline::Spline;

// use super::spline::Spline;

/// a list of sets of control points, one for each incoming edge
#[derive(Debug)]
pub(crate) struct Node(Vec<Spline>);
impl Node {
    /// create a new node with `input_dimension` incoming edges, each with a spline of degree `k` and `coef_size` control points
    ///
    /// All knots for all incoming edges are initialized to a linearly spaced set of values from -1 to 1.
    /// All control points for all incoming edges are initialized to random values from a standard normal distribution with mean 0 and standard deviation 1
    pub(super) fn new(input_dimension: usize, k: usize, coef_size: usize) -> Self {
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

    #[allow(dead_code)]
    // used in KanLayer tests
    pub(super) fn new_from_splines(splines: Vec<Spline>) -> Self {
        Node(splines)
    }

    pub(super) fn num_incoming_edges(&self) -> usize {
        self.0.len()
    }

    /// calculate the activation of the node given the preactivation
    ///
    /// the values of `preactivation` are the output values of the nodes in the previous layer
    //the preactivation length is checked in the layer, so we don't need to check it here
    pub(super) fn activate(&mut self, preactivation: &Vec<f32>) -> f32 {
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
    pub(super) fn update_knots_from_samples(&mut self, samples: &Vec<Vec<f32>>) {
        for i in 0..self.0.len() {
            let samples_for_edge: Vec<f32> = samples.iter().map(|sample| sample[i]).collect();
            self.0[i].update_knots_from_samples(samples_for_edge);
        }
    }

    /// apply the backward propogation step to each incoming edge to this node
    ///
    /// returns the input error for each incoming edge
    ///
    /// # Errors
    /// returns an error is backward is called before forward
    ///
    pub(super) fn backward(&mut self, error: &f32) -> Result<Vec<f32>, String> {
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
    pub(super) fn update(&mut self, learning_rate: f32) {
        for i in 0..self.0.len() {
            self.0[i].update(learning_rate);
        }
    }

    /// zero out the gradients for each incoming edge to this node
    pub(super) fn zero_gradients(&mut self) {
        for i in 0..self.0.len() {
            self.0[i].zero_gradients();
        }
    }

    pub(super) fn get_parameter_count(&self) -> usize {
        self.0.iter().map(|edge| edge.get_parameter_count()).sum()
    }
}

#[cfg(test)]
mod test {
    use crate::kan_layer::spline::KNOT_MARGIN;

    use super::*;

    // new layer tested with doc test
    /// returns a node with two splines, each with 4 control points, and knots from -1 to 1
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

    #[test]
    fn test_update_from_samples() {
        let mut node = build_test_node();

        let mut sample_values = Vec::with_capacity(150);
        // assuming unordered samples for now
        for _ in 0..50 {
            sample_values.push(3.0)
        }
        for i in 0..100 {
            sample_values.push(-3.0 + i as f32 * 0.06);
        }
        // outer samples should be length 150, but each inner vector should be length 2 with the same value, since the node has 2 incoming edges, and the inner vectors are supposed to be the values coming down each edge.
        // at this level, the outer vector spans time, and the inner vectors span the incoming edges
        let mut samples = Vec::with_capacity(150);
        for i in 0..150 {
            samples.push(vec![sample_values[i], sample_values[i]]);
        }

        node.update_knots_from_samples(&samples);

        let mut expected_knots = vec![-3.0, -1.74, -0.48, 0.78, 2.04, 3.0, 3.0, 3.0];
        expected_knots[0] -= KNOT_MARGIN;
        expected_knots[7] += KNOT_MARGIN;
        assert_eq!(
            node.0[0]
                .knots()
                .cloned()
                .map(|v| (v * 10000.0).round() / 10000.0)
                .collect::<Vec<f32>>(),
            expected_knots,
            "spline 1 knots"
        );
        assert_eq!(
            node.0[1]
                .knots()
                .cloned()
                .map(|v| (v * 10000.0).round() / 10000.0)
                .collect::<Vec<f32>>(),
            expected_knots,
            "spline 2 knots"
        );
    }
}
