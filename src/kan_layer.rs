#![allow(dead_code)]

use bspline::BSpline;
use rand::distributions::Distribution; // apparently the statrs distributions use the rand Distribution trait
use rand::prelude::*;
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
/// ```
impl KanLayer {
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

    /// calculate the activations of the nodes in this layer given the preactivations
    ///
    /// `preactivation.len()` must be equal to `input_dimension` provided when the layer was created
    pub fn forward(&self, preactivation: Vec<f32>) -> Result<Vec<f32>, String> {
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
            .iter()
            .map(|node| node.activate(&preactivation))
            .collect();

        Ok(activations)
    }
}

/// a list of sets of control points, one for each incoming edge
#[derive(Clone, Debug)]
pub(crate) struct Node(Vec<IncomingEdge>);
impl Node {
    /// create a new node with the given number of incoming edges
    fn new(input_dimension: usize, k: usize, coef_size: usize) -> Self {
        // let incoming_edges: Vec<IncomingEdge> = vec![IncomingEdge::new(k, coef_size); input_dimension]; this is cloning the exact same edge, which is not what we want
        let mut incoming_edges = Vec::with_capacity(input_dimension);
        for _ in 0..input_dimension {
            incoming_edges.push(IncomingEdge::new(k, coef_size));
        }
        Node(incoming_edges)
    }

    /// calculate the activation of the node given the preactivation
    //the preactivation length is checked in the layer, so we don't need to check it here
    fn activate(&self, preactivation: &Vec<f32>) -> f32 {
        self.0
            .iter()
            .zip(preactivation)
            .map(|(edge, &preact)| edge.0.point(preact))
            .sum()
    }
}

#[derive(Clone, Debug)]
struct IncomingEdge {
    spline: BSpline<f32, f32>,
    gradients: Vec<f32>,
}

impl IncomingEdge {
    fn new(k: usize, coef_size: usize) -> Self {
        assert!(
            coef_size >= (k + 1),
            "too few control points for the degree of the b-spline"
        );

        let mut knots = vec![0.0; coef_size + k + 1];
        let inner_knot_count = knots.len() - 2 * (k + 1);
        // the first k+1 knots should be zero
        // for i in 0..k + 1 {
        //     knots[i] = 0.0;
        // }
        // the last k+1 knots should be one
        for j in knots.len() - k - 1..knots.len() {
            knots[j] = 1.0;
        }
        // the inner knots should be evenly spaced between 0 and 1
        for i in 1..inner_knot_count + 1 {
            knots[k + i] = i as f32 / (inner_knot_count + 1) as f32;
        }

        let mut control_points = vec![0.0; coef_size];
        let norm_dist = Normal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        for i in 0..coef_size {
            control_points[i] = norm_dist.sample(&mut rng) as f32;
        }
        IncomingEdge {
            spline: BSpline::new(k, control_points, knots),
            gradients: vec![0.0; coef_size],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // new layer tested with doc test

    #[test]
    fn test_new_node() {
        let input_dimension = 2;
        let k = 3;
        let coef_size = 4;
        let my_node = Node::new(input_dimension, k, coef_size);
        assert_eq!(my_node.0.len(), input_dimension);
    }

    #[test]
    fn test_new_incoming_edge() {
        let k = 3;
        let coef_size = 4;
        let my_edge = IncomingEdge::new(k, coef_size);
        assert_eq!(my_edge.spline.knots().len(), coef_size + k + 1);
        assert_eq!(my_edge.spline.control_points().len(), coef_size);
    }

    #[test]
    fn test_edge_knot_initialization_with_one_inner_knot() {
        let k = 3;
        let coef_size = 5;
        let my_edge = IncomingEdge::new(k, coef_size);
        let knots: Vec<f32> = my_edge.spline.knots().cloned().collect();
        let expected = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(knots.len(), expected.len(), "knot vector incorrect length");
        assert_eq!(knots, expected, "knot vector incorrect values");
        assert_eq!(my_edge.spline.knot_domain(), (0.0, 1.0), "bad knot domain");
    }

    #[test]
    fn test_edge_knot_initialization_with_no_inner_knots() {
        let k = 3;
        let coef_size = 4;
        let my_edge = IncomingEdge::new(k, coef_size);
        let knots: Vec<f32> = my_edge.spline.knots().cloned().collect();
        let expected = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(knots.len(), expected.len(), "knot vector incorrect length");
        assert_eq!(knots, expected, "knot vector incorrect values");
        assert_eq!(my_edge.spline.knot_domain(), (0.0, 1.0), "bad knot domain");
    }

    #[test]
    fn test_edge_knot_initialization_with_3_inner_knots() {
        let k = 2;
        let coef_size = 6;
        let my_edge = IncomingEdge::new(k, coef_size);
        let knots: Vec<f32> = my_edge.spline.knots().cloned().collect();
        let expected = vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0];
        assert_eq!(knots.len(), expected.len(), "knot vector incorrect length");
        assert_eq!(knots, expected, "knot vector incorrect values");
        assert_eq!(my_edge.spline.knot_domain(), (0.0, 1.0), "bad knot domain");
    }
}
