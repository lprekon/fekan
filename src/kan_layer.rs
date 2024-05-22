#![allow(dead_code)]

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
        let nodes: Vec<Node> = vec![Node::new(input_dimension, k, coef_size); output_dimension];
        KanLayer { nodes }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// a list of sets of control points, one for each incoming edge
#[derive(Clone)]
pub(crate) struct Node(Vec<IncomingEdge>);
impl Node {
    /// create a new node with the given number of incoming edges
    fn new(input_dimension: usize, k: usize, coef_size: usize) -> Self {
        let incoming_edges: Vec<IncomingEdge> =
            vec![IncomingEdge::new(k, coef_size); input_dimension];
        Node(incoming_edges)
    }
}

#[derive(Clone)]
struct IncomingEdge {
    knots: KnotVector,
    control_points: ControlPoints,
}

impl IncomingEdge {
    fn new(k: usize, coef_size: usize) -> Self {
        let knots: KnotVector = KnotVector(vec![0.0; coef_size + k + 1]); // TODO: initialize the knot vector properly, with a random distribution
        let control_points: ControlPoints = ControlPoints(vec![0.0; coef_size]); // TODO: initialize the control points properly, with a random distribution
        IncomingEdge {
            knots,
            control_points,
        }
    }
}

/// the control points used in calculating the b-splines.
#[derive(Clone)]
struct ControlPoints(Vec<f32>);

/// a knot vector which defines a b-spline
#[derive(Clone)]
struct KnotVector(Vec<f32>);

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
        assert_eq!(my_edge.knots.0.len(), coef_size + k + 1);
        assert_eq!(my_edge.control_points.0.len(), coef_size);
    }
}
