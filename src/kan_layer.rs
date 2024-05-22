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
    nodes: Vec<Node>,
    /// the polynomial order of the splines
    k: usize, // this could be defined on a per-edge basis, but I'm going to define it on a per-layer basis for now
}

impl KanLayer {
    /// create a new layer with the given number of nodes in the previous layer and the given number of nodes in this layer
    fn new(input_dimension: usize, output_dimension: usize, k: usize) -> Self {
        todo!()
    }
}

struct IncomingEdge {
    knots: KnotVector,
    control_points: ControlPoints,
}

/// the control points used in calculating the b-splines.
struct ControlPoints(Vec<f32>);

/// a list of sets of control points, one for each incoming edge
struct Node(Vec<IncomingEdge>);

/// a knot vector which defines a b-spline
struct KnotVector(Vec<f32>);
