// #![allow(dead_code)]
mod spline;

use rand::distributions::Distribution;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use spline::{generate_uniform_knots, Spline};
use statrs::distribution::Normal; // apparently the statrs distributions use the rand Distribution trait

use std::{
    fmt::{self, Formatter},
    vec,
};

/// A layer in a Kolmogorov-Arnold neural network
///
/// because the interesting work in a KAN is done on the edges between nodes, the layer needs to know how many nodes are in the previous layer as well in itself,
/// so it can calculate the total number of incoming edges and initialize parameters for each edge
///
/// The layer keeps a vector of nodes, each of which has a vector of incoming edges. Each incoming edge has a knot vector and a set of control points.
/// The control points are the parameters that the network learns.
/// The knot vector is a set of values that define the b-spline.
///
/// the size of the "node" vector is equal to the output dimension of the layer
/// the size of the incoming edge vector for each "node" is equal to the input dimension of the layer

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KanLayer {
    // I think it will make sense to have each KanLayer be a vector of splines, plus the input and output dimension.
    // the first `out_dim` splines will read from the first input, the second `out_dim` splines will read from the second input, etc., with `in_dim` such chunks
    // to caluclate the output of the layer, the first element is the sum of the output of splines 0, out_dim, 2*out_dim, etc., the second element is the sum of splines 1, out_dim+1, 2*out_dim+1, etc.
    /// the splines in this layer. The first `input_dimension` splines belong to the first "node", the second `input_dimension` splines belong to the second "node", etc.
    pub(crate) splines: Vec<Spline>,
    input_dimension: usize,
    output_dimension: usize,
    /// a vector of previous inputs to the layer, used to update the knot vectors for each incoming edge.
    ///
    /// dim0 = number of samples
    ///
    /// dim1 = input_dimension
    #[serde(skip)] // part of the layer's operating state, not part of the model
    samples: Vec<Vec<f32>>,
}

#[derive(Debug, Copy, Clone)]
pub struct KanLayerOptions {
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub degree: usize,
    pub coef_size: usize,
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
    /// assert_eq!(my_layer.total_edges(), output_dimension * input_dimension);
    /// ```
    pub fn new(options: KanLayerOptions) -> Self {
        let num_edges = options.input_dimension * options.output_dimension;
        let num_knots = options.coef_size + options.degree + 1;
        let normal_dist = Normal::new(0.0, 1.0).expect("unable to create normal distribution");
        let mut randomness = thread_rng();
        let splines = (0..num_edges)
            .map(|_| {
                let coefficients: Vec<f32> = (0..options.coef_size)
                    .map(|_| normal_dist.sample(&mut randomness) as f32)
                    .collect();
                Spline::new(
                    options.degree,
                    coefficients,
                    generate_uniform_knots(-1.0, 1.0, num_knots),
                )
                .expect("spline creation error")
            })
            .collect();

        KanLayer {
            splines,
            input_dimension: options.input_dimension,
            output_dimension: options.output_dimension,
            samples: Vec::new(),
        }
    }

    // pub fn len(&self) -> usize {
    //     self.nodes.len()
    // }

    // pub fn total_edges(&self) -> usize {
    //     self.nodes.len() * self.nodes[0].0.len()
    // }

    /// calculate the activations of the nodes in this layer given the preactivations. This operation mutates internal state, which will be read in [`KanLayer::backward()`].
    ///
    /// `preactivation.len()` must be equal to `input_dimension` provided when the layer was created
    /// # Errors
    /// Returns an error if the length of `preactivation` is not equal to the input_dimension this layer
    pub fn forward(&mut self, preactivation: &Vec<f32>) -> Result<Vec<f32>, LayerError> {
        //  check the length here since it's the same check for the entire layer, even though the "node" is technically the part that cares
        if preactivation.len() != self.input_dimension {
            return Err(LayerError::MissizedPreactsError {
                actual: preactivation.len(),
                expected: self.input_dimension,
            });
        }
        self.samples.push(preactivation.clone()); // save a copy of the preactivation for updating the knot vectors later

        // it probably makes sense to move straight down the list of splines, since that theoretically should have better cache performance
        // also, I guess I haven't decided (in code) how the splines are ordered, so there's no reason I can't say the first n splines all belong to the first node, etc.
        // I just have to be consistent when I get to back propagation
        let mut activations: Vec<f32> = vec![0.0; self.output_dimension];
        for (idx, spline) in self.splines.iter_mut().enumerate() {
            let act = spline.forward(preactivation[idx % self.input_dimension]); // the first `input_dimension` splines belong to the first "node", the second `input_dimension` splines belong to the second node, etc.
            activations[(idx / self.input_dimension) as usize] += act; // every `input_dimension` splines, we move to the next node
        }

        if activations.iter().any(|x| x.is_nan()) {
            return Err(LayerError::NaNsError);
        }
        Ok(activations)
    }

    /// update the knot vectors for each incoming edge in this layer using the memoized samples
    ///
    /// # Errors
    /// Returns an error if the layer has no memoized samples, which most likely means that `forward` has not been called since initialization or the last call to `clear_samples`
    pub fn update_knots_from_samples(&mut self, knot_adaptivity: f32) -> Result<(), LayerError> {
        if self.samples.is_empty() {
            return Err(LayerError::NoSamplesError);
        }

        // lets construct a sorted vector of the samples for each incoming value
        // first we transpose the samples, so that dim0 = input_dimension, dim1 = number of samples
        let mut sorted_samples: Vec<Vec<f32>> =
            vec![Vec::with_capacity(self.samples.len()); self.input_dimension];
        for i in 0..self.samples.len() {
            for j in 0..self.input_dimension {
                sorted_samples[j].push(self.samples[i][j]); // remember, push is just an indexed insert that checks capacity first. As long as capacity isn't exceeded, push is O(1)
            }
        }

        // now we sort along dim1
        for j in 0..self.input_dimension {
            sorted_samples[j].sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        // TODO: it might be worth checking if the above operation would be faster if I changed the order of the loops and sorted inside the outer loop. Maybe something to do with cache performance?

        for (idx, spline) in self.splines.iter_mut().enumerate() {
            let sample_idx = idx % self.input_dimension; // the first `input_dimension` splines belong to the first "node", so every `input_dimension` splines, we move to the next node and reset which inner sample vector we're looking at
            let sample = &sorted_samples[sample_idx];
            spline.update_knots_from_samples(sample, knot_adaptivity);
        }

        Ok(())
    }

    /// wipe the internal state that tracks the samples used to update the knot vectors
    pub fn clear_samples(&mut self) {
        self.samples.clear();
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
    pub fn backward(&mut self, error: &Vec<f32>) -> Result<Vec<f32>, LayerError> {
        if error.len() != self.output_dimension {
            return Err(LayerError::MissizedGradientError {
                actual: error.len(),
                expected: self.output_dimension,
            });
        }

        let mut input_error = vec![0.0; self.input_dimension];
        for i in 0..self.splines.len() {
            // every `input_dimension` splines belong to the same node, and thus will use the same error value.
            // "Distribute" the error at a given node among all incoming edges
            let error_at_edge_output =
                error[i / self.input_dimension] / self.input_dimension as f32;
            let error_at_edge_input = self.splines[i].backward(error_at_edge_output)?;
            input_error[i % self.input_dimension] += error_at_edge_input;
        }
        Ok(input_error)
    }

    /// update the control points for each incoming edge in this layer given the learning rate
    ///
    /// this function relies on mutated inner state and should be called after [`KanLayer::backward()`]
    pub fn update(&mut self, learning_rate: f32) {
        for spline in self.splines.iter_mut() {
            spline.update(learning_rate);
        }
    }

    /// zero out the gradients for each incoming edge in this layer
    pub fn zero_gradients(&mut self) {
        for spline in self.splines.iter_mut() {
            spline.zero_gradients();
        }
    }

    /// return the total number of parameters in this layer
    pub fn parameter_count(&self) -> usize {
        self.input_dimension * self.output_dimension * self.splines[0].parameter_count()
    }

    /// returns the total number of trainable parameters in this layer
    pub fn trainable_parameter_count(&self) -> usize {
        self.input_dimension * self.output_dimension * self.splines[0].trainable_parameter_count()
    }

    /// return the number of incoming edges to nodes in this layer
    pub fn total_edges(&self) -> usize {
        self.input_dimension * self.output_dimension
    }
}

impl PartialEq for KanLayer {
    // only in a VERY contrived case would two layers have equal splines but different input/output dimensions
    // but it's technically possible, so we've got to check it
    fn eq(&self, other: &Self) -> bool {
        self.splines == other.splines
            && self.input_dimension == other.input_dimension
            && self.output_dimension == other.output_dimension
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum LayerError {
    MissizedPreactsError { actual: usize, expected: usize },
    // If NaNs are in the activations, it's probably because the spline knot vectors had too many duplicates in a row
    NaNsError,
    NoSamplesError,
    MissizedGradientError { actual: usize, expected: usize },
    BackwardBeforeForwardError,
}

impl std::fmt::Display for LayerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LayerError::MissizedPreactsError { actual, expected } => {
                write!(
                    f,
                    "Bad preactivation length. Expected {}, got {}",
                    expected, actual
                )
            }
            LayerError::NaNsError => {
                write!(f, "NaNs in activations")
            }
            LayerError::NoSamplesError => {
                write!(f, "No samples to update knot vectors")
            }
            LayerError::MissizedGradientError { actual, expected } => {
                write!(
                    f,
                    "received error vector of length {} but required vector of length {}",
                    actual, expected
                )
            }
            LayerError::BackwardBeforeForwardError => {
                write!(f, "backward called before forward")
            }
        }
    }
}

impl From<spline::SplineError> for LayerError {
    fn from(value: spline::SplineError) -> Self {
        match value {
            spline::SplineError::BackwardBeforeForwardError => {
                LayerError::BackwardBeforeForwardError
            }
            _ => {
                panic!(
                    "attempted to convert a non-BackwardBeforeForward SplineError to LayerError"
                ); // panic because this should never happen
            }
        }
    }
}

impl std::error::Error for LayerError {}

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
        KanLayer {
            splines: vec![spline1.clone(), spline2.clone(), spline2, spline1],
            samples: vec![],
            input_dimension: 2,
            output_dimension: 2,
        }
    }

    #[test]
    fn test_new() {
        let input_dimension = 3;
        let output_dimension = 4;
        let k = 5;
        let coef_size = 6;
        let my_layer = KanLayer::new(KanLayerOptions {
            input_dimension,
            output_dimension,
            degree: k,
            coef_size,
        });
        assert_eq!(my_layer.output_dimension, output_dimension);
        assert_eq!(my_layer.input_dimension, input_dimension);
        assert_eq!(my_layer.splines.len(), input_dimension * output_dimension);
    }

    #[test]
    fn test_forward() {
        // to properly test layer forward, I need a layer with output and input dim = 2, which means 4 total edges
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let acts = layer.forward(&preacts).unwrap();
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
        let acts = layer.forward(&preacts);
        assert!(acts.is_err());
        assert_eq!(
            acts.err().unwrap(),
            LayerError::MissizedPreactsError {
                actual: 3,
                expected: 2
            }
        );
    }

    #[test]
    fn test_forward_then_backward() {
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let acts = layer.forward(&preacts).unwrap();
        let expected_activations = vec![0.3177, -0.3177];
        let rounded_activations: Vec<f32> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations, "forward failed");

        let error = vec![1.0, 0.5];
        let input_error = layer.backward(&error).unwrap();
        let expected_input_error = vec![0.0, 0.60156];
        let rounded_input_error: Vec<f32> = input_error
            .iter()
            .map(|f| (f * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(rounded_input_error, expected_input_error, "backward failed");
    }

    #[test]
    fn test_backward_before_forward() {
        let mut layer = build_test_layer();
        let error = vec![1.0, 0.5];
        let input_error = layer.backward(&error);
        assert!(input_error.is_err());
    }

    #[test]
    fn test_backward_bad_error_length() {
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let _ = layer.forward(&preacts).unwrap();
        let error = vec![1.0, 0.5, 0.5];
        let input_error = layer.backward(&error);
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
