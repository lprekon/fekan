// #![allow(dead_code)]
pub mod kan_layer_errors;
pub(crate) mod spline;

use kan_layer_errors::*;
use rand::distributions::Distribution;
use rand::thread_rng;
use rayon::prelude::*;
use rayon::{iter::IntoParallelIterator, ThreadPool};
use serde::{Deserialize, Serialize};
use spline::{linspace, spline_errors::*, Spline};
use statrs::distribution::Normal; // apparently the statrs distributions use the rand Distribution trait

use std::{
    sync::{Arc, Mutex},
    vec,
};

/// A layer in a Kolmogorov-Arnold neural Network (KAN)
///
/// A KAN layer consists of a number of nodes equal to the output dimension of the layer.
/// Each node has a number of incoming edges equal to the input dimension of the layer, and each edge holds a B-spline that operates on the value travelling down the edge

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
    samples: Vec<Vec<f64>>,
}

/// Hyperparameters for a KanLayer
///
/// # Examples
/// see [`KanLayer::new`]
#[derive(Debug, Copy, Clone)]
#[allow(missing_docs)]
pub struct KanLayerOptions {
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub degree: usize,
    pub coef_size: usize,
}

impl KanLayer {
    /// create a new layer with `output_dimension` nodes in this layer that each expect an `input_dimension`-long preactivation vector.
    ///
    /// All incoming edges will be created with a degree `degree` B-spline and `coef_size` control points.
    ///
    /// All B-splines areinitialized with coefficients drawn from astandard normal distribution, and with
    /// `degree + coef_size + 1` knots evenly spaced between -1.0 and 1.0.
    /// # Examples
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let input_dimension = 3;
    /// let output_dimension = 4;
    /// let layer_options = KanLayerOptions {
    ///     input_dimension,
    ///     output_dimension,
    ///     degree: 5,
    ///     coef_size: 6,
    /// };
    /// let my_layer = KanLayer::new(&layer_options);
    /// assert_eq!(my_layer.total_edges(), output_dimension * input_dimension);
    /// ```
    pub fn new(options: &KanLayerOptions) -> Self {
        let num_edges = options.input_dimension * options.output_dimension;
        let num_knots = options.coef_size + options.degree + 1;
        let normal_dist = Normal::new(0.0, 1.0).expect("unable to create normal distribution");
        let mut randomness = thread_rng();
        let splines = (0..num_edges)
            .map(|_| {
                let coefficients: Vec<f64> = (0..options.coef_size)
                    .map(|_| normal_dist.sample(&mut randomness) as f64)
                    .collect();
                Spline::new(options.degree, coefficients, linspace(-1.0, 1.0, num_knots))
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

    /// calculate the activations of the nodes in this layer given the preactivations. For a multi-threaded version of this function, see [`KanLayer::forward_concurrent`]
    /// This operation mutates internal state, which will be read in [`KanLayer::backward()`] and [`KanLayer::update_knots_from_samples()`]
    ///
    /// `preactivation.len()` must be equal to the layer's `input_dimension`
    /// # Errors
    /// Returns an [`LayerError`] if the length of `preactivation` is not equal to the input_dimension this layer, or if the activations contain NaNs.
    /// See [`LayerError`] for more information
    ///
    /// # Examples
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let input_dimension = 3;
    /// let output_dimension = 4;
    /// let layer_options = KanLayerOptions {
    ///     input_dimension,
    ///     output_dimension,
    ///     degree: 5,
    ///     coef_size: 6,
    /// };
    /// let mut my_layer = KanLayer::new(&layer_options);
    /// let preacts = vec![0.0, 0.5, 0.5];
    /// let acts = my_layer.forward(&preacts)?;
    /// assert_eq!(acts.len(), output_dimension);
    /// # Ok::<(), fekan::kan_layer::ForwardLayerError>(())
    /// ```
    pub fn forward(&mut self, preactivation: &[f64]) -> Result<Vec<f64>, ForwardLayerError> {
        self.forward_preamble(preactivation)?;

        // it probably makes sense to move straight down the list of splines, since that theoretically should have better cache performance
        // also, I guess I haven't decided (in code) how the splines are ordered, so there's no reason I can't say the first n splines all belong to the first node, etc.
        // I just have to be consistent when I get to back propagation
        let mut activations: Vec<f64> = vec![0.0; self.output_dimension];
        for (idx, spline) in self.splines.iter_mut().enumerate() {
            let act = spline.forward(preactivation[idx % self.input_dimension]); // the first `input_dimension` splines belong to the first "node", the second `input_dimension` splines belong to the second node, etc.
            if act.is_nan() {
                return Err(ForwardLayerError::NaNsError);
            }
            activations[(idx / self.input_dimension) as usize] += act; // every `input_dimension` splines, we move to the next node
        }

        // if activations.iter().any(|x| x.is_nan()) {
        //     return Err(ForwardLayerError::NaNsError);
        // }
        Ok(activations)
    }

    fn forward_preamble(&mut self, preactivation: &[f64]) -> Result<(), ForwardLayerError> {
        if preactivation.len() != self.input_dimension {
            return Err(ForwardLayerError::MissizedPreactsError {
                actual: preactivation.len(),
                expected: self.input_dimension,
            });
        }
        self.samples.push(preactivation.into());
        //  check the length here since it's the same check for the entire layer, even though the "node" is technically the part that cares
        // save a copy of the preactivation for updating the knot vectors later
        Ok(())
    }

    /// as [KanLayer::forward], but multi-threaded
    pub fn forward_concurrent(
        &mut self,
        preactivation: &[f64],
        thread_pool: &ThreadPool,
    ) -> Result<Vec<f64>, ForwardLayerError> {
        self.forward_preamble(preactivation)?;

        let num_splines = self.splines.len();
        let thread_safe_splines = self
            .splines
            .iter_mut()
            .map(|spline| Arc::new(Mutex::new(spline)))
            .collect::<Vec<_>>();
        let activations = thread_pool
            .install(|| {
                let acts = Arc::new(Mutex::new(vec![0.0; self.output_dimension]));
                (0..num_splines).into_par_iter().for_each(|idx| {
                    let mut acting_spline = thread_safe_splines[idx].lock().unwrap();
                    let act = acting_spline.forward(preactivation[idx % self.input_dimension]);
                    acts.lock().unwrap()[(idx / self.input_dimension) as usize] += act;
                    // every `input_dimension` splines, we move to the next node
                });
                acts
            })
            .lock()
            .unwrap()
            .to_vec();
        Ok(activations)
    }

    /// as [KanLayer::forward], but does not accumulate any internal state
    ///
    /// This method should be used during inference or validation, when the model is not being trained
    ///
    /// # Errors
    /// Returns an [`LayerError`] if the length of `preactivation` is not equal to the input_dimension this layer, or if the activations contain NaNs.

    pub fn infer(&self, preactivation: &[f64]) -> Result<Vec<f64>, ForwardLayerError> {
        if preactivation.len() != self.input_dimension {
            return Err(ForwardLayerError::MissizedPreactsError {
                actual: preactivation.len(),
                expected: self.input_dimension,
            });
        }

        let mut activations: Vec<f64> = vec![0.0; self.output_dimension];
        for (idx, spline) in self.splines.iter().enumerate() {
            let act = spline.infer(preactivation[idx % self.input_dimension]);
            activations[(idx / self.input_dimension) as usize] += act;
        }

        if activations.iter().any(|x| x.is_nan()) {
            return Err(ForwardLayerError::NaNsError);
        }
        Ok(activations)
    }

    /// Using samples memoized by [`KanLayer::forward`], update the knot vectors for each incoming edge in this layer.
    ///
    /// When `knot_adaptivity` is 0, the new knot vectors will be uniformly distributed over the range spanned by the samples;
    /// when `knot_adaptivity` is 1, the new knots will be placed at the quantiles of the samples. 0 < `knot_adaptivity` < 1 will interpolate between these two extremes.
    ///
    /// ## Warning
    /// calling this function with `knot_adaptivity = 1` can result in a large number of knots being placed at the same value, which can cause NaNs in the activations. See [`LayerError`] for more information
    ///
    /// calling this function with fewer samples than the number of knots in a spline AND `knot_adaptivity` > 0 results in undefined behavior
    ///
    /// # Errors
    /// Returns an error if the layer has no memoized samples, which most likely means that [`KanLayer::forward`] has not been called since initialization or the last call to [`KanLayer::clear_samples`]
    ///
    /// # Examples
    /// Update the knot vectors for a layer to cover the range of the samples. In practice, this function should be called every hundred forward passes or so
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// # let some_layer_options = KanLayerOptions {input_dimension: 2,output_dimension: 4,degree: 5, coef_size: 6};
    /// let mut my_layer = KanLayer::new(&some_layer_options);
    /// let sample1 = vec![100f64, -100f64];
    /// let sample2 = vec![-100f64, 100f64];
    ///
    /// let acts = my_layer.forward(&sample1).unwrap();
    /// assert!(acts.iter().all(|x| *x == 0.0)); // the preacts were all outside the initial knot range, so the activations should all be 0
    /// let acts = my_layer.forward(&sample2).unwrap();
    /// assert!(acts.iter().all(|x| *x == 0.0)); // the preacts were all outside the initial knot range, so the activations should all be 0
    /// my_layer.update_knots_from_samples(0.0).unwrap(); // we don't have enough samples to calculate quantiles, so we have to keep the knots uniformly distributed. In practice, this function should be called every few hundred forward passes or so
    /// let new_acts = my_layer.forward(&sample1).unwrap();
    /// assert!(new_acts.iter().all(|x| *x != 0.0)); // the knot range now covers the samples, so the activations should be non-zero
    /// # Ok::<(), fekan::kan_layer::UpdateLayerKnotsError>(())
    /// ```
    pub fn update_knots_from_samples(
        &mut self,
        knot_adaptivity: f64,
    ) -> Result<(), UpdateLayerKnotsError> {
        if self.samples.is_empty() {
            return Err(UpdateLayerKnotsError::NoSamplesError);
        }

        // lets construct a sorted vector of the samples for each incoming value
        // first we transpose the samples, so that dim0 = input_dimension, dim1 = number of samples
        let mut sorted_samples: Vec<Vec<f64>> =
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
    ///
    /// # Examples
    ///
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// # let some_layer_options = KanLayerOptions {input_dimension: 2,output_dimension: 4,degree: 5, coef_size: 6};
    /// let mut my_layer = KanLayer::new(&some_layer_options);
    /// /* After several forward passes... */
    /// # let sample1 = vec![100f64, -100f64];
    /// # let sample2 = vec![-100f64, 100f64];
    /// # let _acts = my_layer.forward(&sample1)?;
    /// # let _acts = my_layer.forward(&sample2)?;
    /// let update_result = my_layer.update_knots_from_samples(0.0);
    /// assert!(update_result.is_ok());
    /// my_layer.clear_samples();
    /// let update_result = my_layer.update_knots_from_samples(0.0);
    /// assert!(update_result.is_err()); // we've cleared the samples, so we can't update the knot vectors
    /// # Ok::<(), fekan::kan_layer::ForwardLayerError>(())
    pub fn clear_samples(&mut self) {
        self.samples.clear();
    }

    /// Given a vector of error values for the nodes in this layer, backpropogate the error through the layer, updating the internal gradients for the incoming edges
    /// and return the error for the previous layer. Use [`KanLayer::backward_concurrent`] for a multi-threaded version of this function
    ///
    /// This function relies on mutated inner state and should be called after [`KanLayer::forward`].
    ///
    /// Calculated gradients are stored internally, and only applied during [`KanLayer::update`].
    ///
    /// # Errors
    /// Returns a [`LayerError`] if the length of `error` is not equal to the number of nodes in this layer, or if `backward` is called before `forward`
    ///
    /// # Examples
    /// Backpropgate the error through a two-layer network, and update the gradients
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let first_layer_options = KanLayerOptions { input_dimension: 2, output_dimension: 4, degree: 5, coef_size: 6 };
    /// let second_layer_options = KanLayerOptions { input_dimension: 4, output_dimension: 3, degree: 5, coef_size: 6 };
    /// let mut first_layer = KanLayer::new(&first_layer_options);
    /// let mut second_layer = KanLayer::new(&second_layer_options);
    /// /* forward pass */
    /// let preacts = vec![0.0, 0.5];
    /// let acts = first_layer.forward(&preacts).unwrap();
    /// let output = second_layer.forward(&acts).unwrap();
    /// /* calculate error */
    /// # let error = vec![1.0, 0.5, 0.5];
    /// assert_eq!(error.len(), second_layer_options.output_dimension);
    /// let first_layer_error = second_layer.backward(&error).unwrap();
    /// assert_eq!(first_layer_error.len(), first_layer_options.output_dimension);
    /// let input_error = first_layer.backward(&first_layer_error).unwrap();
    /// assert_eq!(input_error.len(), first_layer_options.input_dimension);
    ///
    /// // apply the gradients
    /// let learning_rate = 0.1;
    /// first_layer.update(learning_rate);
    /// second_layer.update(learning_rate);
    /// // reset the gradients
    /// first_layer.zero_gradients();
    /// second_layer.zero_gradients();
    /// /* continue training */
    /// ```
    pub fn backward(&mut self, error: &[f64]) -> Result<Vec<f64>, BackwardLayerError> {
        if error.len() != self.output_dimension {
            return Err(BackwardLayerError::MissizedGradientError {
                actual: error.len(),
                expected: self.output_dimension,
            });
        }
        // if error.iter().any(|f| f.is_nan()) {
        //     return Err(BackwardLayerError::ReceivedNanError);
        // }

        let mut input_error = vec![0.0; self.input_dimension];
        for i in 0..self.splines.len() {
            // every `input_dimension` splines belong to the same node, and thus will use the same error value.
            // "Distribute" the error at a given node among all incoming edges
            let error_at_edge_output =
                error[i / self.input_dimension] / self.input_dimension as f64;
            let error_at_edge_input = self.splines[i].backward(error_at_edge_output)?;
            input_error[i % self.input_dimension] += error_at_edge_input;
        }
        Ok(input_error)
    }

    /// as [KanLayer::backward], but divides the work among the passed thread pool
    pub fn backward_concurrent(
        &mut self,
        error: &[f64],
        thread_pool: &ThreadPool,
    ) -> Result<Vec<f64>, BackwardLayerError> {
        if error.len() != self.output_dimension {
            return Err(BackwardLayerError::MissizedGradientError {
                actual: error.len(),
                expected: self.output_dimension,
            });
        }
        // if error.iter().any(|f| f.is_nan()) {
        //     return Err(BackwardLayerError::ReceivedNanError);
        // }

        let backprop_result: (Vec<f64>, Vec<BackwardSplineError>) = thread_pool.install(|| {
            let mut input_gradient = vec![0.0; self.input_dimension];
            let mut spline_errors = Vec::with_capacity(self.splines.len());
            for i in 0..self.splines.len() {
                // every `input_dimension` splines belong to the same node, and thus will use the same error value.
                // "Distribute" the error at a given node among all incoming edges
                let error_at_edge_output =
                    error[i / self.input_dimension] / self.input_dimension as f64;
                match self.splines[i].backward(error_at_edge_output) {
                    Ok(error_at_edge_input) => {
                        input_gradient[i % self.input_dimension] += error_at_edge_input
                    }
                    Err(e) => spline_errors.push(e),
                }
            }
            (input_gradient, spline_errors)
        });
        if !backprop_result.1.is_empty() {
            return Err(backprop_result.1[0].into());
        }
        Ok(backprop_result.0)
    }

    /// set the length of the knot vectors for each incoming edge in this layer
    ///
    /// Generally used multiple times throughout training to increase the number of knots in the spline to increase fidelity of the curve
    /// # Notes
    /// * The number of control points is set to `knot_length - degree - 1`, and the control points are calculated using lstsq over cached samples to approximate the previous curve
    /// * This method clears the internal cache of samples used to update knots in [`KanLayer::update_knots_from_samples`] (as if [`KanLayer::clear_samples`] was called) so this function and [`update_knots_from_samples`](KanLayer::update_knots_from_samples) should not be called in the same training step (ideally, they should be separated by many training steps - maybe 100 or so)
    /// * This method clears the internal cache of samples used by [`KanLayer::forward`], so frequent calls to this method may slow down training
    /// # Errors
    /// Returns a [`UpdateLayerKnotsError`] if the layer has no samples in the internal cache . This error is usually caused by calling this function before calling [`KanLayer::forward`], or by not calling [`KanLayer::forward`] between the last call to a cache-clearing function and this function
    /// # Examples
    /// Extend the knot vectors during training to increase the fidelity of the splines
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let input_dimension = 2;
    /// let output_dimension = 4;
    /// let degree = 5;
    /// let coef_size = 6;
    /// let layer_options = KanLayerOptions {
    ///     input_dimension,
    ///     output_dimension,
    ///     degree,
    ///     coef_size
    /// };
    /// let mut my_layer = KanLayer::new(&layer_options);
    ///
    /// let num_splines = input_dimension * output_dimension;
    /// let expected_knots_per_spline = coef_size + degree + 1;
    /// assert_eq!(my_layer.knot_length(), expected_knots_per_spline, "starting knots per edge");
    /// assert_eq!(my_layer.parameter_count(), num_splines * (coef_size + my_layer.knot_length()), "starting parameter count");
    ///
    /// /* train the layer a bit to start shaping the splines */
    /// # let sample = vec![0.0; input_dimension];
    /// # let _ = my_layer.forward(&sample);
    ///
    /// let new_knot_length = my_layer.knot_length() * 2;
    /// let update_result = my_layer.set_knot_length(new_knot_length);
    /// assert!(update_result.is_ok(), "update knots"); // we have samples from training, so we can update the knot vectors
    ///
    /// assert_eq!(my_layer.knot_length(), new_knot_length, "ending knots per edge");
    /// let new_coef_size = new_knot_length - degree - 1;
    /// assert_eq!(my_layer.parameter_count(), num_splines * (new_coef_size + new_knot_length), "ending parameter count");
    ///
    /// /* continue training layer, now with increased fidelity in the spline */
    /// ```
    /// Try to extend the knot vectors without first calling [`KanLayer::forward`]
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// # let input_dimension = 2;
    /// # let output_dimension = 4;
    /// # let degree = 5;
    /// # let coef_size = 6;
    /// # let layer_options = KanLayerOptions {
    /// #     input_dimension,
    /// #     output_dimension,
    /// #     degree,
    /// #     coef_size
    /// # };
    ///
    /// let mut my_layer = KanLayer::new(&layer_options);
    ///
    /// // oops - I wanted splines with greater fidelity. I'll just extend the knot vector real quick...
    /// let the_knot_lenght_i_really_wanted = my_layer.knot_length() * 2;
    /// let update_result = my_layer.set_knot_length(the_knot_lenght_i_really_wanted);
    /// assert!(update_result.is_err(), "update knots"); // we haven't called forward, so we can't update the knot vectors
    /// ```
    /// Try to extend the knot vectors immediately after updating them with [`KanLayer::update_knots_from_samples`]
    /// ```
    /// # use fekan::kan_layer::KanLayerOptions;
    /// use fekan::kan_layer::KanLayer;
    /// # let input_dimension = 2;
    /// # let output_dimension = 4;
    /// # let degree = 5;
    /// # let coef_size = 6;
    /// # let layer_options = KanLayerOptions {
    /// #     input_dimension,
    /// #     output_dimension,
    /// #     degree,
    /// #     coef_size
    /// # };
    /// # let mut my_layer = KanLayer::new(&layer_options);
    /// # let sample1 = vec![0.0; input_dimension];
    /// # let sample2 = vec![0.0; input_dimension];
    /// let _acts = my_layer.forward(&sample1);
    /// let _acts = my_layer.forward(&sample2);
    /// let update_knot_range_result = my_layer.update_knots_from_samples(0.0);
    /// assert!(update_knot_range_result.is_ok(), "update knot range"); // we have samples from training, so we can update the knot vectors
    /// // Well, that's the end of the epoch. I guess now's a good time to extend the knot vectors...
    /// let extend_knot_vector_result = my_layer.set_knot_length(my_layer.knot_length() * 2);
    /// assert!(extend_knot_vector_result.is_err(), "extend knot vector"); // we've cleared the samples, so we can't update the knot vectors
    /// ```
    /// # Panics
    /// Panics if the Singular Value Decomposition (SVD) used to calculate the control points fails. This should never happen, but if it does, it's a bug
    pub fn set_knot_length(&mut self, knot_length: usize) -> Result<(), UpdateLayerKnotsError> {
        for spline in self.splines.iter_mut() {
            spline.set_knot_length(knot_length)?;
        }
        Ok(())
    }

    /// return the length of the knot vectors for each incoming edge in this layer
    /// # Examples
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let layer_options = KanLayerOptions {
    ///     input_dimension: 2,
    ///     output_dimension: 4,
    ///     degree: 5,
    ///     coef_size: 6
    /// };
    /// let mut my_layer = KanLayer::new(&layer_options);
    /// assert_eq!(my_layer.knot_length(), 6 + 5 + 1);
    pub fn knot_length(&self) -> usize {
        self.splines[0].knots().len()
    }

    /// update the control points for each incoming edge in this layer given the learning rate
    ///
    /// this function relies on internally stored gradients calculated during [`KanLayer::backward()`]
    ///
    /// # Examples
    /// see [`KanLayer::backward`]
    pub fn update(&mut self, learning_rate: f64) {
        for spline in self.splines.iter_mut() {
            spline.update(learning_rate);
        }
    }

    /// clear gradients for each incoming edge in this layer
    ///
    /// # Examples
    /// see [`KanLayer::backward`]
    pub fn zero_gradients(&mut self) {
        for spline in self.splines.iter_mut() {
            spline.zero_gradients();
        }
    }

    /// return the total number of parameters in this layer.
    /// A layer has `input_dimension * output_dimension` splines, each with `degree + coef_size + 1` knots and `coef_size` control points
    ///
    /// #Examples
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let layer_options = KanLayerOptions {
    ///     input_dimension: 2,
    ///     output_dimension: 4,
    ///     degree: 5,
    ///     coef_size: 6
    /// };
    /// let my_layer = KanLayer::new(&layer_options);
    /// assert_eq!(my_layer.parameter_count(), 2 * 4 * (6 + (5 + 6 + 1)));
    ///```
    pub fn parameter_count(&self) -> usize {
        self.input_dimension * self.output_dimension * self.splines[0].parameter_count()
    }

    /// returns the total number of trainable parameters in this layer.
    /// A layer has `input_dimension * output_dimension` splines, each with coef_size` control points, which are the trainable parameter in a KAN layer
    /// # Examples
    /// ```
    /// use fekan::kan_layer::{KanLayer, KanLayerOptions};
    /// let layer_options = KanLayerOptions {
    ///     input_dimension: 2,
    ///     output_dimension: 4,
    ///     degree: 5,
    ///     coef_size: 6
    /// };
    /// let my_layer = KanLayer::new(&layer_options);
    /// assert_eq!(my_layer.trainable_parameter_count(), 2 * 4 * 6);
    ///```
    pub fn trainable_parameter_count(&self) -> usize {
        self.input_dimension * self.output_dimension * self.splines[0].trainable_parameter_count()
    }

    /// return the number of incoming edges to nodes in this layer
    pub fn total_edges(&self) -> usize {
        self.input_dimension * self.output_dimension
    }

    // /// Get the cache stats for each spline in this layer. This is useful for debugging and performance tuning
    // pub fn cache_stats(&self) -> Vec<&Vec<Vec<CacheStats>>> {
    //     self.splines.iter().map(|s| s.cache_stats()).collect()
    // }
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

#[cfg(test)]
mod test {
    use rayon::ThreadPoolBuilder;
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
            knots[i] = -1.0 + (i as f64 / (knot_size - 1) as f64 * 2.0);
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
        let my_layer = KanLayer::new(&KanLayerOptions {
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
        let rounded_activations: Vec<f64> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations);
    }

    #[test]
    fn test_forward_concurrent() {
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let thread_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let acts = layer.forward_concurrent(&preacts, &thread_pool).unwrap();
        let expected_activations = vec![0.3177, -0.3177];
        let rounded_activations: Vec<f64> = acts
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
            ForwardLayerError::MissizedPreactsError {
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
        let rounded_activations: Vec<f64> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations, "forward failed");

        let error = vec![1.0, 0.5];
        let input_error = layer.backward(&error).unwrap();
        let expected_input_error = vec![0.0, 0.60156];
        let rounded_input_error: Vec<f64> = input_error
            .iter()
            .map(|f| (f * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(rounded_input_error, expected_input_error, "backward failed");
    }

    #[test]
    fn test_forward_then_backward_concurrent() {
        let thread_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let acts = layer.forward(&preacts).unwrap();
        let expected_activations = vec![0.3177, -0.3177];
        let rounded_activations: Vec<f64> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations, "forward failed");

        let error = vec![1.0, 0.5];
        let input_error = layer.backward_concurrent(&error, &thread_pool).unwrap();
        let expected_input_error = vec![0.0, 0.60156];
        let rounded_input_error: Vec<f64> = input_error
            .iter()
            .map(|f| (f * 100000.0).round() / 100000.0)
            .collect();
        assert_eq!(rounded_input_error, expected_input_error, "backward failed");
    }

    #[test]
    fn test_forward_concurrent_then_backward_concurrent() {
        let thread_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut layer = build_test_layer();
        let preacts = vec![0.0, 0.5];
        let acts = layer.forward_concurrent(&preacts, &thread_pool).unwrap();
        let expected_activations = vec![0.3177, -0.3177];
        let rounded_activations: Vec<f64> = acts
            .iter()
            .map(|x| (x * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_activations, expected_activations, "forward failed");

        let error = vec![1.0, 0.5];
        let input_error = layer.backward_concurrent(&error, &thread_pool).unwrap();
        let expected_input_error = vec![0.0, 0.60156];
        let rounded_input_error: Vec<f64> = input_error
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
    fn test_backward_concurrent_before_forward() {
        let thread_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut layer = build_test_layer();
        let error = vec![1.0, 0.5];
        let input_error = layer.backward_concurrent(&error, &thread_pool);
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

    #[test]
    fn test_layer_send() {
        fn assert_send<T: Send>() {}
        assert_send::<KanLayer>();
    }

    #[test]
    fn test_layer_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KanLayer>();
    }

    #[test]
    fn test_forward_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ForwardLayerError>();
    }

    #[test]
    fn test_forward_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ForwardLayerError>();
    }

    #[test]
    fn test_backward_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BackwardLayerError>();
    }

    #[test]
    fn test_backward_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BackwardLayerError>();
    }

    #[test]
    fn test_knot_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<UpdateLayerKnotsError>();
    }

    #[test]
    fn test_knot_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<UpdateLayerKnotsError>();
    }
}
