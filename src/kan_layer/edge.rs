//! I tried hard to avoid making this a massive mono-type. I really did. I tried to create an Edge trait to serve as an interface, so KanLayer could
//! be full dyn Edge trait objects, thus hiding the actual function of each individual edge, and making it easier to drop in new edges. I tried.
//!
//! But, it turns out, while Rust is great at many things, it's not great at collections of heterogeneous types. I was able to create the trait, and after
//! some jerry-rigging I was able to effectively implement Clone on it, despite the fact that traits with Sized in their supertrait-ancestry aren't object safe,
//! and I was able to implement Send and Sync, despite the fact that owned trait objects have to live in a Box<>, which is usually neither Send nor Sync.
//! But I just couldn't implement Serialize or Deserialize. I got close, but the trait object abstraction hides critical information necessary for (de)serialization.
//! Even if I was able to rig up a semi-custom serialization scheme, there's just no way to DEserialize into a trait object without matching on the concrete type, because there's no "clever" way to get the trait object to point to the right V-table.
//! One of the major drawbacks of Rust's lack of runtime refelction or type-system in favor of compile-time reflection through macros.
//!
//! Obviously the code that suggests and clamps-to symbolic edges needs to be aware of every possible edge type - that's unavoidable - but if I have to do it anywhere else as well, if we can't have elegance no matter what, we might as well take the complexity out and
//! "brute force" it with a single massive Edge type that matches on a mode flag every method. At least this way, I get serialization and thread safety for free, and all the case-consciouness is in one place
//! (besides maybe the aforementioned code that suggests and clamps-to symbolic edges, but that's rather unavoidable, as I said)

use log::{debug, trace};
use nalgebra::{DMatrix, DVector, SVD};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, slice::Iter, thread, vec};
use strum::{EnumIter, IntoEnumIterator};

pub(crate) mod edge_errors;
use edge_errors::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Edge {
    kind: EdgeType,
    #[serde(skip)]
    // only used during operation
    last_t: Vec<f64>,
    #[serde(skip)] // only used during training
    l1_norm: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
enum EdgeType {
    Spline {
        // degree, control points, and knots are the parameters of the spline
        // these four fields constitute the "identity" of the spline, so they're the only ones that get serialized, considered for equality, etc.
        degree: usize,
        control_points: Vec<f64>,
        knots: Vec<f64>,
        residual_weight: f64,

        // the remaining fields represent the "state" of the spline.
        // They're in flux during operation, and so are ignored for any sort of persistence or comparison.
        /// the most recent parameter used in the forward pass

        /// the activations of the spline at each interval, stored from calls to [`forward()`](Spline::forward) and cleared on calls to [`update_knots_from_samples()`](Spline::update_knots_from_samples)
        #[serde(skip)] // only used during training
        /// dim0: degree (idx 0 = self.degree, idx 1 = self.degree - 1, etc.), dim1: control point index, dim2: t value
        activations: Vec<Vec<FxHashMap<u64, f64>>>,
        /// accumulated gradients for each control point
        #[serde(skip)] // only used during training
        gradients: Vec<Gradient>,
        #[serde(skip)] // only used during training
        residual_gradient: Gradient,
    },
    Symbolic {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        function: SymbolicFunction,
    },
    Pruned,
}

#[derive(Debug, Clone, PartialEq)]
struct Gradient {
    prediction_gradient: f64,
    l1_gradient: f64,
    entropy_gradient: f64,
}

impl Default for Gradient {
    fn default() -> Self {
        Gradient {
            prediction_gradient: 0.0,
            l1_gradient: 0.0,
            entropy_gradient: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, EnumIter)]
enum SymbolicFunction {
    Linear,
    Quadratic,
    Cubic,
    Quartic,
    Quintic,
    SquareRoot,
    CubeRoot,
    FourthRoot,
    FifthRoot,
    // CubeSqrt,
    Sin,
    Tan,
    Log,
    Exp,
    Inverse,
    // InverseSquared,
    // InverseCubed,
    // InverseQuartic,
    // InverseQuintic,
    // InverseSqrt,
    // InverseCbrt,
    // InverseCubeSqrt,
}

impl Edge {
    /// construct a new spline from the given degree, control points, and knots
    ///
    /// # Errors
    /// returns an error if the length of the knot vector is not at least `|control_points| + degree + 1`
    pub(super) fn new(
        degree: usize,
        control_points: Vec<f64>,
        knots: Vec<f64>,
        residual_weight: f64,
    ) -> Result<Self, EdgeError> {
        let size = control_points.len();
        let min_required_knots = size + degree + 1;
        if knots.len() < min_required_knots {
            return Err(EdgeError::TooFewKnots {
                expected: min_required_knots,
                actual: knots.len(),
            });
        }
        let num_control_points = control_points.len();
        let mut activations_cache = Vec::with_capacity(degree);
        for _ in 0..degree {
            let mut activations = Vec::with_capacity(num_control_points + degree);
            for _ in 0..num_control_points + degree {
                activations.push(FxHashMap::default());
            }
            activations_cache.push(activations);
        }
        Ok(Edge {
            kind: EdgeType::Spline {
                degree,
                control_points,
                knots,
                residual_weight,
                activations: activations_cache,
                gradients: vec![Gradient::default(); size],
                residual_gradient: Gradient::default(),
            },
            last_t: vec![],
            l1_norm: None,
        })
    }

    /// compute the point on the spline at the given parameter `t`
    ///
    /// accumulate the activations of the spline at each interval in the internal `activations` field
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.last_t.extend(inputs.iter()); // store the most recent input for use in the backward pass. This happens regardless of the edge type
        let mut outputs = Vec::with_capacity(inputs.len());
        match &mut self.kind {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                activations,
                residual_weight,
                ..
            } => {
                for t in inputs.iter() {
                    let mut sum = 0.0;
                    for (idx, coef) in control_points.iter().enumerate() {
                        let basis_activation =
                            basis_cached(idx, *degree, *t, &knots, activations, *degree);
                        sum += *coef * basis_activation;
                    }
                    let residual_activation = residual_function(*t) * *residual_weight;
                    sum += residual_activation;
                    outputs.push(sum);
                }
                // trace!("edge activations: {:?}", outputs);
            }
            _ => outputs = self.infer(inputs), // symbolic edges don't cache activations, so they have the same forward and infer implementations
        }
        self.l1_norm = Some(outputs.iter().map(|o| o.abs()).sum::<f64>() / outputs.len() as f64);
        outputs
    }

    /// comput the point on the spline at given parameter `t`
    ///
    /// Does not accumulate the activations of the spline at each interval in the internal `activations` field, or any other internal state
    pub fn infer(&self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = Vec::with_capacity(inputs.len());
        match &self.kind {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                residual_weight,
                ..
            } => {
                for t in inputs.iter() {
                    let mut sum = 0.0;
                    for (idx, coef) in control_points.iter().enumerate() {
                        let basis_activation = basis_no_cache(idx, *degree, *t, &knots);
                        sum += *coef * basis_activation;
                    }
                    let residual_activation = residual_function(*t) * *residual_weight;
                    sum += residual_activation;
                    outputs.push(sum);
                }
            }
            EdgeType::Symbolic {
                a,
                b,
                c,
                d,
                function,
            } => {
                let (a, b, c, d) = (*a, *b, *c, *d);
                for t in inputs {
                    let value = match function {
                        SymbolicFunction::Linear => c * (a * t + b) + d,
                        SymbolicFunction::Quadratic => c * (a * t + b).powi(2) + d,
                        SymbolicFunction::Cubic => c * (a * t + b).powi(3) + d,
                        SymbolicFunction::Quartic => c * (a * t + b).powi(4) + d,
                        SymbolicFunction::Quintic => c * (a * t + b).powi(5) + d,
                        SymbolicFunction::SquareRoot => c * (a * t + b).sqrt() + d,
                        SymbolicFunction::CubeRoot => c * (a * t + b).cbrt() + d,
                        SymbolicFunction::FourthRoot => c * (a * t + b).powf(0.25) + d,
                        SymbolicFunction::FifthRoot => c * (a * t + b).powf(0.2) + d,
                        SymbolicFunction::Sin => c * (a * t + b).sin() + d,
                        SymbolicFunction::Tan => c * (a * t + b).tan() + d,
                        SymbolicFunction::Log => c * (a * t + b).ln() + d,
                        SymbolicFunction::Exp => c * (a * t + b).exp() + d,
                        SymbolicFunction::Inverse => c / (a * t.max(f64::EPSILON) + b) + d,
                    };
                    outputs.push(value);
                }
            }
            EdgeType::Pruned => outputs = vec![0.0; inputs.len()], // pruned edges always return 0
        }
        outputs
    }

    /// compute the gradients for each control point  on the spline and accumulate them internally.
    ///
    /// returns the gradient of the input used in the forward pass,to be accumulated by the caller and passed back to the pervious layer as its error
    ///
    /// uses the memoized activations from the most recent forward pass
    ///
    /// # Errors
    /// * Returns [`SplineError::BackwardBeforeForward`] if called before a forward pass
    pub(super) fn backward(
        &mut self,
        edge_gradients: &[f64],
        layer_l1: f64,
    ) -> Result<Vec<f64>, EdgeError> {
        if self.last_t.is_empty() {
            return Err(EdgeError::BackwardBeforeForward);
        }
        // on the one hand, this assert technically shouldn't be here, since one ought to be able to do multiple forward/backward pass-pairs in a row without issue
        // on the other, it's a little annoying to deal with such a case, and the reasons for doing so are quite niche, since we now support batch passes.
        // so I'm just going to include the assert fow now - better than letting the model just be wrong.
        assert!(
            self.last_t.len() == edge_gradients.len(),
            "sorry - for now, please clear state between forward/backward pass-pairs. We have {} t values and {} edge gradients", self.last_t.len(), edge_gradients.len()
        );
        let edge_l1 = self.l1_norm.expect("edge_l1 is None");
        assert_eq!(layer_l1.signum(), 1.0);
        match &mut self.kind {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                residual_weight,
                activations,
                gradients: accumulated_gradients,
                residual_gradient: residual_weight_gradient,
            } => {
                // assert_eq!(activations[0][0].len(), edge_gradients.len());
                // prettu sure this is wrong, but I want to change it when I'm not changing anything else so I can test it in isolation
                let adjusted_error: Vec<f64> = edge_gradients
                    .iter()
                    .map(|e| e / control_points.len() as f64)
                    .collect(); // distribute the error evenly across all control points

                // drt_output_wrt_input = sum_i(dB_ik(t) * C_i) + residual_weight * drt_residual_wrt_input
                let mut drts_output_wrt_input = vec![0.0; edge_gradients.len()];
                let k = *degree;
                // backward pass for the control points
                for i in 0..control_points.len() {
                    // L1 norm of edge = 1/N * Σself.infer(t).abs())
                    // L1 loss of model = ΣL1 norm of edge ( * L1_LAMBDA)
                    // ∴ d_model_L1/d_C_i = d_model_L1/d_edge_L1 * d_edge_L1/d_C_i
                    // d_model_L1/d_edge_L1 = 1
                    // d_edge_L1/d_C_i = 1/N * ΣB_i(t) * sign(C_i)
                    assert!(!activations[0][i].is_empty());
                    let basis_activations: Vec<f64> = self
                        .last_t
                        .iter()
                        .map(|t| *activations[0][i].get(&t.to_bits()).unwrap())
                        .collect();
                    let d_edge_l1_d_ci = basis_activations.iter().sum::<f64>()
                        / basis_activations.len() as f64
                        * control_points[i].signum();
                    assert!(d_edge_l1_d_ci.is_finite(), "d_edge_l1_d_ci is not finite");

                    // Entropy loss of layer = -Σ (edge_L1/layer_L1 * log(edge_L1/layer_L1))
                    // ∴ d_layer_entropy/d_C_i = d_layer_entropy/d_edge_L1 * d_edge_L1/d_C_i
                    // assert_ne!(edge_l1_norm, 0.0, "edge_l1_norm is 0");
                    // assert_ne!(layer_l1, 0.0, "layer_l1 is 0");
                    let c = layer_l1 - edge_l1;
                    assert!(c.is_finite(), "c is not finite");
                    let d =
                        (edge_l1 / (layer_l1 + f64::MIN_POSITIVE) + f64::MIN_POSITIVE).ln() + 1.0;
                    assert!(d.is_finite(), "d is not finite");
                    let e = layer_l1.powi(2) + f64::MIN_POSITIVE;
                    assert!(e.is_finite(), "e is not finite");
                    let d_entropy_d_edge_l1 = -1.0 * c * d / e;
                    assert!(
                        d_entropy_d_edge_l1.is_finite(),
                        "d_entropy_d_edge_l1 is not finite"
                    );

                    let l1_gradient = d_edge_l1_d_ci;
                    let entrop_gradient = d_entropy_d_edge_l1 * d_edge_l1_d_ci;
                    // d_pred_loss/d_C_i = d_pred_loss/d_edge_output * d_edge_output/d_C_i
                    // d_pred_loss/d_edge_output = 1
                    // d_edge_output/d_C_i = B_i(t)
                    let basis_activations = self
                        .last_t
                        .iter()
                        .map(|t| activations[0][i].get(&t.to_bits()).unwrap());
                    let prediction_gradients = adjusted_error
                        .iter()
                        .zip(basis_activations)
                        .map(|(e, a)| e * a);
                    accumulated_gradients[i].prediction_gradient +=
                        prediction_gradients.sum::<f64>();
                    accumulated_gradients[i].l1_gradient += l1_gradient;
                    accumulated_gradients[i].entropy_gradient += entrop_gradient;

                    // calculate the derivative of the spline output with respect to the input (as opposed to wrt the control points)
                    // dB_ik(t) = (k-1)/(t_i+k-1 - t_i) * B_i(k-1)(t) - (k-1)/(t_i+k - t_i+1) * B_i+1(k-1)(t)
                    let left_coef = (k as f64 - 1.0) / (knots[i + k - 1] - knots[i]);
                    let right_coef = (k as f64 - 1.0) / (knots[i + k] - knots[i + 1]);
                    let recurses = self.last_t.iter().map(|t| {
                        (
                            basis_cached(i, k - 1, *t, knots, activations, k),
                            basis_cached(i + 1, k - 1, *t, &knots, activations, k),
                        )
                    });

                    // println!(
                    //     "i: {} left: {}, right: {}, left_recurse: {}, right_recurse: {}",
                    //     i, left, right, left_recurse, right_recurse
                    // );
                    let basis_derivative =
                        recurses.map(|(left, right)| left * left_coef - right * right_coef);
                    drts_output_wrt_input = drts_output_wrt_input
                        .iter()
                        .zip(basis_derivative)
                        .map(|(drt, bd)| drt + bd * control_points[i])
                        .collect();
                }
                // backward pass for the residual
                for i in 0..adjusted_error.len() {
                    let drt_residual_wrt_weight =
                        adjusted_error[i] * residual_function(self.last_t[i]);
                    residual_weight_gradient.prediction_gradient += drt_residual_wrt_weight;
                    assert!(
                        drt_residual_wrt_weight.is_finite(),
                        "drt_residual_wrt_weight is not finite"
                    );

                    let d_edge_l1_d_rw =
                        residual_function(self.last_t[i]) * residual_weight.signum();
                    assert!(d_edge_l1_d_rw.is_finite(), "d_edge_l1_d_rw is not finite");

                    let c = layer_l1 - edge_l1;
                    assert!(c.is_finite(), "c2 is not finite");
                    let d =
                        (edge_l1 / (layer_l1 + f64::MIN_POSITIVE) + f64::MIN_POSITIVE).ln() + 1.0;
                    assert!(d.is_finite(), "d2 is not finite");
                    let e = layer_l1.powi(2) + f64::MIN_POSITIVE;
                    assert!(e.is_finite(), "e2 is not finite");
                    let d_entropy_d_edge_l1 = -1.0 * c * d / e;
                    assert!(
                        d_entropy_d_edge_l1.is_finite(),
                        "d_entropy_d_edge_l1 is not finite"
                    );

                    let l1_gradient = d_edge_l1_d_rw;
                    let entrop_gradient = d_entropy_d_edge_l1 * d_edge_l1_d_rw;

                    residual_weight_gradient.l1_gradient += l1_gradient;
                    residual_weight_gradient.entropy_gradient += entrop_gradient;

                    let residual_input_gradient =
                        *residual_weight * drt_residual_wrt_t(self.last_t[i]);
                    assert!(
                        residual_input_gradient.is_finite(),
                        "residual_input_gradient is not finite - weight: {}, drt_residual_wrt_t: {}, t: {}", residual_weight, drt_residual_wrt_t(self.last_t[i]), self.last_t[i]
                    );
                    drts_output_wrt_input[i] += residual_input_gradient;
                }
                // input_gradient = drt_output_wrt_input * error
                return Ok(drts_output_wrt_input
                    .iter()
                    .zip(edge_gradients.iter())
                    .map(|(drt, g)| drt * g)
                    .collect());
            }
            EdgeType::Symbolic {
                a,
                b,
                c,
                d,
                function,
            } => {
                let (a, b, c, _) = (*a, *b, *c, *d);
                let drts_output_wrt_input = self.last_t.iter().map(|t| match function {
                    SymbolicFunction::Linear => c * a,
                    SymbolicFunction::Quadratic => 2.0 * a * c * (a * t + b),
                    SymbolicFunction::Cubic => 3.0 * a * c * (a * t + b).powi(2),
                    SymbolicFunction::Quartic => 4.0 * a * c * (a * t + b).powi(3),
                    SymbolicFunction::Quintic => 5.0 * a * c * (a * t + b).powi(4),
                    SymbolicFunction::SquareRoot => 0.5 * c * (a * t + b).powf(-0.5),
                    SymbolicFunction::CubeRoot => (1.0 / 3.0) * c * (a * t + b).powf(-2.0 / 3.0),
                    SymbolicFunction::FourthRoot => 0.25 * c * (a * t + b).powf(-0.75),
                    SymbolicFunction::FifthRoot => 0.2 * c * (a * t + b).powf(-0.8),
                    SymbolicFunction::Sin => c * a * (a * t + b).cos(),
                    SymbolicFunction::Tan => c * a / (a * t + b).cos().powi(2),
                    SymbolicFunction::Log => c * a / (a * t + b),
                    SymbolicFunction::Exp => c * a * (a * t + b).exp(),
                    SymbolicFunction::Inverse => -c * a / (a * t + b).powi(2),
                });
                Ok(drts_output_wrt_input
                    .zip(edge_gradients.iter())
                    .map(|(ig, g)| ig * g)
                    .collect())
            }
            EdgeType::Pruned => Ok(vec![0.0; edge_gradients.len()]), // pruned edges always return 0
        }
    }

    pub(super) fn update_control_points(
        &mut self,
        learning_rate: f64,
        l1_lambda: f64,
        entropy_lambda: f64,
    ) {
        match &mut self.kind {
            EdgeType::Spline {
                control_points,
                residual_weight,
                gradients,
                residual_gradient,
                ..
            } => {
                for i in 0..control_points.len() {
                    control_points[i] -= learning_rate
                        * (gradients[i].prediction_gradient
                            + l1_lambda * gradients[i].l1_gradient
                            + entropy_lambda * gradients[i].entropy_gradient);
                }
                *residual_weight -= learning_rate
                    * (residual_gradient.prediction_gradient
                        + l1_lambda * residual_gradient.l1_gradient
                        + entropy_lambda * residual_gradient.entropy_gradient);
            }
            _ => (), // update on a non-spline edge is a no-op
        }
    }

    pub(super) fn zero_gradients(&mut self) {
        match &mut self.kind {
            EdgeType::Spline {
                gradients,
                residual_gradient,
                ..
            } => {
                for i in 0..gradients.len() {
                    gradients[i] = Gradient::default();
                }
                *residual_gradient = Gradient::default();
            }
            _ => (), // zeroing gradients on a non-spline edge is a no-op
        }
    }

    #[allow(dead_code)]
    // used in tests for parent module
    pub(super) fn knots<'a>(&'a self) -> Iter<'a, f64> {
        match &self.kind {
            EdgeType::Spline { knots, .. } => knots.iter(),
            _ => Iter::default(),
        }
    }

    pub(super) fn l1_norm(&self) -> Option<f64> {
        self.l1_norm
    }

    // pub(super) fn control_points(&self) -> Iter<'_, f64> {
    //     self.control_points.iter()
    // }

    /// given a sorted slice of previously seen inputs, update the knot vector to be a linear combination of a uniform vector and a vector of quantiles of the samples.
    ///
    /// If the new knots would contain `degree` or more duplicates - generally caused by too many duplicates in the samples - the knots are not updated
    ///
    /// If `samples` is not sorted, the results of the update and future spline operation are undefined.
    pub(super) fn update_knots_from_samples(&mut self, samples: &[f64], knot_adaptivity: f64) {
        trace!("updating knots from samples: {:?}", samples);
        self.clear_state();
        match &mut self.kind {
            EdgeType::Spline { degree, knots, .. } => {
                let knot_count = knots.len();
                let base_knot_count = knot_count - 2 * (*degree);
                let mut adaptive_knots: Vec<f64> = Vec::with_capacity(base_knot_count);
                let num_intervals = base_knot_count - 1;
                let stride_size = samples.len() / (num_intervals);
                for i in 0..num_intervals {
                    adaptive_knots.push(samples[i * stride_size]);
                }
                adaptive_knots.push(samples[samples.len() - 1]);

                let span_min = samples[0];
                let span_max = samples[samples.len() - 1];
                let uniform_knots = linspace(span_min, span_max, base_knot_count);

                let mut new_knots: Vec<f64> = adaptive_knots
                    .iter()
                    .zip(uniform_knots.iter())
                    .map(|(a, b)| a * knot_adaptivity + b * (1.0 - knot_adaptivity))
                    .collect();

                // make sure new_knots doesn't have too many duplicates
                let mut duplicate_count = 0;

                // pad the knot vectors at either end to prevent edge effects
                let step_size = (span_max - span_min) / (knot_count as f64);
                for _ in 0..*degree {
                    new_knots.insert(0, new_knots[0] - step_size);
                    new_knots.push(new_knots[new_knots.len() - 1] + step_size);
                }

                for i in 1..new_knots.len() {
                    if new_knots[i] == new_knots[i - 1] {
                        duplicate_count += 1;
                    } else {
                        duplicate_count = 0;
                    }
                    if duplicate_count >= base_knot_count.min(*degree) {
                        trace!("too many duplicate knots, not updating");
                        return; // we have too many duplicate knots, so we don't update the knots
                    }
                }
                *knots = new_knots;
            }
            _ => trace!("We don't update knots for non-spline edges"), // non-spline edges don't have knots, so this is a no-op
        }
    }

    pub(super) fn clear_state(&mut self) {
        self.last_t.clear();
        match self.kind {
            EdgeType::Spline {
                ref mut activations,
                ref mut gradients,
                ref mut residual_gradient,
                ..
            } => {
                for cache_line in activations.iter_mut() {
                    for cache in cache_line.iter_mut() {
                        cache.clear();
                    }
                }
                for gradient in gradients.iter_mut() {
                    *gradient = Gradient::default();
                }
                *residual_gradient = Gradient::default();
            }
            _ => (),
        }
    }

    /// set the length of the knot vector to `knot_length` by linearly interpolating between the first and last knot.
    /// calculates a new set of control points using least squares regression over any and all cached activations. Clears the cache after use.
    /// # Errors
    /// * returns an [`EdgeError::NansInControlPoints`] if the calculated control points contain `NaN` values
    pub(super) fn set_knot_length(&mut self, new_knot_length: usize) -> Result<(), EdgeError> {
        match &mut self.kind {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                activations,
                gradients,
                residual_weight,
                residual_gradient,
            } => {
                let degree = *degree;
                let new_knots = linspace(knots[0], knots[knots.len() - 1], new_knot_length);
                // build regressor matrix
                let inputs = linspace(
                    knots[0],
                    knots[knots.len() - 1],
                    100.max(knots.len() * degree * 2),
                );
                let copy_edge = Edge {
                    // dummy edge to infer outputs, to play nice with the borrow checker
                    kind: EdgeType::Spline {
                        degree,
                        control_points: control_points.clone(),
                        knots: knots.clone(),
                        residual_weight: *residual_weight,
                        activations: vec![], // never gets used
                        gradients: vec![],   // never gets used
                        residual_gradient: Gradient::default(),
                    },
                    last_t: vec![],
                    l1_norm: None,
                };
                let target_outputs = copy_edge.infer(&inputs);

                let new_control_point_len = new_knots.len() - degree - 1;
                let target_matrix = DVector::from_vec(target_outputs);
                let regressor_matrix =
                    DMatrix::from_fn(inputs.len(), new_control_point_len, |i, j| {
                        basis_no_cache(j, degree, inputs[i], &new_knots)
                    });

                // solve the least squares problem
                let xtx = regressor_matrix.tr_mul(&regressor_matrix);
                assert_eq!(xtx.nrows(), xtx.ncols());
                let xty = regressor_matrix.tr_mul(&target_matrix);
                let svd = SVD::new(xtx, true, true);
                let solution = svd.solve(&xty, 1e-6).expect("SVD solve failed");
                // check new control points for errors
                let new_control_points: Vec<f64> = solution.iter().map(|v| *v).collect();
                if new_control_points.iter().any(|c| c.is_nan()) {
                    return Err(EdgeError::NansInControlPoints {
                        offending_spline: self.clone(),
                    });
                }
                // update parameters
                *control_points = new_control_points;
                *knots = new_knots;
                // reset state
                *activations =
                    vec![vec![FxHashMap::default(); new_control_point_len + degree]; degree];
                *gradients = vec![Gradient::default(); control_points.len()];
                *residual_gradient = Gradient::default();
                Ok(())
            }
            _ => Ok(()), // setting the knot length on a non-spline edge is a no-op
        }
    }

    // copying pykan for now. TODO: think more about this
    const PARAM_MIN: f64 = -10.0;
    const PARAM_MAX: f64 = 10.0;
    const PARAM_STEPS: usize = 21;
    const PARAM_ITERATIONS: usize = 5;

    /// Find symbolic functions that best fit the spline over the given input data. Return the `num_suggestions` best fits, along with their coefficients of determination (R^2)
    ///
    /// IMPORTANT NOTE: despite my wishes, this function does an unreliable job at suggesting constant functions. I'm not going to make constant function a class, because I'm just going to add a bias node at some point
    pub(super) fn suggest_symbolic(&self, num_suggestions: usize) -> Vec<(Edge, f64)> {
        // if the edge is pruned or symbolic, don't suggest anything
        if matches!(&self.kind, EdgeType::Pruned) || matches!(&self.kind, EdgeType::Symbolic { .. })
        {
            return vec![];
        }
        trace!("suggesting symbolic functions for spline {}", self);
        let (degree, knots) = match &self.kind {
            EdgeType::Spline { degree, knots, .. } => (degree, knots),
            _ => unreachable!(),
        };
        let inputs = linspace(knots[*degree], knots[knots.len() - degree - 1], 100);
        let expected_outputs: Vec<f64> = self.infer(&inputs);
        trace!("inputs: {:?}", inputs);
        trace!("expected_outputs: {:?}", expected_outputs);
        // iterate over all possible symbolic functions
        let mut best_functions = thread::scope(|s| {
            let mut handles: Vec<thread::ScopedJoinHandle<(Edge, f64)>> = vec![];
            let mut best_functions: Vec<(Edge, f64)> = vec![];
            for edge_type in SymbolicFunction::iter() {
                trace!("trying symbolic function {:?}", edge_type);
                match edge_type {
                    SymbolicFunction::Linear => {
                        let x_matrix = DMatrix::from_fn(inputs.len(), 2, |i, j| {
                            // add a constant column so we can calculate the intercept aka b
                            if j == 0 {
                                inputs[i]
                            } else {
                                1.0
                            }
                        });
                        let y_matrix = DVector::from_vec(expected_outputs.to_vec());
                        let xtx = x_matrix.tr_mul(&x_matrix);
                        let xty = x_matrix.tr_mul(&y_matrix);
                        let svd = SVD::new(xtx, true, true);
                        let solution = svd.solve(&xty, 1e-6).expect("SVD solve failed");
                        let best_linear_edge = Edge {
                            kind: EdgeType::Symbolic {
                                a: solution[0],
                                b: solution[1],
                                c: 1.0,
                                d: 0.0,
                                function: edge_type,
                            },
                            last_t: vec![],
                            l1_norm: None,
                        };
                        let function_outputs: Vec<f64> = best_linear_edge.infer(&inputs);
                        let r2 =
                            calculate_coef_of_determination(&expected_outputs, &function_outputs);
                        best_functions.push((best_linear_edge, r2));
                    }
                    _ => {
                        let edge_type = edge_type.clone();
                        // a bit of clever shadowing to work with the borrow checker
                        let inputs = &inputs;
                        let expected_outputs = &expected_outputs;
                        let handle: thread::ScopedJoinHandle<(Edge, f64)> = s.spawn(move || {
                            let best_edge_of_the_type = Self::parameter_search(
                                edge_type,
                                Self::PARAM_STEPS,
                                Self::PARAM_ITERATIONS,
                                inputs,
                                expected_outputs,
                            );
                            let function_outputs: Vec<f64> = best_edge_of_the_type.infer(&inputs);
                            let r2 = calculate_coef_of_determination(
                                &expected_outputs,
                                &function_outputs,
                            );
                            trace!(
                                "Best edge of type {:?} - R2: {} {}",
                                edge_type,
                                r2,
                                best_edge_of_the_type
                            );
                            return (best_edge_of_the_type, r2);
                        });
                        handles.push(handle);
                    }
                }
            }
            [
                best_functions,
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<(Edge, f64)>>(),
            ]
            .concat()
        });
        assert_ne!(best_functions.len(), 0);
        best_functions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_suggestions = best_functions.len().min(num_suggestions);
        let suggestions = best_functions[0..max_suggestions].to_vec();
        trace!("fitting results: {:#?}", suggestions);
        return suggestions;
    }

    fn parameter_search(
        kind: SymbolicFunction,
        step_count: usize,
        iterations: usize,
        inputs: &[f64],
        expected_outputs: &[f64],
    ) -> Edge {
        trace!(
            "searching for best parameters for symbolic function {:?}",
            kind,
        );
        let mut best_edge = Edge {
            kind: EdgeType::Symbolic {
                a: 0.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                function: kind,
            },
            last_t: vec![],
            l1_norm: None,
        }; // arbitrary initial value
        let mut a_min = Self::PARAM_MIN;
        let mut a_max = Self::PARAM_MAX;
        let mut b_min = Self::PARAM_MIN;
        let mut b_max = Self::PARAM_MAX;
        let mut c_min = Self::PARAM_MIN;
        let mut c_max = Self::PARAM_MAX;
        for it in 1..=iterations {
            let mut iteration_best_r2 = f64::NEG_INFINITY;
            let mut iteration_best_a_idx = 0;
            let mut iteration_best_b_idx = 0;
            let mut iteration_best_c_idx = 0;

            let a_values = linspace(a_min, a_max, step_count);
            let b_values = linspace(b_min, b_max, step_count);
            let c_values = linspace(c_min, c_max, step_count);

            for i in (0..step_count).rev() {
                // go in reverse to favor positive values of a in functions where positive and negative values are equivalent (e.g x^2)
                for j in (0..step_count).rev() {
                    for k in 0..step_count {
                        let function_under_test = Edge {
                            kind: EdgeType::Symbolic {
                                a: a_values[i],
                                b: b_values[j],
                                c: c_values[k],
                                d: 0.0,
                                function: kind,
                            },
                            last_t: vec![],
                            l1_norm: None,
                        };
                        let function_outputs: Vec<f64> = function_under_test.infer(inputs);
                        let r2 =
                            calculate_coef_of_determination(expected_outputs, &function_outputs);
                        if r2 > iteration_best_r2 {
                            iteration_best_r2 = r2;
                            iteration_best_a_idx = i;
                            iteration_best_b_idx = j;
                            iteration_best_c_idx = k;
                            best_edge = function_under_test;
                        }
                    }
                }
            }
            trace!(
                "iteration {} a_values: [{}, {}, ... {}] b_values: [{}, {}, ..., {}] c_values: [{}, {}, ..., {}]\nbest function: {} with r2: {}",
                it,
                a_min,
                a_values[1],
                a_max,
                b_min,
                b_values[1],
                b_max,
                c_min,
                c_values[1],
                c_max,
                best_edge,
                iteration_best_r2
            );
            // prepare for next iteration
            a_min = match iteration_best_a_idx {
                0 => a_values[0] - (a_values[1] - a_values[0]),
                _ => a_values[iteration_best_a_idx - 1],
            };
            b_min = match iteration_best_b_idx {
                0 => b_values[0] - (b_values[1] - b_values[0]),
                _ => b_values[iteration_best_b_idx - 1],
            };
            c_min = match iteration_best_c_idx {
                0 => c_values[0] - (c_values[1] - c_values[0]),
                _ => c_values[iteration_best_c_idx - 1],
            };
            let last_val = step_count - 1;
            a_max = if iteration_best_a_idx == last_val {
                a_values[step_count - 1] + (a_values[step_count - 1] - a_values[step_count - 2])
            } else {
                a_values[iteration_best_a_idx + 1]
            };
            b_max = if iteration_best_b_idx == last_val {
                b_values[step_count - 1] + (b_values[step_count - 1] - b_values[step_count - 2])
            } else {
                b_values[iteration_best_b_idx + 1]
            };
            c_max = if iteration_best_c_idx == last_val {
                c_values[step_count - 1] + (c_values[step_count - 1] - c_values[step_count - 2])
            } else {
                c_values[iteration_best_c_idx + 1]
            };
        }

        // now use linear regression to find the best c and d values
        // let x_matrix = DMatrix::from_fn(inputs.len(), 2, |i, j| {
        //     // add a constant column so we can calculate the intercept aka d
        //     if j == 0 {
        //         best_edge.infer(inputs[i])
        //     } else {
        //         1.0
        //     }
        // });
        // let y_matrix = DVector::from_vec(expected_outputs.to_vec());
        // let xtx = x_matrix.tr_mul(&x_matrix);
        // let xty = x_matrix.tr_mul(&y_matrix);
        // let svd = SVD::new(xtx, true, true);
        // let solution = svd.solve(&xty, 1e-6).expect("SVD solve failed");
        // let best_c = solution[0];
        // let best_d = solution[1];
        let best_d = best_edge
            .infer(inputs)
            .iter()
            .zip(expected_outputs.iter())
            .map(|(y_pred, y)| y - y_pred)
            .sum::<f64>()
            / inputs.len() as f64;
        let (best_a, best_b, best_c) = match best_edge.kind {
            EdgeType::Symbolic { a, b, c, .. } => (a, b, c),
            _ => unreachable!(),
        };
        best_edge.kind = EdgeType::Symbolic {
            a: best_a,
            b: best_b,
            c: best_c,
            d: best_d,
            function: kind,
        };

        best_edge
    }

    /// If the average absolute value of the output of the spline over it's input range (defined as the range between the first and last non-padding knot) is less than `threshold`, lock the edge to y=0;
    /// If called on a symbolic edge... do nothing(?)
    /// # Returns
    /// * `true` if the edge was pruned, `false` otherwise
    pub(super) fn prune(&mut self, threshold: f64) -> bool {
        debug!("pruning edge {}", self);
        match &mut self.kind {
            EdgeType::Spline { degree, knots, .. } => {
                let inputs = linspace(
                    knots[*degree],
                    knots[knots.len() - 1 - *degree],
                    100.max(knots.len() * *degree), // make sure we have enough points to get a good estimate.
                );
                let outputs: Vec<f64> = self.infer(&inputs);
                let mean_displacement =
                    outputs.iter().map(|v| v.abs()).sum::<f64>() / outputs.len() as f64;
                debug!(
                    "inputs = {:?}\noutputs = {:?}\nmean_displacement: {}",
                    inputs, outputs, mean_displacement
                );
                if mean_displacement < threshold {
                    self.kind = EdgeType::Pruned;
                    return true;
                }
                return false;
            }
            _ => return false, // trying to prune a non-spline edge is a no-op
        }
    }

    /// return the number of control points and knots in the spline
    pub(super) fn parameter_count(&self) -> usize {
        match &self.kind {
            EdgeType::Spline {
                degree: _,
                control_points,
                knots,
                residual_weight: _,
                activations: _,
                gradients: _,
                residual_gradient: _,
            } => control_points.len() + knots.len() + 1, // plus 1 for residual weight
            EdgeType::Symbolic { .. } => 4, // every symbolic edge has 4 parameters - a, b, c, and d
            EdgeType::Pruned => 0,          // pruned edges have no parameters
        }
    }

    /// return the number of control points in the spline
    pub(super) fn trainable_parameter_count(&self) -> usize {
        match &self.kind {
            EdgeType::Spline {
                degree: _,
                control_points,
                knots: _,
                residual_weight: _,
                activations: _,
                gradients: _,
                residual_gradient: _,
            } => control_points.len() + 1, // plus 1 for residual weight
            EdgeType::Symbolic { .. } => 0, // symbolic edges have no trainable parameters
            EdgeType::Pruned => 0,          // pruned edges have no parameters
        }
    }

    /// merge a slice of splines into a single spline by averaging the control points and knots
    /// # Errors
    /// * returns [`SplineError::MergeNoSplines`] if the input slice is empty
    /// * returns [`SplineError::MergeMismatchedDegree`] if the splines have different degrees
    /// * returns [`SplineError::MergeMismatchedControlPointCount`] if the splines have different numbers of control points
    /// * returns [`SplineError::MergeMismatchedKnotCount`] if the splines have different numbers of knots
    pub(crate) fn merge_edges(edges: Vec<Edge>) -> Result<Edge, EdgeError> {
        if edges.len() == 0 {
            return Err(EdgeError::MergeNoEdges);
        }
        let expected_variant = std::mem::discriminant(&edges[0].kind);
        for idx in 1..edges.len() {
            if std::mem::discriminant(&edges[idx].kind) != expected_variant {
                return Err(EdgeError::MergeMismatchedEdgeTypes {
                    pos: idx,
                    expected: edges[0].kind.clone(),
                    actual: edges[idx].kind.clone(),
                });
            }
        }
        let total_edges = edges.len();
        let mut edge_queue = VecDeque::from(edges);
        match edge_queue
            .pop_front()
            .expect("Edge queue empty even after check")
            .kind
        {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                residual_weight,
                ..
            } => {
                let expected_degree = degree;
                let mut new_control_points = control_points;
                let mut new_knots = knots;
                let mut new_residual_weight = residual_weight;
                let expected_control_point_count = new_control_points.len();
                let expected_knot_count = new_knots.len();
                let mut i = 0;
                while let Some(edge) = edge_queue.pop_front() {
                    i += 1;
                    match edge.kind {
                        EdgeType::Spline {
                            degree,
                            control_points,
                            knots,
                            residual_weight,
                            activations: _,
                            gradients: _,
                            residual_gradient: _,
                        } => {
                            // check for mismatched degrees, control points, and knots
                            if degree != expected_degree {
                                return Err(EdgeError::MergeMismatchedDegree {
                                    pos: i,
                                    expected: expected_degree,
                                    actual: degree,
                                });
                            }
                            if control_points.len() != expected_control_point_count {
                                return Err(EdgeError::MergeMismatchedControlPointCount {
                                    pos: i,
                                    expected: expected_control_point_count,
                                    actual: control_points.len(),
                                });
                            }
                            if knots.len() != expected_knot_count {
                                return Err(EdgeError::MergeMismatchedKnotCount {
                                    pos: i,
                                    expected: expected_knot_count,
                                    actual: knots.len(),
                                });
                            }
                            // merge in the control points and knots
                            for j in 0..expected_control_point_count {
                                new_control_points[j] += control_points[j];
                            }
                            for j in 0..expected_knot_count {
                                new_knots[j] += knots[j];
                            }
                            new_residual_weight += residual_weight;
                        }
                        _ => unreachable!("all edges should be splines"),
                    }
                }
                // divide by the number of edges to get the average
                // doing it inplace like this avoids any allocations that might come with using map
                for j in 0..expected_control_point_count {
                    new_control_points[j] /= total_edges as f64;
                }
                for j in 0..expected_knot_count {
                    new_knots[j] /= total_edges as f64;
                }
                new_residual_weight /= total_edges as f64;
                Ok(Edge::new(
                    expected_degree,
                    new_control_points,
                    new_knots,
                    new_residual_weight,
                )
                .unwrap())
            }
            EdgeType::Symbolic {
                a,
                b,
                c,
                d,
                function,
            } => {
                // symbolic edges aren't trained, but we want to support symbolifying before merging, just to be safe.
                let expected_function = function;
                let mut new_a = a;
                let mut new_b = b;
                let mut new_c = c;
                let mut new_d = d;
                let mut i = 0;
                while let Some(edge) = edge_queue.pop_front() {
                    i += 1;
                    match edge.kind {
                        EdgeType::Symbolic {
                            a,
                            b,
                            c,
                            d,
                            function,
                        } => {
                            // check for mismatched functions
                            if function != expected_function {
                                return Err(EdgeError::MergeMismatchedSymbolicFunctions {
                                    pos: i,
                                    expected: expected_function,
                                    actual: function,
                                });
                            }
                            // merge in the coefficients
                            new_a += a;
                            new_b += b;
                            new_c += c;
                            new_d += d;
                        }
                        _ => unreachable!("all edges should be symbolic"),
                    }
                }
                // divide by the number of edges to get the average
                new_a /= total_edges as f64;
                new_b /= total_edges as f64;
                new_c /= total_edges as f64;
                new_d /= total_edges as f64;
                Ok(Edge {
                    kind: EdgeType::Symbolic {
                        a: new_a,
                        b: new_b,
                        c: new_c,
                        d: new_d,
                        function: expected_function,
                    },
                    last_t: vec![],
                    l1_norm: None,
                })
            }
            EdgeType::Pruned => Ok(Edge {
                kind: EdgeType::Pruned,
                last_t: vec![],
                l1_norm: Some(0.0),
            }),
        }
    }

    pub(super) fn get_full_input_range(&self) -> (f64, f64) {
        match &self.kind {
            EdgeType::Spline { knots, .. } => {
                let min = knots[0];
                let max = knots[knots.len() - 1];
                (min, max)
            }
            _ => (f64::NEG_INFINITY, f64::INFINITY), // symbolic edges have unbounded input range
        }
    }
}

/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `t`
/// checks the provided cache for a memoized result before computing it. If the result is not found, it is computed and stored in the cache before being returned.
///
/// Passing the cache into the function rather than having the caller cache the result allows caching the results of recursive calls, which is useful during backproopagation
// since this function neither takes nor returns a Spline struct, it doesn't make sense to have it as a method on the struct, so I'm moving it outside the impl block
// TODO fix caching to not cache past first recursion and to add a no-cache option
// fn b(
//     cache: &mut FxHashMap<(usize, usize, u32), f64>,
//     i: usize,
//     k: usize,
//     knots: &Vec<f64>,
//     t: f64,
// ) -> f64 {
//     if k == 0 {
//         if knots[i] <= t && t < knots[i + 1] {
//             return 1.0;
//         } else {
//             return 0.0;
//         }
//     } else {
//         if let Some(cached_result) = cache.get(&(i, k, t.to_bits())) {
//             return *cached_result;
//         }
//         let left = (t - knots[i]) / (knots[i + k] - knots[i]);
//         let right = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
//         let result = left * b(cache, i, k - 1, knots, t) + right * b(cache, i + 1, k - 1, knots, t);
//         cache.insert((i, k, t.to_bits()), result);
//         return result;
//     }
// }

/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `t`
/// checks the provided cache for a memoized result before computing it. If the result is not found, it is computed and stored in the cache before being returned.
/// Only the initial call and the first recursion are cached. Any further recursions are not cached, and basis_no_cache is called instead.
///
/// These functions need to be outside the impl block because they need to borrow the cache mutably, which would conflict with the borrow of self used to iterate over the coefficients
fn basis_cached(
    i: usize,
    k: usize,
    t: f64,
    knots: &[f64],
    cache: &mut [Vec<FxHashMap<u64, f64>>],
    degree: usize,
) -> f64 {
    if k == 0 {
        if knots[i] <= t && t < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    // only cache the resuts of the initial call and the first recursion
    if k > degree - 2 {
        if let Some(cached_result) = cache[degree - k][i].get(&t.to_bits()) {
            return *cached_result;
        }
        let left_coefficient = (t - knots[i]) / (knots[i + k] - knots[i]);
        let right_coefficient = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
        let left_val = basis_cached(i, k - 1, t, knots, cache, degree);
        let right_val = basis_cached(i + 1, k - 1, t, knots, cache, degree);
        let result = left_coefficient * left_val + right_coefficient * right_val;
        cache[degree - k][i].insert(t.to_bits(), result);
        return result;
    }
    let left_coefficient = (t - knots[i]) / (knots[i + k] - knots[i]);
    let right_coefficient = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
    let result = left_coefficient * basis_no_cache(i, k - 1, t, knots)
        + right_coefficient * basis_no_cache(i + 1, k - 1, t, knots);
    return result;
}

/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `t`
/// These functions need to be outside the impl block because they need to borrow the cache mutably, which would conflict with the borrow of self used to iterate over the coefficients
fn basis_no_cache(i: usize, k: usize, t: f64, knots: &[f64]) -> f64 {
    if k == 0 {
        if knots[i] <= t && t < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    let left_coefficient = (t - knots[i]) / (knots[i + k] - knots[i]);
    let right_coefficient = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
    let result = left_coefficient * basis_no_cache(i, k - 1, t, knots)
        + right_coefficient * basis_no_cache(i + 1, k - 1, t, knots);
    return result;
}

fn residual_function(t: f64) -> f64 {
    let numerator = t;
    let denominator = (-1.0 * t).exp() + 1.0;
    numerator / denominator
}

fn drt_residual_wrt_t(t: f64) -> f64 {
    if t >= 20.0 {
        1.0 // for t >= 20, the true value is incredibly close to 1, and attempting to calculate the true value results in overflow
    } else {
        t.exp() * (t + t.exp() + 1.0) / (t.exp() + 1.0).powi(2)
    }
}

/// generate `num` values evenly spaced between `min` and `max` inclusive
pub(crate) fn linspace(min: f64, max: f64, num: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(num);
    let num_intervals = num - 1;
    let step_size = (max - min) / (num_intervals) as f64;
    for i in 0..num_intervals {
        knots.push(min + i as f64 * step_size);
    }
    knots.push(max);
    knots
}

fn calculate_coef_of_determination(expected: &[f64], actual: &[f64]) -> f64 {
    let mean_expected = expected.iter().sum::<f64>() / expected.len() as f64;
    let ss_res = expected
        .iter()
        .zip(actual.iter())
        .map(|(e, a)| (e - a).powi(2))
        .sum::<f64>();
    let ss_tot = expected
        .iter()
        .map(|e| (e - mean_expected).powi(2))
        .sum::<f64>();
    1.0 - (ss_res / (ss_tot + f64::EPSILON))
}

impl std::fmt::Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            EdgeType::Spline { degree, knots, .. } => {
                write!(f, "Spline(k: {}, |knots|: {})", degree, knots.len())
            }
            EdgeType::Symbolic {
                a,
                b,
                c,
                d,
                function,
            } => {
                let type_string = format!("{:?}", function);
                match function {
                    SymbolicFunction::Linear => {
                        write!(f, "{}: {} * ( {} * x + {}) + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Quadratic => {
                        write!(f, "{}: {} * ({} * x + {})^2 + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Cubic => {
                        write!(f, "{}: {} * ({} * x + {})^3 + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Quartic => {
                        write!(f, "{}: {} * ({} * x + {})^4 + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Quintic => {
                        write!(f, "{}: {} * ({} * x + {})^5 + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::SquareRoot => {
                        write!(
                            f,
                            "{}: {} * sqrt({} * x + {}) + {}",
                            type_string, c, a, b, d
                        )
                    }
                    SymbolicFunction::CubeRoot => {
                        write!(
                            f,
                            "{}: {} * cbrt({} * x + {}) + {}",
                            type_string, c, a, b, d
                        )
                    }
                    SymbolicFunction::FourthRoot => {
                        write!(
                            f,
                            "{}: {} * ({} * x + {})^(1/4)) + {}",
                            type_string, c, a, b, d
                        )
                    }
                    SymbolicFunction::FifthRoot => {
                        write!(
                            f,
                            "{}: {} * ({} * x + {})^(1/5) + {}",
                            type_string, c, a, b, d
                        )
                    }
                    SymbolicFunction::Log => {
                        write!(f, "{}: {} * log({} * x + {}) + {}", type_string, c, a, b, d)
                    }

                    SymbolicFunction::Exp => {
                        write!(f, "{}: {} * e^({} * x + {}) + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Sin => {
                        write!(f, "{}: {} * sin({} * x + {}) + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Tan => {
                        write!(f, "{}: {} * tan({} * x + {}) + {}", type_string, c, a, b, d)
                    }
                    SymbolicFunction::Inverse => {
                        write!(f, "{}: {} / ({} * x + {}) + {}", type_string, c, a, b, d)
                    }
                }
            }
            EdgeType::Pruned => write!(f, "Pruned"),
        }
    }
}

#[cfg(test)]
mod tests {

    use statrs::assert_almost_eq;
    use test_log::test;

    use super::*;

    /// build a test edge with the following parameters:
    /// * `degree`: `3`
    /// * `control_points`: `[0.75, 1.0, 1.6, -1.0]`
    /// * `knots`: `[0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0]`
    /// * `residual_weight`: `1.2`
    fn build_test_edge() -> Edge {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let residual_weight = 1.2;
        Edge::new(3, control_points, knots, residual_weight).unwrap()
    }

    #[test]
    fn test_new_spline_with_too_few_knots() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let result = Edge::new(3, control_points, knots, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_basis_cached() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let expected_results = vec![0.0513, 0.5782, 0.3648, 0.0057];
        let k = 3;
        let t = 0.95;
        for i in 0..4 {
            let result_from_caching_function = basis_cached(
                i,
                k,
                t,
                &knots,
                &mut vec![vec![FxHashMap::default(); knots.len() - 1]; k],
                k,
            );
            let result_from_non_caching_function = basis_no_cache(i, k, t, &knots);
            assert_eq!(
                result_from_caching_function, result_from_non_caching_function,
                "idx {}, caching and non-caching functions should return the same result",
                i
            );
            let rounded_result = (result_from_caching_function * 10000.0).round() / 10000.0; // multiple by 10^4, round, then divide by 10^4, in order to round to 4 decimal places
            assert_eq!(rounded_result, expected_results[i], "i = {}", i);
        }
    }

    #[test]
    fn test_b_2() {
        let knots = vec![-1.0, -0.7143, -0.4286, -0.1429, 0.1429, 0.4286, 0.7143, 1.0];
        let expected_results = vec![0.0208, 0.4792, 0.4792, 0.0208];
        let k = 3;
        let t = 0.0;
        for i in 0..4 {
            let result_from_caching_function = basis_cached(
                i,
                k,
                t,
                &knots,
                &mut vec![vec![FxHashMap::default(); knots.len() - 1]; k],
                k,
            );
            let result_from_non_caching_function = basis_no_cache(i, k, t, &knots);
            assert_eq!(
                result_from_caching_function, result_from_non_caching_function,
                "idx {}, caching and non-caching functions should return the same result",
                i
            );
            let rounded_result = (result_from_caching_function * 10000.0).round() / 10000.0; // multiple by 10^4, round, then divide by 10^4, in order to round to 4 decimal places
            assert_eq!(rounded_result, expected_results[i], "i = {}", i);
        }
    }

    #[test]
    fn test_residual_function() {
        let t = 0.95;
        let residual_activation = residual_function(t);
        let expected_residual_activation = 0.685059;
        assert_almost_eq!(residual_activation, expected_residual_activation, 1e-5);
    }

    #[test]
    fn test_residual_drt() {
        let t = 0.95;
        let drt_residual = drt_residual_wrt_t(t);
        let expected_drt_residual = 0.912168;
        assert_almost_eq!(drt_residual, expected_drt_residual, 1e-5);
    }

    #[test]
    fn test_forward_and_infer_wo_residual() {
        let mut spline = build_test_edge();
        // wipe the residual weight so we can test without it
        match &mut spline.kind {
            EdgeType::Spline {
                residual_weight, ..
            } => {
                *residual_weight = 0.0;
            }
            _ => unreachable!(),
        }
        let t = vec![0.95];
        let forward_result = spline.forward(&t)[0];
        let infer_result = spline.infer(&t)[0];
        assert_eq!(
            forward_result, infer_result,
            "forward and infer should return the same result"
        );
        let expected_spline_result =
            (0.05127 * 0.75) + (0.57818 * 1.0) + (0.36483 * 1.6) + (0.00573 * -1.0);
        assert_almost_eq!(forward_result, expected_spline_result, 1e-4);
    }

    #[test]
    fn test_forward_and_infer_w_residual() {
        let mut spline = build_test_edge();
        let t = 0.95;
        let result = spline.forward(&vec![t]);
        let infer_result = spline.infer(&vec![t]);
        assert_eq!(
            result, infer_result,
            "forward and infer should return the same result"
        );
        let expected_spline_result =
            (0.05127 * 0.75) + (0.57818 * 1.0) + (0.36483 * 1.6) + (0.00573 * -1.0);
        let expected_residual_result = 0.685059 * 1.2;
        let expected_result = expected_residual_result + expected_spline_result;
        assert_almost_eq!(result[0], expected_result, 1e-4);
    }

    #[test]
    fn test_forward_and_infer_2() {
        let k = 3;
        let coef_size = 4;
        let knot_size = coef_size + k + 1;
        let mut knots = vec![0.0; knot_size];
        knots[0] = -1.0;
        for i in 1..knots.len() {
            knots[i] = -1.0 + (i as f64 / (knot_size - 1) as f64 * 2.0);
        }
        let residual_weight = 1.2;
        let mut spline1 = Edge::new(k, vec![1.0; coef_size], knots, residual_weight).unwrap();
        println!("{:#?}", spline1);
        let t = vec![0.0];
        let result = spline1.forward(&t);
        let infer_result = spline1.infer(&t);
        assert_eq!(
            result, infer_result,
            "forward and infer should return the same result"
        );
        println!("{:#?}", spline1);
        let rounded_activation = (result[0] * 10000.0).round() / 10000.0;
        assert_eq!(rounded_activation, 1.0);
    }

    #[test]
    // backward can't be run without forward, so we include forward in the name to make it obvious that if forward fails, backward will also fail
    fn test_forward_then_backward() {
        let mut spline = build_test_edge();
        let t = vec![0.95];
        let _result = spline.forward(&t);
        trace!("post forward {:#?}", spline);
        let error = vec![-0.6];
        let input_gradient = spline.backward(&error, 1.0).unwrap();
        trace!("post backward {:#?}", spline);
        let expected_spline_drt_wrt_input = 1.2290;
        let expected_residual_drt_wrt_input = 0.912168 * 1.2;
        let expedted_control_point_gradients = vec![-0.0077, -0.0867, -0.0547, -0.0009];
        let rounded_control_point_gradients: Vec<f64> = match spline.kind {
            EdgeType::Spline { gradients, .. } => gradients
                .iter()
                .map(|g| (g.prediction_gradient * 10000.0).round() / 10000.0)
                .collect(),
            _ => unreachable!(),
        };
        assert_eq!(
            rounded_control_point_gradients, expedted_control_point_gradients,
            "control point gradients"
        );
        assert_almost_eq!(
            input_gradient[0],
            (expected_spline_drt_wrt_input + expected_residual_drt_wrt_input) * error[0],
            1e-4
        );
    }

    #[test]
    fn test_forward_then_backward_2() {
        let k = 3;
        let coef_size = 4;
        let knot_size = coef_size + k + 1;
        let mut knots = vec![0.0; knot_size];
        knots[0] = -1.0;
        for i in 1..knots.len() {
            knots[i] = -1.0 + (i as f64 / (knot_size - 1) as f64 * 2.0);
        }
        let residual_weight = 1.2;
        let mut spline1 =
            Edge::new(k, vec![1.0; coef_size], knots.clone(), residual_weight).unwrap();

        let t = vec![0.0];
        let activation = spline1.forward(&t);
        assert_almost_eq!(activation[0], 1.0, 1e-4);

        let output_gradient = vec![0.5];
        let input_gradient = spline1.backward(&output_gradient, 1.0).unwrap();
        let expected_residual_drt_wrt_input = 0.5;
        let expected_input_gradient =
            (0.0 + residual_weight * expected_residual_drt_wrt_input) * output_gradient[0];
        assert_almost_eq!(input_gradient[0], expected_input_gradient, 1e-4);
    }

    #[test]
    fn test_backward_before_forward() {
        let mut spline = build_test_edge();
        let error = vec![-0.6];
        let result = spline.backward(&error, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn backward_after_infer() {
        let mut spline = build_test_edge();
        let _ = spline.infer(&vec![0.95]);
        let error = vec![-0.6];
        let result = spline.backward(&error, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_knots() {
        let mut spline = build_test_edge();
        let mut samples = Vec::with_capacity(150);
        // assuming unordered samples for now
        for _ in 0..50 {
            samples.push(3.0)
        }
        for i in 0..100 {
            samples.push(-3.0 + i as f64 * 0.06);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap()); // this is annoying, but f64 DOESN'T IMPLEMENT ORD, so we have to use partial_cmp // this is annoying, but f64 DOESN'T IMPLEMENT ORD, so we have to use partial_cmp)
        println!("{:?}", samples);
        spline.update_knots_from_samples(samples.as_slice(), 1.0);
        let expected_knots = vec![-5.25, -4.5, -3.75, -3.0, 3.0, 3.75, 4.5, 5.25];
        let rounded_knots: Vec<f64> = match spline.kind {
            EdgeType::Spline { knots, .. } => knots
                .iter()
                .map(|k| (k * 10000.0).round() / 10000.0)
                .collect(),
            _ => unreachable!(),
        };
        assert_eq!(rounded_knots, expected_knots);
    }

    #[test]
    fn test_update_knots_from_bad_samples() {
        let knots = linspace(-1.0, 1.0, 10);
        let control_points = vec![1.0; 6];
        let mut spline = Edge::new(3, control_points, knots.clone(), 0.0).unwrap();
        let samples = vec![0.0; 20];
        spline.update_knots_from_samples(&samples, 0.0);
        let current_knots = match spline.kind {
            EdgeType::Spline { knots, .. } => knots,
            _ => unreachable!(),
        };
        assert_eq!(
            current_knots, knots,
            "knots updated when they shouldn't have been"
        );
    }

    #[test]
    fn test_set_knot_length_increasing() {
        let k = 3;
        let coef_size = 5;
        let knot_length = coef_size + k + 1;
        let knots = linspace(-1., 1., knot_length);
        let mut spline = Edge::new(k, vec![1.0; coef_size], knots, 0.0).unwrap();

        let sample_size = 100;
        let inputs = linspace(-1., 1.0, sample_size);
        let expected_outputs = spline.forward(&inputs); // use forward because we want the activations to be memoized

        let new_knot_length = knot_length * 2 - 1; // increase knot length
        spline.set_knot_length(new_knot_length).unwrap();

        let test_outputs = spline.forward(&inputs);

        let rmse = (expected_outputs
            .iter()
            .zip(test_outputs.iter())
            .map(|(e, t)| (e - t).powi(2))
            .sum::<f64>()
            / sample_size as f64)
            .sqrt();
        let control_points = match spline.kind {
            EdgeType::Spline { control_points, .. } => control_points,
            _ => unreachable!(),
        };
        assert_ne!(control_points, vec![0.0; control_points.len()]);
        assert_almost_eq!(rmse as f64, 0., 1e-3);
    }

    #[test]
    fn test_set_knot_length_decreasing() {
        // I don't know when one would do this, but let's make sure it works anyway
        let k = 3;
        let coef_size = 10;
        let knot_length = coef_size + k + 1;
        let knots = linspace(-1., 1., knot_length);
        let mut spline = Edge::new(k, vec![1.0; coef_size], knots, 0.0).unwrap();

        let sample_size = 100;
        let inputs = linspace(-1., 1.0, sample_size);
        let expected_outputs = spline.forward(&inputs); // use forward because we want the activations to be memoized

        let new_knot_length = knot_length - 2; // decrease knot length
        spline.set_knot_length(new_knot_length).unwrap();

        let test_outputs = spline.forward(&inputs);

        let rmse = (expected_outputs
            .iter()
            .zip(test_outputs.iter())
            .map(|(e, t)| (e - t).powi(2))
            .sum::<f64>()
            / sample_size as f64)
            .sqrt();
        assert_almost_eq!(rmse as f64, 0., 1e-1); // it doesn't work as well as the other way around, but it still works
    }

    #[test]
    fn test_merge_splines() {
        let spline1 = Edge::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1.0,
        )
        .unwrap();
        let spline2 = Edge::new(
            3,
            vec![2.0, 3.0, -4.0],
            vec![-1.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0],
            2.0,
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let new_spline = Edge::merge_edges(splines).unwrap();
        let expected_spline = Edge::new(
            3,
            vec![1.5, 2.5, -0.5],
            vec![-0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0],
            1.5,
        )
        .unwrap();
        assert_eq!(new_spline, expected_spline);
    }

    #[test]
    fn test_merge_splines_mismatched_degree() {
        let spline1 = Edge::new(
            2,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            0.0,
        )
        .unwrap();
        let spline2 = Edge::new(
            1,
            vec![2.0, 3.0, -4.0],
            vec![-1.0, 1.0, 2.0, 5.0, 6.0, 7.0],
            0.0,
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let result = Edge::merge_edges(splines);
        assert!(matches!(
            result,
            Err(EdgeError::MergeMismatchedDegree { .. })
        ));
    }

    #[test]
    fn test_merge_splines_mismatched_control_points() {
        let spline1 = Edge::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            0.0,
        )
        .unwrap();
        let spline2 = Edge::new(
            3,
            vec![2.0, 3.0, -4.0, 0.0],
            vec![-1.0, 1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0],
            0.0,
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let result = Edge::merge_edges(splines);
        assert!(matches!(
            result,
            Err(EdgeError::MergeMismatchedControlPointCount { .. })
        ));
    }

    #[test]
    fn test_merge_splines_mismatched_knots() {
        let spline1 = Edge::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            0.0,
        )
        .unwrap();
        let spline2 = Edge::new(
            3,
            vec![2.0, 3.0, -4.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5],
            0.0,
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let result = Edge::merge_edges(splines);
        assert!(matches!(
            result,
            Err(EdgeError::MergeMismatchedKnotCount { .. })
        ));
    }

    #[test]
    fn test_merge_splines_empty_spline() {
        let splines = vec![];
        let result = Edge::merge_edges(splines);
        assert!(matches!(result, Err(EdgeError::MergeNoEdges)));
    }

    #[test]
    fn test_merged_identical_splines_yield_identical_outputs() {
        let mut spline1 = Edge::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1.2,
        )
        .unwrap();
        let mut spline2 = spline1.clone();
        let t = vec![0.5];
        let output1 = spline1.forward(&t);
        let output2 = spline2.forward(&t);
        assert_eq!(output1, output2);
        let mut new_spline = Edge::merge_edges(vec![spline1, spline2]).unwrap();
        let output3 = new_spline.forward(&t);
        assert_eq!(output1, output3);
    }

    #[test]
    fn test_suggest_symbolic_y_equals_x() {
        let spline = Edge::new(3, linspace(1.0, 5.0, 5), linspace(-0.9, 6.3, 9), 0.0).unwrap();
        let suggest_symbolic = spline.suggest_symbolic(1);
        let (edge, r2) = &suggest_symbolic[0];
        assert_almost_eq!(*r2, 1.0, 1e-1);
        assert!(matches!(edge.kind, EdgeType::Symbolic { .. }));
        let symbolic_function = match &edge.kind {
            EdgeType::Symbolic { function, .. } => function,
            _ => unreachable!(),
        };
        assert_eq!(*symbolic_function, SymbolicFunction::Linear);
        let (a, b, c, d) = match &edge.kind {
            EdgeType::Symbolic { a, b, c, d, .. } => (*a, *b, *c, *d),
            _ => unreachable!(),
        };
        println!("{:?}", (a, b, c, d));
        assert_almost_eq!(a * c, 1.0, 2e-1);
        assert_almost_eq!(b + d, 0.0, 1e-1);
    }

    #[test]
    fn test_suggest_symbolic_quadratic() {
        let spline = Edge::new(
            3,
            vec![
                5.461754152326189,
                14.684164047901085,
                9.729434405954665,
                7.109876768689976,
                5.192089279141963,
                2.785392198709862,
                1.8614584066574362,
                0.3800260937192396,
                0.26549085192647953,
                -0.24376372776114547,
                0.4081693409078997,
                0.9751627408644358,
                2.258028105391018,
                4.037848705116066,
                5.842017427255563,
                8.801872078915483,
                11.566190301734391,
                14.336448294257394,
                21.865717624924052,
                6.107422827686103,
            ],
            vec![
                -0.35779061274722074,
                -0.23367091216715832,
                -0.10955121158709591,
                0.014568488992966491,
                0.18834919418793442,
                0.3627098436213202,
                0.5363930923826065,
                0.7093561619401689,
                0.8818964274635398,
                1.0555924300320363,
                1.234078094752721,
                1.4077823191315206,
                1.5850961016625251,
                1.7600507845515514,
                1.9350333095682102,
                2.1099749997650497,
                2.284256032319133,
                2.4587533550132807,
                2.6327935177528623,
                2.807766326092836,
                2.9934413029144644,
                3.1175610034945267,
                3.241680704074589,
                3.3658004046546517,
            ],
            0.0,
        )
        .unwrap();
        let inputs = linspace(0., 2., 30);
        let suggest_symbolic = spline.suggest_symbolic(1);
        let (edge, r2) = &suggest_symbolic[0];
        println!("R2: {:?}\n{}", r2, edge);
        assert_almost_eq!(*r2, 1.0, 1e-2);
        let final_outputs = edge.infer(&inputs);
        println!("{:?}", final_outputs);
        assert!(matches!(edge.kind, EdgeType::Symbolic { .. }));
        let symbolic_function = match &edge.kind {
            EdgeType::Symbolic { function, .. } => function,
            _ => unreachable!(),
        };
        assert_eq!(*symbolic_function, SymbolicFunction::Quadratic);
        // we can't count on the exact values. As long as the type is correct and R2 sufficiently high, we're good
        // let params = match &edge.kind {
        //     EdgeType::Symbolic { a, b, c, d, .. } => (*a, *b, *c, *d),
        //     _ => unreachable!(),
        // };
        // assert_eq!((2.2, -3.0, 1.5, 0.0), params);
    }

    #[test]
    fn prune_dead_edge() {
        let mut spline = Edge::new(
            3,
            vec![1e-7, 2e-7, 3e-7],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            0.0,
        )
        .unwrap();
        spline.prune(1e-6);
        assert!(matches!(spline.kind, EdgeType::Pruned));
    }

    #[test]
    fn prune_alive_edge() {
        let mut spline = Edge::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            1.2,
        )
        .unwrap();
        spline.prune(1e-6);
        assert!(matches!(spline.kind, EdgeType::Spline { .. }));
    }

    mod symbolic_tests {
        use super::*;
        use test_log::test;

        #[test]
        fn test_symbolic_backward_before_forward() {
            let mut edge = Edge {
                kind: EdgeType::Symbolic {
                    a: 1.0,
                    b: 0.0,
                    c: 1.0,
                    d: 0.0,
                    function: SymbolicFunction::Linear,
                },
                last_t: vec![],
                l1_norm: None,
            };
            let result = edge.backward(&vec![0.5], 1.0);
            assert!(result.is_err());
        }

        #[test]
        fn test_linear() {
            let mut edge = Edge {
                kind: EdgeType::Symbolic {
                    a: 2.0,
                    b: 3.0,
                    c: 4.0,
                    d: 5.0,
                    function: SymbolicFunction::Linear,
                },
                last_t: vec![],
                l1_norm: None,
            };
            let result = edge.forward(&vec![0.5]);
            assert_eq!(result[0], 21.0, "forward");
            let backward = edge.backward(&vec![-0.5], 1.0).unwrap();
            assert_eq!(backward[0], -4.0, "backward");
        }

        #[test]
        fn test_quadratic() {
            let mut edge = Edge {
                kind: EdgeType::Symbolic {
                    a: 1.5,
                    b: 2.0,
                    c: 3.0,
                    d: 7.0,
                    function: SymbolicFunction::Quadratic,
                },
                last_t: vec![],
                l1_norm: None,
            };
            let result = edge.forward(&vec![2.0]);
            assert_eq!(82.0, result[0], "forward");
            let gradient = edge.backward(&vec![0.7], 1.0).unwrap();
            let expected_gradient = 31.5; // (d/dx c(ax + b)^2 + d) * gradient
            assert_almost_eq!(gradient[0], expected_gradient, 1e-6);
        }

        #[test]
        fn test_cubic() {
            let mut edge = Edge {
                kind: EdgeType::Symbolic {
                    a: 1.5,
                    b: 2.0,
                    c: 3.0,
                    d: 7.0,
                    function: SymbolicFunction::Cubic,
                },
                last_t: vec![],
                l1_norm: None,
            };
            let result = edge.forward(&vec![2.0]);
            let expected_result = 3.0 * ((1.5 * 2.0 + 2.0) as f64).powf(3.0) + 7.0;
            assert_almost_eq!(result[0], expected_result, 1e-6);
            let gradient = edge.backward(&vec![0.7], 1.0).unwrap();
            let expected_gradient = 236.25; // (d/dx c(ax + b)^3 + d) * gradient
            assert_almost_eq!(gradient[0], expected_gradient, 1e-6);
        }

        #[test]
        fn test_quartic() {
            let mut edge = Edge {
                kind: EdgeType::Symbolic {
                    a: 1.5,
                    b: 2.0,
                    c: 3.0,
                    d: 7.0,
                    function: SymbolicFunction::Quartic,
                },
                last_t: vec![],
                l1_norm: None,
            };
            let result = edge.forward(&vec![2.0]);
            let expected_result = 1882.0; // 3.0 * ((1.5 * 2.0 + 2.0) as f64).powf(4.0) + 7.0;
            assert_almost_eq!(result[0], expected_result, 1e-6);
            let gradient = edge.backward(&vec![0.7], 1.0).unwrap();
            let expected_gradient = 1575.0; // (d/dx c(ax + b)^3 + d) * gradient
            assert_almost_eq!(gradient[0], expected_gradient, 1e-6);
        }

        #[test]
        fn test_quintic() {
            let mut edge = Edge {
                kind: EdgeType::Symbolic {
                    a: 1.5,
                    b: 2.0,
                    c: 3.0,
                    d: 7.0,
                    function: SymbolicFunction::Quintic,
                },
                last_t: vec![],
                l1_norm: None,
            };
            let result = edge.forward(&vec![0.5]);
            let expected_result = 478.8291015625; //3.0 * ((1.5 * 0.5 + 2.0) as f64).powf(5.0) + 7.0;
            assert_almost_eq!(result[0], expected_result, 1e-6);
            let gradient = edge.backward(&vec![0.7], 1.0).unwrap();
            let expected_gradient = 900.7646484375; // (d/dx c(ax + b)^3 + d) * gradient
            assert_almost_eq!(gradient[0], expected_gradient, 1e-6);
        }
    }
    #[test]
    fn test_spline_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Edge>();
    }

    #[test]
    fn test_spline_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Edge>();
    }
}
