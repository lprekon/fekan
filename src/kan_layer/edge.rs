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
use std::{collections::VecDeque, fmt, thread, vec};
use strum::{EnumIter, IntoEnumIterator};

pub(crate) mod edge_errors;
use edge_errors::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Edge {
    kind: EdgeType,
    #[serde(skip)]
    // only used during operation
    last_t: Vec<f64>,
    #[serde(skip)] // only used during training
    l1_norm: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
enum EdgeType {
    Spline {
        // degree, control points, and knots are the parameters of the spline
        // these three fields constitute the "identity" of the spline, so they're the only ones that get serialized, considered for equality, etc.
        degree: usize,
        control_points: Vec<f64>,
        knots: Vec<f64>,

        // the remaining fields represent the "state" of the spline.
        // They're in flux during operation, and so are ignored for any sort of persistence or comparison.
        /// the most recent parameter used in the forward pass

        /// the activations of the spline at each interval, stored from calls to [`forward()`](Spline::forward) and cleared on calls to [`update_knots_from_samples()`](Spline::update_knots_from_samples)
        // the following two fields weren't being serialized because their values represented operating state, not identity. However, they need to be initialized to an appropriate size when deserializing, and manually implementing serde::Deserialize is more trouble than I want right now, so I'm just going to serialize them.
        /// dim0: degree (idx 0 = self.degree, idx 1 = self.degree - 1, etc.), dim1: control point index, dim2: t value
        activations: Vec<Vec<FxHashMap<u64, f64>>>,
        /// accumulated gradients for each control point
        gradients: Vec<Gradient>,
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

impl fmt::Debug for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                ..
            } => f
                .debug_struct("Spline")
                .field("degree", degree)
                .field("control_points", control_points)
                .field("knots", knots)
                .finish(),
            EdgeType::Symbolic {
                a,
                b,
                c,
                d,
                function,
            } => f
                .debug_struct("Symbolic Function")
                .field("function", function)
                .field("a", a)
                .field("b", b)
                .field("c", c)
                .field("d", d)
                .finish(),
            EdgeType::Pruned => {
                write!(f, "Pruned")
            }
        }
    }
}

impl PartialEq for EdgeType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                EdgeType::Spline {
                    degree: d1,
                    control_points: cp1,
                    knots: k1,
                    ..
                },
                EdgeType::Spline {
                    degree: d2,
                    control_points: cp2,
                    knots: k2,
                    ..
                },
            ) => d1 == d2 && cp1 == cp2 && k1 == k2,
            (
                EdgeType::Symbolic {
                    a: a1,
                    b: b1,
                    c: c1,
                    d: d1,
                    function: f1,
                },
                EdgeType::Symbolic {
                    a: a2,
                    b: b2,
                    c: c2,
                    d: d2,
                    function: f2,
                },
            ) => a1 == a2 && b1 == b2 && c1 == c2 && d1 == d2 && f1 == f2,
            (EdgeType::Pruned, EdgeType::Pruned) => true,
            _ => false,
        }
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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
                activations: activations_cache,
                gradients: vec![Gradient::default(); size],
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
                ..
            } => {
                for t in inputs.iter() {
                    let mut sum = 0.0;
                    for (idx, coef) in control_points.iter().enumerate() {
                        let basis_activation =
                            basis_cached(idx, *degree, *t, &knots, activations, *degree);
                        sum += *coef * basis_activation;
                    }
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
                ..
            } => {
                for t in inputs.iter() {
                    let mut sum = 0.0;
                    for (idx, coef) in control_points.iter().enumerate() {
                        let basis_activation = basis_no_cache(idx, *degree, *t, &knots);
                        sum += *coef * basis_activation;
                    }
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
        let edge_l1 = self.l1_norm.expect("edge_l1 is None");
        assert_eq!(layer_l1.signum(), 1.0);
        match &mut self.kind {
            EdgeType::Spline {
                degree,
                control_points,
                knots,
                activations,
                gradients: accumulated_gradients,
            } => {
                // assert_eq!(activations[0][0].len(), edge_gradients.len());

                // drt_output_wrt_input = sum_i(dB_ik(t) * C_i)
                let mut drts_output_wrt_input = vec![0.0; edge_gradients.len()];
                let k = *degree;
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
                    let prediction_gradients = edge_gradients
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
                degree: _,
                control_points,
                knots: _,
                activations: _,
                gradients,
            } => {
                for i in 0..control_points.len() {
                    control_points[i] -= learning_rate
                        * (gradients[i].prediction_gradient
                            + l1_lambda * gradients[i].l1_gradient
                            + entropy_lambda * gradients[i].entropy_gradient);
                }
            }
            _ => (), // update on a non-spline edge is a no-op
        }
    }

    pub(super) fn zero_gradients(&mut self) {
        match &mut self.kind {
            EdgeType::Spline {
                degree: _,
                control_points: _,
                knots: _,
                activations: _,
                gradients,
            } => {
                for i in 0..gradients.len() {
                    gradients[i] = Gradient::default();
                }
            }
            _ => (), // zeroing gradients on a non-spline edge is a no-op
        }
    }

    #[allow(dead_code)]
    // used in tests for parent module
    pub(crate) fn knots<'a>(&'a self) -> &'a [f64] {
        match &self.kind {
            EdgeType::Spline {
                degree: _,
                control_points: _,
                knots,
                activations: _,
                gradients: _,
            } => knots,
            _ => &[],
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
        match &mut self.kind {
            EdgeType::Spline {
                degree,
                control_points: _,
                knots,
                activations,
                gradients: _,
            } => {
                activations
                    .iter_mut()
                    .for_each(|v| v.iter_mut().for_each(|h| h.clear())); // clear the memoized activations. They're no longer valid, now that the knots are changing
                self.last_t.clear(); // clear the last_t cache, since the activations cache is clear
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
                        activations: activations.clone(),
                        gradients: gradients.clone(),
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
    /// Already multithreads - no need for multithreading in the caller
    ///
    ///  IMPORTANT NOTE: despite my wishes, this function does an unreliable job at suggesting constant functions. I'm not going to make constant function a class, because I'm just going to add a bias node at some point
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
    pub(super) fn prune(&mut self, samples: &[f64], threshold: f64) -> bool {
        debug!("pruning edge {}", self);
        match &mut self.kind {
            // this is bad - coefficients that don't see much use don't get trained down.
            EdgeType::Spline { .. } => {
                let outputs: Vec<f64> = self.infer(samples);
                let mean_displacement =
                    outputs.iter().map(|v| v.abs()).sum::<f64>() / outputs.len() as f64;
                debug!(
                    "inputs = {:?}\noutputs = {:?}\nmean_displacement: {}",
                    samples, outputs, mean_displacement
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
                activations: _,
                gradients: _,
            } => control_points.len() + knots.len(),
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
                activations: _,
                gradients: _,
            } => control_points.len(),
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
                ..
            } => {
                let expected_degree = degree;
                let mut new_control_points = control_points;
                let mut new_knots = knots;
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
                            activations: _,
                            gradients: _,
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
                Ok(Edge::new(expected_degree, new_control_points, new_knots).unwrap())
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

/// calculate the basis activation over all i values at once - testing compiler autovectorization
fn basis_autovectorize_across_i(i_vec: &[usize], k: usize, t: f64, knots: &[f64]) -> Vec<f64> {
    if k == 0 {
        let mut result = Vec::with_capacity(i_vec.len());
        for i_val in i_vec {
            if knots[*i_val] <= t && t < knots[*i_val + 1] {
                result.push(1.0);
            } else {
                result.push(0.0);
            }
        }
        return result;
    }
    let left_coefficients: Vec<f64> = i_vec
        .iter()
        .map(|i_val| (t - knots[*i_val]) / (knots[*i_val + k] - knots[*i_val]))
        .collect();
    let right_coefficients: Vec<f64> = i_vec
        .iter()
        .map(|i_val| (knots[*i_val + k + 1] - t) / (knots[*i_val + k + 1] - knots[*i_val + 1]))
        .collect();
    let left_vals = basis_autovectorize_across_i(i_vec, k - 1, t, knots);
    let right_is: Vec<usize> = i_vec.iter().map(|i_val| i_val + 1).collect();
    let right_vals = basis_autovectorize_across_i(&right_is, k - 1, t, knots);

    let left_results = left_coefficients
        .iter()
        .zip(left_vals.iter())
        .map(|(c, v)| c * v);
    let right_results = right_coefficients
        .iter()
        .zip(right_vals.iter())
        .map(|(c, v)| c * v);
    return left_results
        .zip(right_results)
        .map(|(l, r)| l + r)
        .collect();
}

use std::simd::prelude::*;
/// calculate the basis activation over multiple i values at once, using the rust portable SIMD crate
#[inline] // recommended by the crate docs to avoid large function prologues and epilogues, since SIMD values passed/returned in memory and not in registers due to ABI/safety guarantees
fn basis_portable_simd_across_i(i_vec: usizex4, k: usize, t: f64, knots: &[f64]) -> f64x4 {
    let knots_i = Simd::gather_or(knots, i_vec, f64x4::splat(-1.0));
    let knots_i_1 = Simd::gather_or(knots, i_vec + usizex4::splat(1), f64x4::splat(-1.0));
    let t_splat = f64x4::splat(t);
    if k == 0 {
        let left_mask = knots_i.simd_le(t_splat);
        let right_mask = t_splat.simd_lt(knots_i_1);
        let full_mask = left_mask & right_mask;
        trace!("k: {k}, t:{t}\ni_vec: {i_vec:?}\nknots_i: {knots_i:?}\nknots_i_plus_1: {knots_i_1:?}\nleft_mask: {left_mask:?}\nright_mask: {right_mask:?}\nfull_mask: {full_mask:?}\n");
        return full_mask.select(f64x4::splat(1.0), f64x4::splat(0.0));
    }
    let knots_i_k = Simd::gather_or(knots, i_vec + usizex4::splat(k), f64x4::splat(-1.0));
    let knots_i_k_1 = Simd::gather_or(knots, i_vec + usizex4::splat(k + 1), f64x4::splat(-1.0));
    let left_coefficients = (t_splat - knots_i) / (knots_i_k - knots_i);
    let right_coefficients = (knots_i_k_1 - t_splat) / (knots_i_k_1 - knots_i_1);
    let left_vals = basis_portable_simd_across_i(i_vec, k - 1, t, knots);
    let right_vals = basis_portable_simd_across_i(i_vec + usizex4::splat(1), k - 1, t, knots);
    let result = left_coefficients * left_vals + right_coefficients * right_vals;
    trace!("k: {k}, t: {t}\ni_vec: {i_vec:?}\nknots_i: {knots_i:?}\nknots_i_plus_1: {knots_i_1:?}\nknots_i_plus_k_plus_1: {knots_i_k_1:?}\nleft_coefficients: {left_coefficients:?}\nleft_vals: {left_vals:?}\nright_coefficients: {right_coefficients:?}\nright_vals: {right_vals:?}\nresult: {result:?}\n");
    return result;
}

#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    target_feature = "avx2",
))]
/// calculate the basis activation over multiple i values at once, using extended x86 intrinsics. Takes 4 64-bit integers as the `i` values, and returns 4 64-bit floats as the basis values
fn basis_x86_intrinsics_across_i(
    i_vec: std::arch::x86_64::__m256i,
    k: usize,
    t: f64,
    knots: &[f64],
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;
    // unsafe justification: this function is only available when both SSE2 and AVX2 are enabled, which promises the availability of all instructions below
    unsafe {
        let t_splat: __m256d = _mm256_set1_pd(t);
        let knots_i = _mm256_i64gather_pd::<1>(knots.as_ptr(), i_vec);

        let i_1 = _mm256_add_epi64(i_vec, _mm256_set1_epi64x(1));
        let knots_i_1 = _mm256_i64gather_pd::<1>(knots.as_ptr(), i_1);
        if k == 0 {
            let left_mask = _mm256_cmp_pd(knots_i, t_splat, _CMP_LE_OQ);
            let right_mask = _mm256_cmp_pd(t_splat, knots_i_1, _CMP_LT_OQ);
            let full_mask = _mm256_and_pd(left_mask, right_mask);
            return
                // _mm256_mask_blend_pd(full_mask, _mm256_set1_pd(1.0), _mm256_set1_pd(0.0))
                _mm256_and_pd(_mm256_set1_pd(1.0) , full_mask);
        }
        let i_k = _mm256_add_epi64(i_vec, _mm256_set1_epi64x(k.try_into().unwrap()));
        let i_k_1 = _mm256_add_epi32(i_vec, _mm256_set1_epi32((k + 1) as i32));
        let knots_i_k = _mm256_i64gather_pd::<1>(knots.as_ptr(), i_k);
        let knots_i_k_1 = _mm256_i64gather_pd::<1>(knots.as_ptr(), i_k_1);
        let left_numerator = _mm256_sub_pd(t_splat, knots_i);
        let left_denominator = _mm256_sub_pd(knots_i_k, knots_i);
        let left_coefficients = _mm256_div_pd(left_numerator, left_denominator);

        let right_numerator = _mm256_sub_pd(knots_i_k_1, t_splat);
        let right_denominator = _mm256_sub_pd(knots_i_k_1, knots_i_1);
        let right_coefficients = _mm256_div_pd(right_numerator, right_denominator);

        let left_vals = basis_x86_intrinsics_across_i(i_vec, k - 1, t, knots);
        let right_vals = basis_x86_intrinsics_across_i(i_1, k - 1, t, knots);

        let left_results = _mm256_mul_pd(left_coefficients, left_vals);
        let right_results = _mm256_mul_pd(right_coefficients, right_vals);
        let result = _mm256_add_pd(left_results, right_results);
        return result;
    }
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

    #[test]
    fn test_new_spline_with_too_few_knots() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let result = Edge::new(3, control_points, knots);
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
    fn test_forward_and_infer() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Edge::new(3, control_points, knots).unwrap();
        let t = 0.95;
        //0.02535 + 0.5316 + 0.67664 - 0.0117 = 1.22189
        let result = spline.forward(&vec![t]);
        let infer_result = spline.infer(&vec![t]);
        assert_eq!(
            result, infer_result,
            "forward and infer should return the same result"
        );
        let rounded_result = (result[0] * 10000.0).round() / 10000.0;
        assert_eq!(rounded_result, 1.1946);
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
        let mut spline1 = Edge::new(k, vec![1.0; coef_size], knots.clone()).unwrap();
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
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Edge::new(3, control_points, knots).unwrap();
        let t = vec![0.95];
        let _result = spline.forward(&t);
        trace!("post forward {:#?}", spline);
        let error = vec![-0.6];
        let input_gradient = spline.backward(&error, 1.0).unwrap();
        trace!("post backward {:#?}", spline);
        let expected_spline_drt_wrt_input = 1.2290;
        let expedted_control_point_gradients = vec![-0.0308, -0.3469, -0.2189, -0.0034];
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
        let rounded_input_gradient = (input_gradient[0] * 10000.0).round() / 10000.0;
        assert_eq!(
            rounded_input_gradient,
            expected_spline_drt_wrt_input * error[0]
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
        let mut spline1 = Edge::new(k, vec![1.0; coef_size], knots.clone()).unwrap();
        println!("setup: {:#?}", spline1);

        let activation = spline1.forward(&vec![0.0]);
        println!("forward: {:#?}", spline1);
        let rounded_activation = (activation[0] * 10000.0).round() / 10000.0;
        assert_eq!(rounded_activation, 1.0);

        let input_gradient = spline1.backward(&vec![0.5], 1.0).unwrap();
        println!("backward: {:#?}", spline1);
        let expected_input_gradient = 0.0;
        let rounded_input_gradient = (input_gradient[0] * 10000.0).round() / 10000.0;
        assert_eq!(rounded_input_gradient, expected_input_gradient);
    }

    #[test]
    fn test_backward_before_forward() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Edge::new(3, control_points, knots).unwrap();
        let error = vec![-0.6];
        let result = spline.backward(&error, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn backward_after_infer() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Edge::new(3, control_points, knots).unwrap();
        let _ = spline.infer(&vec![0.95]);
        let error = vec![-0.6];
        let result = spline.backward(&error, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_knots() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Edge::new(3, control_points, knots).unwrap();
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
        let mut spline = Edge::new(3, control_points, knots.clone()).unwrap();
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
        let mut spline = Edge::new(k, vec![1.0; coef_size], knots).unwrap();

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
        let mut spline = Edge::new(k, vec![1.0; coef_size], knots).unwrap();

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
        )
        .unwrap();
        let spline2 = Edge::new(
            3,
            vec![2.0, 3.0, -4.0],
            vec![-1.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let new_spline = Edge::merge_edges(splines).unwrap();
        let expected_spline = Edge::new(
            3,
            vec![1.5, 2.5, -0.5],
            vec![-0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        assert_eq!(new_spline, expected_spline);
    }

    #[test]
    fn test_merge_splines_mismatched_degree() {
        let spline1 =
            Edge::new(2, vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let spline2 =
            Edge::new(1, vec![2.0, 3.0, -4.0], vec![-1.0, 1.0, 2.0, 5.0, 6.0, 7.0]).unwrap();
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
        )
        .unwrap();
        let spline2 = Edge::new(
            3,
            vec![2.0, 3.0, -4.0, 0.0],
            vec![-1.0, 1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0],
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
        )
        .unwrap();
        let spline2 = Edge::new(
            3,
            vec![2.0, 3.0, -4.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5],
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

    // removing this test because we shouldn't count on suggest_symbolic to properly match constant functions
    // #[test]
    // fn test_suggest_symbolic_constant_zero() {
    //     let spline = Edge::new(3, vec![0.; 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    //     let inputs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    //     let suggest_symbolic = spline.suggest_symbolic(inputs.as_slice(), 1);
    //     let (edge, r2) = &suggest_symbolic[0];
    //     assert_almost_eq!(*r2, 1.0, 1e-2);
    //     assert!(matches!(edge.kind, EdgeType::Symbolic { .. }));
    //     let symbolic_function = match &edge.kind {
    //         EdgeType::Symbolic { function, .. } => function,
    //         _ => unreachable!(),
    //     };
    //     assert_eq!(*symbolic_function, SymbolicFunction::Linear);
    //     let params = match &edge.kind {
    //         EdgeType::Symbolic { a, b, c, d, .. } => (*a, *b, *c, *d),
    //         _ => unreachable!(),
    //     };
    //     // assert_eq!(*a, 0.0, "a");
    //     assert_eq!(params, (0.0, 0.0, 0.0, 0.0));
    // }

    #[test]
    fn test_suggest_symbolic_y_equals_x() {
        let spline = Edge::new(3, linspace(1.0, 5.0, 5), linspace(-0.9, 6.3, 9)).unwrap();
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
        )
        .unwrap();
        let inputs = linspace(0.5, 5.5, 30);
        spline.prune(&inputs, 1e-6);
        assert!(matches!(spline.kind, EdgeType::Pruned));
    }

    #[test]
    fn prune_alive_edge() {
        let mut spline = Edge::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let inputs = linspace(0.5, 5.5, 30);
        spline.prune(&inputs, 1e-6);
        assert!(matches!(spline.kind, EdgeType::Spline { .. }));
    }

    mod simd {

        use super::*;
        use test_log::test;

        #[test]
        fn test_basis_autovec_i() {
            let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
            let expected_results = vec![0.0513, 0.5782, 0.3648, 0.0057];
            let k = 3;
            let t = 0.95;
            let result = basis_autovectorize_across_i(&[0, 1, 2, 3], k, t, &knots);
            let rounded_result: Vec<f64> = result
                .iter()
                .map(|r| (r * 10000.0).round() / 10000.0)
                .collect();
            assert_eq!(rounded_result, expected_results);
        }

        #[test]
        fn test_basis_portable_i_k0() {
            let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
            let i_vec = usizex4::from_array([0, 1, 2, 3]);
            let t = 0.3;
            let expected_results = f64x4::from_array([0.0, 1.0, 0.0, 0.0]);
            let result = basis_portable_simd_across_i(i_vec, 0, t, &knots);
            assert_eq!(result, expected_results, "knot[1] < t < knot[2]");

            let t = 0.95;
            let result = basis_portable_simd_across_i(i_vec, 0, t, &knots);
            let expected_results = f64x4::from_array([0.0, 0.0, 0.0, 1.0]);
            assert_eq!(result, expected_results, "knot[3] < t < knot[4]");

            let t = 1.5;
            let result = basis_portable_simd_across_i(i_vec, 0, t, &knots);
            let expected_results = f64x4::from_array([0.0, 0.0, 0.0, 0.0]);
            assert_eq!(result, expected_results, "t > knot[5]");

            let t = -0.5;
            let result = basis_portable_simd_across_i(i_vec, 0, t, &knots);
            let expected_results = f64x4::from_array([0.0, 0.0, 0.0, 0.0]);
            assert_eq!(result, expected_results, "t < knot[0]");
        }

        #[test]
        fn test_basis_portable_i_k3() {
            let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
            let k = 3;
            let t = 0.95;
            let expected_results: Vec<f64> =
                (0..4).map(|i| basis_no_cache(i, k, t, &knots)).collect();
            let i_vec = usizex4::from_array([0, 1, 2, 3]);
            let result = basis_portable_simd_across_i(i_vec, k, t, &knots);
            // let rounded_result: Vec<f64> = result
            //     .to_array()
            //     .iter()
            //     .map(|r| (r * 10000.0).round() / 10000.0)
            //     .collect();
            assert_eq!(result.to_array().to_vec(), expected_results);
        }

        #[test]
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "avx512f",
        ))]
        fn test_basis_x86_i_k0() {
            use std::{arch::x86_64::*, mem};
            let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
            let i_slice = [0, 1, 2, 3];
            let i_vec = unsafe { _mm_load_si128(i_slice.as_ptr() as *const __m128i) };
            let t = 0.3;
            let expected_results = vec![0.0, 1.0, 0.0, 0.0];

            let result: __m256d = basis_x86_intrinsics_across_i(i_vec, 0, t, &knots);
            let transmuted_result: [f64; 4] = mem::transmute(result); // the in-memory representation of a __m256d is the same as an array of 4 f64s
            assert_eq!(
                trasmuted_result.to_vec(),
                expected_results,
                "knot[1] < t < knot[2]"
            );

            let t = 0.95;
            let expected_results = vec![0.0, 0.0, 0.0, 1.0];
            let result: __m256d = basis_x86_intrinsics_across_i(i_vec, 0, t, &knots);
            let transmuted_result: [f64; 4] = mem::transmute(result); // the in-memory representation of a __m256d is the same as an array of 4 f64s
            assert_eq!(
                trasmuted_result.to_vec(),
                expected_results,
                "knot[3] < t < knot[4]"
            );

            let t = 1.5;
            let expected_results = vec![0.0, 0.0, 0.0, 0.0];
            let result = basis_x86_intrinsics_across_i(i_vec, 0, t, &knots);
            let transmuted_result: [f64; 4] = mem::transmute(result); // the in-memory representation of a __m256d is the same as an array of 4 f64s
            assert_eq!(trasmuted_result.to_vec(), expected_results, "t > knot[5]");

            let t = -0.5;
            let expected_results = vec![0.0, 0.0, 0.0, 0.0];
            let result = basis_x86_intrinsics_across_i(i_vec, 0, t, &knots);
            let transmuted_result: [f64; 4] = mem::transmute(result); // the in-memory representation of a __m256d is the same as an array of 4 f64s
            assert_eq!(trasmuted_result.to_vec(), expected_results, "t < knot[0]");
        }
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
