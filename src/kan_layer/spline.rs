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

use nalgebra::{DMatrix, DVector, SVD};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::slice::Iter;

pub(crate) mod spline_errors;
use spline_errors::*;

/// margin to add to the beginning and end of the knot vector when updating it from samples
pub(super) const KNOT_MARGIN: f64 = 0.01;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Spline {
    // degree, control points, and knots are the parameters of the spline
    // these three fields constitute the "identity" of the spline, so they're the only ones that get serialized, considered for equality, etc.
    degree: usize,
    control_points: Vec<f64>,
    knots: Vec<f64>,

    // the remaining fields represent the "state" of the spline.
    // They're in flux during operation, and so are ignored for any sort of persitence or comparison.
    /// the most recent parameter used in the forward pass
    #[serde(skip)]
    // only used during operation
    last_t: Option<f64>,
    /// the activations of the spline at each interval, stored from calls to [`forward()`](Spline::forward) and cleared on calls to [`update_knots_from_samples()`](Spline::update_knots_from_samples)
    #[serde(skip)] // only used during training
    activations: FxHashMap<(usize, usize, u64), f64>,
    /// accumulated gradients for each control point
    #[serde(skip)] // only used during training
    gradients: Vec<f64>,
}

impl Spline {
    /// construct a new spline from the given degree, control points, and knots
    ///
    /// # Errors
    /// returns an error if the length of the knot vector is not at least `|control_points| + degree + 1`
    pub(super) fn new(
        degree: usize,
        control_points: Vec<f64>,
        knots: Vec<f64>,
    ) -> Result<Self, SplineError> {
        let size = control_points.len();
        let min_required_knots = size + degree + 1;
        if knots.len() < min_required_knots {
            return Err(SplineError::TooFewKnots {
                expected: min_required_knots,
                actual: knots.len(),
            });
        }
        Ok(Spline {
            degree,
            control_points,
            knots,
            last_t: None,
            activations: FxHashMap::default(),
            gradients: vec![0.0; size],
        })
    }

    /// compute the point on the spline at the given parameter `t`
    ///
    /// accumulate the activations of the spline at each interval in the internal `activations` field
    pub fn forward(&mut self, t: f64) -> f64 {
        self.last_t = Some(t); // store the most recent input for use in the backward pass
        let mut sum = 0.0;
        for (idx, coef) in self.control_points.iter().enumerate() {
            let basis_activation = basis_cached(
                idx,
                self.degree,
                t,
                &self.knots,
                &mut self.activations,
                self.degree,
            );
            sum += *coef * basis_activation;
        }
        sum
    }

    /// comput the point on the spline at given parameter `t`
    ///
    /// Does not accumulate the activations of the spline at each interval in the internal `activations` field, or any other internal state
    pub fn infer(&self, t: f64) -> f64 {
        self.control_points
            .iter()
            .enumerate()
            .map(|(idx, coef)| *coef * basis_no_cache(idx, self.degree, t, &self.knots))
            .sum()
    }

    /// compute the gradients for each control point  on the spline and accumulate them internally.
    ///
    /// returns the gradient of the input used in the forward pass,to be accumulated by the caller and passed back to the pervious layer as its error
    ///
    /// uses the memoized activations from the most recent forward pass
    ///
    /// # Errors
    /// * Returns [`SplineError::BackwardBeforeForward`] if called before a forward pass
    pub(super) fn backward(&mut self, error: f64) -> Result<f64, SplineError> {
        if let None = self.last_t {
            return Err(SplineError::BackwardBeforeForward);
        }
        let last_t = self.last_t.unwrap();

        let adjusted_error = error / self.control_points.len() as f64; // distribute the error evenly across all control points

        // drt_output_wrt_input = sum_i(dB_ik(t) * C_i)
        let mut drt_output_wrt_input = 0.0;
        let k = self.degree;
        for i in 0..self.control_points.len() {
            // calculate control point gradients
            // dC_i = B_ik(t) * adjusted_error
            let basis_activation = self.activations.get(&(i, k, last_t.to_bits())).unwrap();
            // gradients aka drt_output_wrt_control_point * error
            let gradient_update = adjusted_error * basis_activation;
            self.gradients[i] += gradient_update;

            // calculate the derivative of the spline output with respect to the input (as opposed to wrt the control points)
            // dB_ik(t) = (k-1)/(t_i+k-1 - t_i) * B_i(k-1)(t) - (k-1)/(t_i+k - t_i+1) * B_i+1(k-1)(t)
            let left = (k as f64 - 1.0) / (self.knots[i + k - 1] - self.knots[i]);
            let right = (k as f64 - 1.0) / (self.knots[i + k] - self.knots[i + 1]);
            let left_recurse = basis_cached(
                i,
                k - 1,
                last_t,
                &self.knots,
                &mut self.activations,
                self.degree,
            );
            let right_recurse = basis_cached(
                i + 1,
                k - 1,
                last_t,
                &self.knots,
                &mut self.activations,
                self.degree,
            );
            // println!(
            //     "i: {} left: {}, right: {}, left_recurse: {}, right_recurse: {}",
            //     i, left, right, left_recurse, right_recurse
            // );
            let basis_derivative = left * left_recurse - right * right_recurse;
            drt_output_wrt_input += self.control_points[i] * basis_derivative;
        }
        // input_gradient = drt_output_wrt_input * error
        return Ok(drt_output_wrt_input * error);
    }

    pub(super) fn update(&mut self, learning_rate: f64) {
        for i in 0..self.control_points.len() {
            self.control_points[i] -= learning_rate * self.gradients[i];
        }
    }

    pub(super) fn zero_gradients(&mut self) {
        for i in 0..self.gradients.len() {
            self.gradients[i] = 0.0;
        }
    }

    #[allow(dead_code)]
    // used in tests for parent module
    pub(super) fn knots<'a>(&'a self) -> Iter<'a, f64> {
        self.knots.iter()
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
        self.activations.clear(); // clear the memoized activations. They're no longer valid, now that the knots are changing

        let knot_size = self.knots.len();
        let mut adaptive_knots: Vec<f64> = Vec::with_capacity(knot_size);
        let num_intervals = self.knots.len() - 1;
        let step_size = samples.len() / (num_intervals);
        for i in 0..num_intervals {
            adaptive_knots.push(samples[i * step_size]);
        }
        adaptive_knots.push(samples[samples.len() - 1]);

        let span_min = samples[0];
        let span_max = samples[samples.len() - 1];
        let uniform_knots = linspace(span_min, span_max, self.knots.len());

        let mut new_knots: Vec<f64> = adaptive_knots
            .iter()
            .zip(uniform_knots.iter())
            .map(|(a, b)| a * knot_adaptivity + b * (1.0 - knot_adaptivity))
            .collect();

        // make sure new_knots doesn't have too many duplicates
        let mut duplicate_count = 0;
        for i in 1..new_knots.len() {
            if new_knots[i] == new_knots[i - 1] {
                duplicate_count += 1;
            } else {
                duplicate_count = 0;
            }
            if duplicate_count >= self.degree {
                return; // we have too many duplicate knots, so we don't update the knots
            }
        }

        new_knots[0] -= KNOT_MARGIN;
        new_knots[knot_size - 1] += KNOT_MARGIN;
        self.knots = new_knots;
    }

    /// set the length of the knot vector to `knot_length` by linearly interpolating between the first and last knot.
    /// calculates a new set of control points using least squares regression over any and all cached activations. Clears the cache after use.
    /// # Errors
    /// * returns [`SplineError::ActivationsEmpty`] if the activations cache is empty. The most likely cause of this is calling `set_knot_length`  after initializing the spline or calling `update_knots_from_samples`, without first calling `forward` at least once.
    /// * returns [`SplineError::NansInControlPoints`] if the calculated control points contain `NaN` values
    pub(super) fn set_knot_length(&mut self, knot_length: usize) -> Result<(), SplineError> {
        let new_knots = linspace(self.knots[0], self.knots[self.knots.len() - 1], knot_length);
        // build regressor matrix
        let mut something = self
            .activations
            .iter()
            .filter(|((_i, k, _t), _b)| *k == self.degree)
            .map(|((i, _k, t), b)| (*i, f64::from_bits(*t), *b))
            .collect::<Vec<_>>();
        if something.is_empty() {
            return Err(SplineError::ActivationsEmpty);
        }
        // put the activations in the correct order. We want each row of the final matrix to be all the basis functions for a single value of t, but since the nalgebra constructors take
        // inputs in column major order, we build the transpose of the matrix we actually want, and sort by 't' first, then 'i'.
        something.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        something.sort_by_key(|(i, _t, _b)| *i);
        /* something = [
            (0, t_0, B_0(t_0)), (0, t_1, B_0(t_1)), ..., (0, t_m, B_0(t_m)),
            (1, t_0, B_1(t_0)), (1, t_1, B_1(t_1)), ..., (1, t_m, B_1(t_m)),
            ...
            (n, t_0, B_n(t_0)), (n, t_1, B_n(t_1)), ..., (n, t_m, B_n(t_m))
        ]
         */
        let num_samples = something.len() / self.control_points.len();
        let new_control_point_len = new_knots.len() - self.degree - 1;
        let activation_matrix =
            DMatrix::from_vec(num_samples, self.control_points.len(), something);
        /* activation_matrix = [
            [(0, t_0, B_0(t_0)), (1, t_0, B_1(t_0)), ..., (n, t_0, B_n(t_0))],
            [(0, t_1, B_0(t_1)), (1, t_1, B_1(t_1)), ..., (n, t_1, B_n(t_1))],
            ...
            [(0, t_m, B_0(t_m)), (1, t_m, B_1(t_m)), ..., (n, t_m, B_n(t_m))]
        ]
         */
        let regressor_matrix = DMatrix::from_fn(num_samples, new_control_point_len, |i, j| {
            let this_t = activation_matrix.row(i)[0].1;
            basis_no_cache(j, self.degree, this_t, &new_knots)
        });

        // build the target matrix by recombining the basis values in the activation matrix with the control points
        let target_matrix = DVector::from_fn(num_samples, |row, _| {
            let row = activation_matrix.row(row);
            row.iter()
                .fold(0.0, |sum, (i, _t, b)| sum + self.control_points[*i] * b)
        });
        // solve the least squares problem
        let xtx = regressor_matrix.tr_mul(&regressor_matrix);
        assert_eq!(xtx.nrows(), xtx.ncols());
        let xty = regressor_matrix.tr_mul(&target_matrix);
        let svd = SVD::new(xtx, true, true);
        let solution = svd.solve(&xty, 1e-6).expect("SVD solve failed");
        // update parameters
        self.control_points = solution.iter().map(|v| *v).collect();
        if self.control_points.iter().any(|c| c.is_nan()) {
            return Err(SplineError::NansInControlPoints {
                offending_spline: self.clone(),
            });
        }
        self.knots = new_knots;
        // reset state
        self.activations.clear();
        self.gradients = vec![0.0; self.control_points.len()];
        Ok(())
    }

    /// return the number of control points and knots in the spline
    pub(super) fn parameter_count(&self) -> usize {
        self.control_points.len() + self.knots.len()
    }

    /// return the number of control points in the spline
    pub(super) fn trainable_parameter_count(&self) -> usize {
        self.control_points.len()
    }

    /// merge a slice of splines into a single spline by averaging the control points and knots
    /// # Errors
    /// * returns [`SplineError::MergeNoSplines`] if the input slice is empty
    /// * returns [`SplineError::MergeMismatchedDegree`] if the splines have different degrees
    /// * returns [`SplineError::MergeMismatchedControlPointCount`] if the splines have different numbers of control points
    /// * returns [`SplineError::MergeMismatchedKnotCount`] if the splines have different numbers of knots
    pub(crate) fn merge_splines(splines: &[Spline]) -> Result<Spline, SplineError> {
        if splines.len() == 0 {
            return Err(SplineError::MergeNoSplines);
        }
        let expected_degree = splines[0].degree;
        let expected_knot_count = splines[0].knots.len();
        let expected_control_point_count = splines[0].control_points.len();
        let num_splines = splines.len();
        let mut control_points = vec![0.0; expected_control_point_count];
        let mut knots = vec![0.0; expected_knot_count];
        for (idx, spline) in splines.iter().enumerate() {
            if spline.degree != expected_degree {
                return Err(SplineError::MergeMismatchedDegree {
                    pos: idx,
                    expected: expected_degree,
                    actual: spline.degree,
                });
            }
            if spline.control_points.len() != expected_control_point_count {
                return Err(SplineError::MergeMismatchedControlPointCount {
                    pos: idx,
                    expected: expected_control_point_count,
                    actual: spline.control_points.len(),
                });
            }
            if spline.knots.len() != expected_knot_count {
                return Err(SplineError::MergeMismatchedKnotCount {
                    pos: idx,
                    expected: expected_knot_count,
                    actual: spline.knots.len(),
                });
            }
            for i in 0..expected_control_point_count {
                control_points[i] += spline.control_points[i];
            }
            for i in 0..expected_knot_count {
                knots[i] += spline.knots[i];
            }
        }
        for i in 0..expected_control_point_count {
            control_points[i] /= num_splines as f64;
        }
        for i in 0..expected_knot_count {
            knots[i] /= num_splines as f64;
        }
        Ok(Spline::new(expected_degree, control_points, knots).unwrap())
    }
}

impl PartialEq for Spline {
    // if two splines have the same degree, control points and knots, they are equivalent, even if they are different instances
    // if one wants to know if two splines are the same instance, one can compare references
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree
            && self.control_points == other.control_points
            && self.knots == other.knots
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
    cache: &mut FxHashMap<(usize, usize, u64), f64>,
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
        if let Some(cached_result) = cache.get(&(i, k, t.to_bits())) {
            return *cached_result;
        }
        let left_coefficient = (t - knots[i]) / (knots[i + k] - knots[i]);
        let right_coefficient = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
        let left_val = basis_cached(i, k - 1, t, knots, cache, degree);
        let right_val = basis_cached(i + 1, k - 1, t, knots, cache, degree);
        let result = left_coefficient * left_val + right_coefficient * right_val;
        cache.insert((i, k, t.to_bits()), result);
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

#[cfg(test)]
mod tests {

    use statrs::assert_almost_eq;

    use super::*;

    #[test]
    fn test_new_spline_with_too_few_knots() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let result = Spline::new(3, control_points, knots);
        assert!(result.is_err());
    }

    #[test]
    fn test_basis_cached() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let expected_results = vec![0.0513, 0.5782, 0.3648, 0.0057];
        let k = 3;
        let t = 0.95;
        for i in 0..4 {
            let result_from_caching_function =
                basis_cached(i, k, t, &knots, &mut FxHashMap::default(), k);
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
            let result_from_caching_function =
                basis_cached(i, k, t, &knots, &mut FxHashMap::default(), k);
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
        let mut spline = Spline::new(3, control_points, knots).unwrap();
        let t = 0.95;
        //0.02535 + 0.5316 + 0.67664 - 0.0117 = 1.22189
        let result = spline.forward(t);
        let infer_result = spline.infer(t);
        assert_eq!(
            result, infer_result,
            "forward and infer should return the same result"
        );
        let rounded_result = (result * 10000.0).round() / 10000.0;
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
        let mut spline1 = Spline::new(k, vec![1.0; coef_size], knots.clone()).unwrap();
        println!("{:#?}", spline1);
        let t: f64 = 0.0;
        let result = spline1.forward(t);
        let infer_result = spline1.infer(t);
        assert_eq!(
            result, infer_result,
            "forward and infer should return the same result"
        );
        println!("{:#?}", spline1);
        let rounded_activation = (result * 10000.0).round() / 10000.0;
        assert_eq!(rounded_activation, 1.0);
    }

    #[test]
    fn test_forward_then_backward() {
        // backward can't be run without forward, so we include forward in the name to make it obvious that if forward fails, backward will also fail
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Spline::new(3, control_points, knots).unwrap();
        let t = 0.95;
        let _result = spline.forward(t);
        let error = -0.6;
        let input_gradient = spline.backward(error).unwrap();
        let expected_spline_drt_wrt_input = 1.2290;
        let expedted_control_point_gradients = vec![-0.0077, -0.0867, -0.0547, -0.0009];
        let rounded_control_point_gradients: Vec<f64> = spline
            .gradients
            .iter()
            .map(|g| (g * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(
            rounded_control_point_gradients, expedted_control_point_gradients,
            "control point gradients"
        );
        let rounded_input_gradient = (input_gradient * 10000.0).round() / 10000.0;
        assert_eq!(
            rounded_input_gradient,
            expected_spline_drt_wrt_input * error
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
        let mut spline1 = Spline::new(k, vec![1.0; coef_size], knots.clone()).unwrap();
        println!("setup: {:#?}", spline1);

        let activation = spline1.forward(0.0);
        println!("forward: {:#?}", spline1);
        let rounded_activation = (activation * 10000.0).round() / 10000.0;
        assert_eq!(rounded_activation, 1.0);

        let input_gradient = spline1.backward(0.5).unwrap();
        println!("backward: {:#?}", spline1);
        let expected_input_gradient = 0.0;
        let rounded_input_gradient = (input_gradient * 10000.0).round() / 10000.0;
        assert_eq!(rounded_input_gradient, expected_input_gradient);
    }

    #[test]
    fn test_backward_before_forward() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Spline::new(3, control_points, knots).unwrap();
        let error = -0.6;
        let result = spline.backward(error);
        assert!(result.is_err());
    }

    #[test]
    fn backward_after_infer() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Spline::new(3, control_points, knots).unwrap();
        let _ = spline.infer(0.95);
        let error = -0.6;
        let result = spline.backward(error);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_knots() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Spline::new(3, control_points, knots).unwrap();
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
        let mut expected_knots = vec![-3.0, -1.74, -0.48, 0.78, 2.04, 3.0, 3.0, 3.0];
        expected_knots[0] -= KNOT_MARGIN;
        expected_knots[7] += KNOT_MARGIN;
        let rounded_knots: Vec<f64> = spline
            .knots
            .iter()
            .map(|k| (k * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_knots, expected_knots);
    }

    #[test]
    fn test_set_knot_length_increasing() {
        let k = 3;
        let coef_size = 5;
        let knot_length = coef_size + k + 1;
        let knots = linspace(-1., 1., knot_length);
        let mut spline = Spline::new(k, vec![1.0; coef_size], knots).unwrap();

        let sample_size = 100;
        let inputs = linspace(-1., 1.0, sample_size);
        let expected_outputs = inputs
            .iter()
            .map(|i| spline.forward(*i))
            .collect::<Vec<f64>>(); // use forward because we want the activations to be memoized

        let new_knot_length = knot_length * 2 - 1; // increase knot length
        spline.set_knot_length(new_knot_length).unwrap();

        let test_outputs = inputs
            .iter()
            .map(|i| spline.forward(*i))
            .collect::<Vec<f64>>();

        let rmse = (expected_outputs
            .iter()
            .zip(test_outputs.iter())
            .map(|(e, t)| (e - t).powi(2))
            .sum::<f64>()
            / sample_size as f64)
            .sqrt();
        assert_ne!(
            spline.control_points,
            vec![0.0; spline.control_points.len()]
        );
        assert_almost_eq!(rmse as f64, 0., 1e-3);
    }

    #[test]
    fn test_set_knot_length_decreasing() {
        // I don't know when one would do this, but let's make sure it works anyway
        let k = 3;
        let coef_size = 10;
        let knot_length = coef_size + k + 1;
        let knots = linspace(-1., 1., knot_length);
        let mut spline = Spline::new(k, vec![1.0; coef_size], knots).unwrap();

        let sample_size = 100;
        let inputs = linspace(-1., 1.0, sample_size);
        let expected_outputs = inputs
            .iter()
            .map(|i| spline.forward(*i))
            .collect::<Vec<f64>>(); // use forward because we want the activations to be memoized

        let new_knot_length = knot_length - 2; // decrease knot length
        spline.set_knot_length(new_knot_length).unwrap();

        let test_outputs = inputs
            .iter()
            .map(|i| spline.forward(*i))
            .collect::<Vec<f64>>();

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
        let spline1 = Spline::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let spline2 = Spline::new(
            3,
            vec![2.0, 3.0, -4.0],
            vec![-1.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let new_spline = Spline::merge_splines(&splines).unwrap();
        let expected_spline = Spline::new(
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
            Spline::new(2, vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let spline2 =
            Spline::new(1, vec![2.0, 3.0, -4.0], vec![-1.0, 1.0, 2.0, 5.0, 6.0, 7.0]).unwrap();
        let splines = vec![spline1, spline2];
        let result = Spline::merge_splines(&splines);
        assert!(matches!(
            result,
            Err(SplineError::MergeMismatchedDegree { .. })
        ));
    }

    #[test]
    fn test_merge_splines_mismatched_control_points() {
        let spline1 = Spline::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let spline2 = Spline::new(
            3,
            vec![2.0, 3.0, -4.0, 0.0],
            vec![-1.0, 1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0],
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let result = Spline::merge_splines(&splines);
        assert!(matches!(
            result,
            Err(SplineError::MergeMismatchedControlPointCount { .. })
        ));
    }

    #[test]
    fn test_merge_splines_mismatched_knots() {
        let spline1 = Spline::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let spline2 = Spline::new(
            3,
            vec![2.0, 3.0, -4.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5],
        )
        .unwrap();
        let splines = vec![spline1, spline2];
        let result = Spline::merge_splines(&splines);
        assert!(matches!(
            result,
            Err(SplineError::MergeMismatchedKnotCount { .. })
        ));
    }

    #[test]
    fn test_merge_splines_empty_spline() {
        let splines = vec![];
        let result = Spline::merge_splines(&splines);
        assert!(matches!(result, Err(SplineError::MergeNoSplines)));
    }

    #[test]
    fn test_merged_identical_splines_yield_identical_outputs() {
        let mut spline1 = Spline::new(
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let mut spline2 = spline1.clone();
        let t = 0.5;
        let output1 = spline1.forward(t);
        let output2 = spline2.forward(t);
        assert_eq!(output1, output2);
        let mut new_spline = Spline::merge_splines(&[spline1, spline2]).unwrap();
        let output3 = new_spline.forward(t);
        assert_eq!(output1, output3);
    }

    #[test]
    fn test_spline_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Spline>();
    }

    #[test]
    fn test_spline_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Spline>();
    }
}
