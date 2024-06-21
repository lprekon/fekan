use core::fmt;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::slice::Iter;

/// margin to add to the beginning and end of the knot vector when updating it from samples
pub(super) const KNOT_MARGIN: f32 = 0.01;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Spline {
    degree: usize,
    control_points: Vec<f32>,
    knots: Vec<f32>,
    /// the most recent parameter used in the forward pass
    last_t: Option<f32>,
    /// the activations of the spline at each interval, memoized from the most recent forward pass
    activations: FxHashMap<(usize, usize, u32), f32>,
    /// accumulated gradients for each control point
    gradients: Vec<f32>,
}

impl Spline {
    /// construct a new spline from the given degree, control points, and knots
    ///
    /// # Errors
    /// returns an error if the length of the knot vector is not at least `|control_points| + degree + 1`
    pub(super) fn new(
        degree: usize,
        control_points: Vec<f32>,
        knots: Vec<f32>,
    ) -> Result<Self, SplineError> {
        let size = control_points.len();
        let min_required_knots = size + degree + 1;
        if knots.len() < min_required_knots {
            return Err(SplineError::BadKnotVectorLength {
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
    pub fn forward(&mut self, t: f32) -> f32 {
        self.last_t = Some(t); // store the most recent input for use in the backward pass
        self.control_points
            .iter()
            .enumerate()
            .map(|(idx, coef)| *coef * b(&mut self.activations, idx, self.degree, &self.knots, t))
            .sum()
    }

    /// compute the gradients for each control point  on the spline and accumulate them internally.
    ///
    /// returns the gradient of the input used in the forward pass,to be accumulated by the caller and passed back to the pervious layer as its error
    ///
    /// uses the memoized activations from the most recent forward pass
    ///
    /// # Errors
    /// returns an error if `backward` is called before `forward`
    pub(super) fn backward(&mut self, error: f32) -> Result<f32, SplineError> {
        if let None = self.last_t {
            return Err(SplineError::BackwardBeforeForward);
        }
        let last_t = self.last_t.unwrap();

        let adjusted_error = error / self.control_points.len() as f32; // distribute the error evenly across all control points

        // drt_output_wrt_input = sum_i(dB_ik(t) * C_i)
        let mut drt_output_wrt_input = 0.0;
        let k = self.degree;
        for i in 0..self.control_points.len() {
            // calculate control point gradients
            // dC_i = B_ik(t) * adjusted_error
            let basis_activation = self.activations.get(&(i, k, last_t.to_bits())).unwrap();
            // gradients aka drt_output_wrt_control_point * error
            self.gradients[i] += adjusted_error * basis_activation;

            // calculate the derivative of the spline output with respect to the input (as opposed to wrt the control points)
            // dB_ik(t) = (k-1)/(t_i+k-1 - t_i) * B_i(k-1)(t) - (k-1)/(t_i+k - t_i+1) * B_i+1(k-1)(t)
            let left = (k as f32 - 1.0) / (self.knots[i + k - 1] - self.knots[i]);
            let right = (k as f32 - 1.0) / (self.knots[i + k] - self.knots[i + 1]);
            let left_recurse = b(&mut self.activations, i, k - 1, &self.knots, last_t);
            let right_recurse = b(&mut self.activations, i + 1, k - 1, &self.knots, last_t);
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

    pub(super) fn update(&mut self, learning_rate: f32) {
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
    pub(super) fn knots(&self) -> Iter<'_, f32> {
        self.knots.iter()
    }

    // pub(super) fn control_points(&self) -> Iter<'_, f32> {
    //     self.control_points.iter()
    // }

    /// given a sorted slice of previously seen inputs, update the knot vector to be a linear combination of a uniform vector and a vector of quantiles of the samples.
    ///
    /// If the new knots would contain `degree` or more duplicates - generally caused by too many duplicates in the samples - the knots are not updated
    ///
    /// If `samples` is not sorted, the results of the update and future spline operation are undefined.
    pub(super) fn update_knots_from_samples(&mut self, samples: &[f32], knot_adaptivity: f32) {
        self.activations.clear(); // clear the memoized activations. They're no longer valid, now that the knots are changing

        let knot_size = self.knots.len();
        let mut adaptive_knots: Vec<f32> = Vec::with_capacity(knot_size);
        let num_intervals = self.knots.len() - 1;
        let step_size = samples.len() / (num_intervals);
        for i in 0..num_intervals {
            adaptive_knots.push(samples[i * step_size]);
        }
        adaptive_knots.push(samples[samples.len() - 1]);

        let span_min = samples[0];
        let span_max = samples[samples.len() - 1];
        let uniform_knots = generate_uniform_knots(span_min, span_max, self.knots.len());

        let mut new_knots: Vec<f32> = adaptive_knots
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

    /// return the number of control points and knots in the spline
    pub(super) fn get_parameter_count(&self) -> usize {
        self.control_points.len() + self.knots.len()
    }

    /// return the number of control points in the spline
    pub(super) fn get_trainable_parameter_count(&self) -> usize {
        self.control_points.len()
    }
}

// this is a bit of a heavy duty implementation, but it makes building errors at higher levels and in the future easier

#[derive(Debug, Clone)]
pub(crate) enum SplineError {
    BadKnotVectorLength { expected: usize, actual: usize },
    BackwardBeforeForward,
}

impl fmt::Display for SplineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SplineError::BackwardBeforeForward => write!(f, "backward called before forward"),
            SplineError::BadKnotVectorLength { expected, actual } => write!(
                f,
                "knot vector has length {}, but expected length at least {}",
                actual, expected
            ),
        }
    }
}

impl std::error::Error for SplineError {}

/// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `t`
/// checks the provided cache for a memoized result before computing it. If the result is not found, it is computed and stored in the cache before being returned.
///
/// Passing the cache into the function rather than having the caller cache the result allows caching the results of recursive calls, which is useful during backproopagation
// since this function neither takes nor returns a Spline struct, it doesn't make sense to have it as a method on the struct, so I'm moving it outside the impl block
fn b(
    cache: &mut FxHashMap<(usize, usize, u32), f32>,
    i: usize,
    k: usize,
    knots: &Vec<f32>,
    t: f32,
) -> f32 {
    if k == 0 {
        if knots[i] <= t && t < knots[i + 1] {
            return 1.0;
        } else {
            return 0.0;
        }
    } else {
        if let Some(cached_result) = cache.get(&(i, k, t.to_bits())) {
            return *cached_result;
        }
        let left = (t - knots[i]) / (knots[i + k] - knots[i]);
        let right = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
        let result = left * b(cache, i, k - 1, knots, t) + right * b(cache, i + 1, k - 1, knots, t);
        cache.insert((i, k, t.to_bits()), result);
        return result;
    }
}

/// generate a uniform distribution of `num_knots` values spanning the range from `min` to `max` inclusive
pub(crate) fn generate_uniform_knots(min: f32, max: f32, num_knots: usize) -> Vec<f32> {
    let mut knots = Vec::with_capacity(num_knots);
    let num_intervals = num_knots - 1;
    let step_size = (max - min) / (num_intervals) as f32;
    for i in 0..num_intervals {
        knots.push(min + i as f32 * step_size);
    }
    knots.push(max);
    knots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_spline_with_too_few_knots() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let result = Spline::new(3, control_points, knots);
        assert!(result.is_err());
    }

    #[test]
    fn test_b() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let expected_results = vec![0.0513, 0.5782, 0.3648, 0.0057];
        let k = 3;
        let t = 0.95;
        for i in 0..4 {
            let result = b(&mut FxHashMap::default(), i, k, &knots, t);
            let rounded_result = (result * 10000.0).round() / 10000.0; // multiple by 10^4, round, then divide by 10^4, in order to round to 4 decimal places
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
            let result = b(&mut FxHashMap::default(), i, k, &knots, t);
            let rounded_result = (result * 10000.0).round() / 10000.0; // multiple by 10^4, round, then divide by 10^4, in order to round to 4 decimal places
            assert_eq!(rounded_result, expected_results[i]);
        }
    }

    #[test]
    fn test_forward() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Spline::new(3, control_points, knots).unwrap();
        let t = 0.95;
        //0.02535 + 0.5316 + 0.67664 - 0.0117 = 1.22189
        let result = spline.forward(t);
        let rounded_result = (result * 10000.0).round() / 10000.0;
        assert_eq!(rounded_result, 1.1946);
    }

    #[test]
    fn test_forward_2() {
        let k = 3;
        let coef_size = 4;
        let knot_size = coef_size + k + 1;
        let mut knots = vec![0.0; knot_size];
        knots[0] = -1.0;
        for i in 1..knots.len() {
            knots[i] = -1.0 + (i as f32 / (knot_size - 1) as f32 * 2.0);
        }
        let mut spline1 = Spline::new(k, vec![1.0; coef_size], knots.clone()).unwrap();
        println!("{:#?}", spline1);
        let activation = spline1.forward(0.0);
        println!("{:#?}", spline1);
        let rounded_activation = (activation * 10000.0).round() / 10000.0;
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
        let rounded_control_point_gradients: Vec<f32> = spline
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
            knots[i] = -1.0 + (i as f32 / (knot_size - 1) as f32 * 2.0);
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
            samples.push(-3.0 + i as f32 * 0.06);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap()); // this is annoying, but f32 DOESN'T IMPLEMENT ORD, so we have to use partial_cmp // this is annoying, but f32 DOESN'T IMPLEMENT ORD, so we have to use partial_cmp)
        println!("{:?}", samples);
        spline.update_knots_from_samples(samples.as_slice(), 1.0);
        let mut expected_knots = vec![-3.0, -1.74, -0.48, 0.78, 2.04, 3.0, 3.0, 3.0];
        expected_knots[0] -= KNOT_MARGIN;
        expected_knots[7] += KNOT_MARGIN;
        let rounded_knots: Vec<f32> = spline
            .knots
            .iter()
            .map(|k| (k * 10000.0).round() / 10000.0)
            .collect();
        assert_eq!(rounded_knots, expected_knots);
    }
}
