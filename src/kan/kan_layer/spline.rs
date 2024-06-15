use serde::{Deserialize, Serialize};
use std::{collections::HashMap, slice::Iter};

/// margin to add to the beginning and end of the knot vector when updating it from samples
pub(super) const KNOT_MARGIN: f32 = 0.01;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Spline {
    degree: usize,
    control_points: Vec<f32>,
    knots: Vec<f32>,
    /// the most recent parameter used in the forward pass
    last_t: Option<f32>,
    /// the activations of the spline at each interval, memoized from the most recent forward pass
    activations: HashMap<(usize, usize, u32), f32>,
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
    ) -> Result<Self, String> {
        let size = control_points.len();
        if knots.len() < size + degree + 1 {
            return Err(format!(
                "knot vector has length {}, but expected length at least {}",
                knots.len(),
                size + degree + 1
            ));
        }
        Ok(Spline {
            degree,
            control_points,
            knots,
            last_t: None,
            activations: HashMap::new(),
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
            .map(|(idx, coef)| {
                *coef * Spline::b(&mut self.activations, idx, self.degree, &self.knots, t)
            })
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
    pub(super) fn backward(&mut self, error: f32) -> Result<f32, String> {
        if let None = self.last_t {
            return Err("backward called before forward".to_string());
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
            let left_recurse = Spline::b(&mut self.activations, i, k - 1, &self.knots, last_t);
            let right_recurse = Spline::b(&mut self.activations, i + 1, k - 1, &self.knots, last_t);
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

    /// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `t`
    /// checks the provided cache for a memoized result before computing it. If the result is not found, it is computed and stored in the cache before being returned
    /// This way, both intermediate and final results are cached, and the caches belong to individual splines which can clear them when their knots change
    fn b(
        cache: &mut HashMap<(usize, usize, u32), f32>,
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
            let result = left * Spline::b(cache, i, k - 1, knots, t)
                + right * Spline::b(cache, i + 1, k - 1, knots, t);
            cache.insert((i, k, t.to_bits()), result);
            return result;
        }
    }

    pub(super) fn update_knots_from_samples(
        &mut self,
        mut samples: Vec<f32>,
        knot_adaptivity: f32,
    ) {
        self.activations.clear(); // clear the memoized activations. They're no longer valid, now that the knots are changing
                                  // at some point I'll requure samples to be sorted so we can just reference a slice, but for now we'll take ownership and sort
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap()); // this is annoying, but f32 DOESN'T IMPLEMENT ORD, so we have to use partial_cmp
        let knot_size = self.knots.len();
        let mut adaptive_knots: Vec<f32> = Vec::with_capacity(knot_size);
        let num_intervals = self.knots.len() - 1;
        let step_size = samples.len() / (num_intervals);
        for i in 0..num_intervals {
            adaptive_knots.push(samples[i * step_size]);
        }
        adaptive_knots.push(samples[samples.len() - 1]);
        adaptive_knots[0] -= KNOT_MARGIN;
        adaptive_knots[knot_size - 1] += KNOT_MARGIN;
        let uniform_knots = {
            let min_knot = adaptive_knots[0];
            let max_knot = adaptive_knots[knot_size - 1];
            let step_size = (max_knot - min_knot) / (knot_size - 1) as f32;
            let mut knots = Vec::with_capacity(knot_size);
            for i in 0..knot_size {
                knots.push(min_knot + i as f32 * step_size);
            }
            knots
        };
        self.knots = adaptive_knots
            .iter()
            .zip(uniform_knots.iter())
            .map(|(a, b)| a * knot_adaptivity + b * (1.0 - knot_adaptivity))
            .collect();
    }

    pub(super) fn get_parameter_count(&self) -> usize {
        self.control_points.len() + self.knots.len()
    }
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
            let result = Spline::b(&mut HashMap::new(), i, k, &knots, t);
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
            let result = Spline::b(&mut HashMap::new(), i, k, &knots, t);
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
        println!("{:?}", samples);
        spline.update_knots_from_samples(samples, 1.0);
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
