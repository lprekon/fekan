use std::slice::Iter;

#[derive(Debug)]
pub(super) struct Spline {
    degree: usize,
    control_points: Vec<f32>,
    knots: Vec<f32>,
    /// the activations of the spline at each interval, memoized from the most recent forward pass
    activations: Vec<f32>,
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
            activations: vec![0.0; size],
            gradients: vec![0.0; size],
        })
    }

    /// construct a new spline from the given degree, control points, and inner knots, padding the inner knots by
    /// prepending and appending at least `degree` knots of the first and last inner knot respectively
    ///
    /// if the total number of knots after padding is less than `|control_points| + degree + 1`, additional knots are added
    pub(super) fn new_from_inner_knots(
        degree: usize,
        control_points: Vec<f32>,
        inner_knots: Vec<f32>,
    ) -> Self {
        let starting_knot = inner_knots[0];
        let ending_knot = inner_knots[inner_knots.len() - 1];
        // ensure that there are at least `degree` knots at the beginning and end of the knot vector, but also add more than `degree` knots if necessary to ensure that the knot vector is at least `|control_points| + degree + 1
        let knots_to_add_per_end = if inner_knots.len() >= control_points.len() + 1 - degree {
            degree
        } else {
            ((control_points.len() + degree + 1) - inner_knots.len()) / 2
        };

        let mut knots = inner_knots;
        for _ in 0..knots_to_add_per_end {
            knots.insert(0, starting_knot);
            knots.push(ending_knot);
        }
        Spline::new(degree, control_points, knots).unwrap()
    }

    /// compute the point on the spline at the given parameter `t`
    ///
    /// accumulate the activations of the spline at each interval in the internal `activations` field
    pub fn forward(&mut self, t: f32) -> f32 {
        for i in 0..self.control_points.len() {
            self.activations[i] = Spline::b(i, self.degree, &self.knots, t)
        }

        self.activations
            .iter()
            .zip(self.control_points.iter())
            .fold(0.0, |acc, (a, c)| {
                print!("multiplying activation {a} with control point {c}\n");
                acc + a * c
            })
    }

    /// compute the gradients for each control point  on the spline and accumulate them internally.
    ///
    /// returns the gradient of the input used in the forward pass,to be accumulated by the caller and passed back to the pervious layer as its error
    ///
    /// uses the memoized activations from the most recent forward pass
    pub(super) fn backward(&mut self, error: f32) -> f32 {
        let adjusted_error = error / self.control_points.len() as f32; // distribute the error evenly across all control points

        let mut input_gradient = 0.0;
        let k = self.degree;
        for i in 0..self.control_points.len() {
            // calculate control point gradients
            // dC_i = B_ik(t) * adjusted_error
            let t = self.activations[i];
            self.gradients[i] += adjusted_error * t;

            // calculate input gradient
            // dt = sum_i(dB_ik(t) * C_i)
            // dB_ik(t) = (k-1)/(t_i+k-1 - t_i) * B_i(k-1)(t) - (k-1)/(t_i+k - t_i+1) * B_i+1(k-1)(t)
            let left = (k as f32 - 1.0) / (self.knots[i + k - 1] - self.knots[i]);
            let right = (k as f32 - 1.0) / (self.knots[i + k] - self.knots[i + 1]);
            let spline_derivative = left * Spline::b(i, k - 1, &self.knots, t)
                - right * Spline::b(i + 1, k - 1, &self.knots, t);
            input_gradient += spline_derivative * self.control_points[i];
        }

        return input_gradient;
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

    pub(super) fn update_and_zero(&mut self, learning_rate: f32) {
        self.update(learning_rate);
        self.zero_gradients();
    }

    pub(super) fn knots(&self) -> Iter<'_, f32> {
        self.knots.iter()
    }

    pub(super) fn control_points(&self) -> Iter<'_, f32> {
        self.control_points.iter()
    }

    /// recursivly compute the b-spline basis function for the given index `i`, degree `k`, and knot vector, at the given parameter `t`
    // this function is a static method because it needs to recurse down values of 'k', so there's no point in getting the degree from 'self'
    // TODO: memoize this function, since it's called again for the same `i` and `t` in the backward pass. This is apparently difficult to do with floats
    fn b(i: usize, k: usize, knots: &Vec<f32>, t: f32) -> f32 {
        if k == 0 {
            if knots[i] <= t && t < knots[i + 1] {
                return 1.0;
            } else {
                return 0.0;
            }
        } else {
            let left = (t - knots[i]) / (knots[i + k] - knots[i]);
            let right = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]);
            return left * Self::b(i, k - 1, knots, t) + right * Self::b(i + 1, k - 1, knots, t);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    // fn test_b() {
    //     let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    //     assert_eq!(Spline::b(0, 2, &knots, 0.0), 1.0);
    //     assert_eq!(Spline::b(0, 2, &knots, 0.5), 0.5);
    //     assert_eq!(Spline::b(0, 2, &knots, 1.0), 0.0);
    // }

    #[test]
    fn test_new_from_inner_knots_with_plenty_of_knots() {
        let inner_knots = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let my_spline = Spline::new_from_inner_knots(2, vec![1.0, 1.0, 1.0], inner_knots);
        assert_eq!(my_spline.knots.len(), 9);
    }

    #[test]
    fn test_new_from_inner_knots_with_exactly_right_amount_of_knots() {
        let inner_knots = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let my_spline = Spline::new_from_inner_knots(2, vec![1.0; 6], inner_knots);
        assert_eq!(my_spline.knots.len(), 9);
    }

    #[test]
    fn test_new_from_inner_knots_with_not_enough_knots_a() {
        let inner_knots = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let my_spline = Spline::new_from_inner_knots(2, vec![1.0; 10], inner_knots);
        assert_eq!(my_spline.knots.len(), 13);
    }

    #[test]
    fn test_new_from_inner_knots_with_not_enough_knots_b() {
        let inner_knots = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let my_spline = Spline::new_from_inner_knots(3, vec![1.0; 9], inner_knots);
        assert_eq!(my_spline.knots.len(), 13);
    }

    #[test]
    fn test_b() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let expected_results = vec![0.0513, 0.5782, 0.3648, 0.0057];
        let k = 3;
        let t = 0.95;
        for i in 0..4 {
            let result = Spline::b(i, k, &knots, t);
            let rounded_result = (result * 10000.0).round() / 10000.0; // multiple by 10^4, round, then divide by 10^4, in order to round to 4 decimal places
            assert_eq!(rounded_result, expected_results[i], "i = {}", i);
        }
    }

    #[test]
    fn test_forward() {
        let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
        let control_points = vec![0.75, 1.0, 1.6, -1.0];
        let mut spline = Spline::new(3, control_points, knots).unwrap();
        let t = 0.975;
        //0.02535 + 0.5316 + 0.67664 - 0.0117 = 1.22189
        let result = spline.forward(t);
        let rounded_result = (result * 10000.0).round() / 10000.0;
        assert_eq!(rounded_result, 1.2219);
    }
}
