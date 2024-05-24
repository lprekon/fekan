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
        if knots.len() != size + degree + 1 {
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
        let mut knots = inner_knots;
        let knots_to_add_per_end = max(
            degree,
            (degree + control_points.len() + 1) / 2 - knots.len(),
        );
        for _ in 0..knots_to_add_per_end {
            knots.insert(0, starting_knot);
            knots.push(ending_knot);
        }
        Spline::new(degree, control_points, knots).unwrap()
    }

    /// compute the point on the spline at the given parameter `t`
    ///
    /// accumulate the activations of the spline at each interval in the internal `activations` field
    pub(super) fn forward(&mut self, t: f32) -> f32 {
        for i in 0..self.control_points.len() {
            self.activations[i] = Spline::b(i, self.degree, &self.knots, t)
        }

        self.activations.iter().sum()
    }

    /// compute and accumulate the gradients for each control point on the spline given the error value
    ///
    /// uses the memoized activations from the most recent forward pass
    pub(super) fn backward(&mut self, error: f32) {
        todo!("implement the backward pass for the spline")
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
    fn test_new_from_inner_knots_with_not_enough_knots() {
        let inner_knots = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let my_spline = Spline::new_from_inner_knots(2, vec![1.0; 10], inner_knots);
        assert_eq!(my_spline.knots.len(), 13);
    }
}