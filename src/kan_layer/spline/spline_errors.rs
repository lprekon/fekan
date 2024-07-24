use super::Spline;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SplineError {
    TooFewKnots {
        expected: usize,
        actual: usize,
    },
    BackwardBeforeForward,
    ActivationsEmpty,
    NansInControlPoints {
        offending_spline: Spline,
    },
    MergeMismatchedDegree {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeMismatchedControlPointCount {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeMismatchedKnotCount {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MergeNoSplines,
}

impl fmt::Display for SplineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SplineError::TooFewKnots { expected, actual } => {
                write!(
                    f,
                    "knot vector has length {}, but expected length at least {} (degree + |control points| + 1)",
                    actual, expected
                )
            }
            SplineError::BackwardBeforeForward => {
                write!(f, "backward called before forward")
            }
            SplineError::ActivationsEmpty => {
                write!(f, "activations cache is empty")
            }
            SplineError::NansInControlPoints{offending_spline} => {
                write!(f, "updated control points contain NaN values - {:?}", offending_spline)
            }
            SplineError::MergeMismatchedDegree {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "unable to merge splines of different degree. spline at position {} has degree {}, but expected degree {}",
                pos, actual, expected
            ),
            SplineError::MergeMismatchedControlPointCount {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "unable to merge splines with different numbers of control points. spline at position {} has {} control points, but expected {}",
                pos, actual, expected
            ),
            SplineError::MergeMismatchedKnotCount {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "unable to merge splines with different numbers of knots. spline at position {} has {} knots, but expected {}",
                pos, actual, expected
            ),
            SplineError::MergeNoSplines => write!(f, "no splines to merge"),
        }
    }
}

impl std::error::Error for SplineError {}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_spline_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SplineError>();
    }

    #[test]
    fn test_spline_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SplineError>();
    }
}
