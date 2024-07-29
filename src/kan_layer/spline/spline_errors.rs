use super::Edge;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EdgeError {
    TooFewKnots {
        expected: usize,
        actual: usize,
    },
    BackwardBeforeForward,
    ActivationsEmpty,
    NansInControlPoints {
        offending_spline: Edge,
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
    MergeNoEdges,
    MergeMismatchedEdgeTypes,
}

impl fmt::Display for EdgeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EdgeError::TooFewKnots { expected, actual } => {
                write!(
                    f,
                    "knot vector has length {}, but expected length at least {} (degree + |control points| + 1)",
                    actual, expected
                )
            }
            EdgeError::BackwardBeforeForward => {
                write!(f, "backward called before forward")
            }
            EdgeError::ActivationsEmpty => {
                write!(f, "activations cache is empty")
            }
            EdgeError::NansInControlPoints{offending_spline} => {
                write!(f, "updated control points contain NaN values - {:?}", offending_spline)
            }
            EdgeError::MergeMismatchedDegree {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "unable to merge splines of different degree. spline at position {} has degree {}, but expected degree {}",
                pos, actual, expected
            ),
            EdgeError::MergeMismatchedControlPointCount {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "unable to merge splines with different numbers of control points. spline at position {} has {} control points, but expected {}",
                pos, actual, expected
            ),
            EdgeError::MergeMismatchedKnotCount {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "unable to merge splines with different numbers of knots. spline at position {} has {} knots, but expected {}",
                pos, actual, expected
            ),
            EdgeError::MergeNoEdges => write!(f, "no splines to merge"),
            EdgeError::MergeMismatchedEdgeTypes => write!(f, "unable to merge splines of different edge types"),
        }
    }
}

impl std::error::Error for EdgeError {}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_spline_error_send() {
        fn assert_send<T: Send>() {}
        assert_send::<EdgeError>();
    }

    #[test]
    fn test_spline_error_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<EdgeError>();
    }
}
