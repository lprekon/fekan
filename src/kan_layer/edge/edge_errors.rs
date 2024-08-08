use super::{Edge, EdgeType, SymbolicFunction};
use std::fmt;

#[allow(private_interfaces)] // allow private interfaces for error types, since they just need to be displayed
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
    MergeMismatchedEdgeTypes {
        pos: usize,
        expected: EdgeType,
        actual: EdgeType,
    },
    MergeMismatchedSymbolicFunctions {
        pos: usize,
        expected: SymbolicFunction,
        actual: SymbolicFunction,
    },
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
            EdgeError::MergeMismatchedEdgeTypes{pos, expected, actual} => write!(f, "unable to merge edges of different edge types. edge at position {} has edge type {:?}, but expected edge type {:?}", pos, actual, expected),
            EdgeError::MergeMismatchedSymbolicFunctions{pos, expected, actual} => write!(f, "unable to merge edges with different symbolic functions. edge at position {} has symbolic function {:?}, but expected symbolic function {:?}", pos, actual, expected),
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
