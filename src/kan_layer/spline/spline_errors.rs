use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CreateSplineError {
    TooFewKnotsError { expected: usize, actual: usize },
}

impl fmt::Display for CreateSplineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CreateSplineError::TooFewKnotsError { expected, actual } => {
                write!(
                    f,
                    "knot vector has length {}, but expected length at least {}",
                    actual, expected
                )
            }
        }
    }
}

impl std::error::Error for CreateSplineError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BackwardSplineError {
    BackwardBeforeForwardError,
    ReceivedNanError,
    // GradientIsNanError,
}

impl fmt::Display for BackwardSplineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BackwardSplineError::BackwardBeforeForwardError => {
                write!(f, "backward called before forward")
            }
            BackwardSplineError::ReceivedNanError => write!(f, "received `NaN` as error value"),
            // BackwardSplineError::GradientIsNanError => write!(f, "calculated gradient is NaN"),
        }
    }
}

impl std::error::Error for BackwardSplineError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum UpdateSplineKnotsError {
    ActivationsEmptyError,
    NansInControlPointsError,
}

impl fmt::Display for UpdateSplineKnotsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UpdateSplineKnotsError::ActivationsEmptyError => {
                write!(f, "activations cache is empty")
            }
            UpdateSplineKnotsError::NansInControlPointsError => {
                write!(f, "control points contain NaN values")
            }
        }
    }
}

impl std::error::Error for UpdateSplineKnotsError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MergeSplinesError {
    MismatchedDegreeError {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MismatchedControlPointCountError {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    MismatchedKnotCountError {
        pos: usize,
        expected: usize,
        actual: usize,
    },
    NoSplinesError,
}

impl std::fmt::Display for MergeSplinesError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MergeSplinesError::MismatchedDegreeError {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "spline at position {} has degree {}, but expected degree {}",
                pos, actual, expected
            ),
            MergeSplinesError::MismatchedControlPointCountError {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "spline at position {} has {} control points, but expected {}",
                pos, actual, expected
            ),
            MergeSplinesError::MismatchedKnotCountError {
                pos,
                expected,
                actual,
            } => write!(
                f,
                "spline at position {} has {} knots, but expected {}",
                pos, actual, expected
            ),
            MergeSplinesError::NoSplinesError => write!(f, "no splines to merge"),
        }
    }
}

impl std::error::Error for MergeSplinesError {}
