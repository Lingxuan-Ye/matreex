/// Creates a new [`Matrix`] instance from literal.
///
/// # Examples
///
/// ```
/// use matreex::matrix;
///
/// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
/// ```
///
/// [`Matrix`]: crate::matrix::Matrix
#[macro_export]
macro_rules! matrix {
    [$($col:expr),+ $(,)?] => {
        $crate::matrix::Matrix::from(std::boxed::Box::new([$($col,)+]))
    };
}

/// Creates a new [`Vector`] instance from literal.
///
/// # Examples
///
/// ```
/// use matreex::{vector, Vector};
///
/// let foo: Vector<i32> = vector![];
/// let bar = vector![0; 3];
/// let baz = vector![0, 1, 2];
/// ```
///
/// [`Vector`]: crate::vector::Vector
#[macro_export]
macro_rules! vector {
    [] => {
        $crate::vector::Vector::from_vec(vec![])
    };

    [$elem:expr; $n:expr] => {
        $crate::vector::Vector::from_vec(vec![$elem; $n])
    };

    [$($x:expr),+ $(,)?] => {
        $crate::vector::Vector::from_vec(vec![$($x),+])
    };
}
