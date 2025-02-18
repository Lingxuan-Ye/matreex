/// Creates a new [`Matrix<T>`] from literal.
///
/// > I witnessed His decree:
/// >
/// > > Let there be matrix!
/// >
/// > Thus, [`matrix!`] was brought forth into existence.
///
/// # Examples
///
/// ```
/// use matreex::{matrix, Matrix};
///
/// let foo: Matrix<i32> = matrix![];
/// let bar = matrix![[0; 3]; 2];
/// let baz = matrix![[0, 1, 2]; 2];
/// let qux = matrix![[0, 1, 2], [3, 4, 5]];
/// ```
///
/// [`Matrix<T>`]: crate::matrix::Matrix
/// [`matrix!`]: crate::matrix!
#[macro_export]
macro_rules! matrix {
    [] => {
        $crate::matrix::Matrix::new()
    };

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {
        match $crate::matrix::Matrix::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::std::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    };

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {
        $crate::matrix::Matrix::from(::std::vec![[$($elem),+]; $nrows])
    };

    [$($row:expr),+ $(,)?] => {
        $crate::matrix::Matrix::from([$($row),+])
    };
}

/// Creates a new row vector from literal.
///
/// # Examples
///
/// ```
/// use matreex::{matrix, row_vec, Matrix};
///
/// let foo: Matrix<i32> = row_vec![];
/// assert_eq!(foo.nrows(), 1);
/// assert_eq!(foo.ncols(), 0);
///
/// let bar = row_vec![0; 3];
/// assert_eq!(bar, matrix![[0, 0, 0]]);
///
/// let baz = row_vec![0, 1, 2];
/// assert_eq!(baz, matrix![[0, 1, 2]]);
/// ```
#[macro_export]
macro_rules! row_vec {
    [] => {
        $crate::matrix::Matrix::from_row(::std::vec::Vec::new());
    };

    [$elem:expr; $n:expr] => {
        $crate::matrix::Matrix::from_row(::std::vec![$elem; $n])
    };

    [$($elem:expr),+ $(,)?] => {
        $crate::matrix::Matrix::from_row(::std::vec![$($elem),+])
    };
}

/// Creates a new column vector from literal.
///
/// # Examples
///
/// ```
/// use matreex::{matrix, col_vec, Matrix};
///
/// let foo: Matrix<i32> = col_vec![];
/// assert_eq!(foo.nrows(), 0);
/// assert_eq!(foo.ncols(), 1);
///
/// let bar = col_vec![0; 3];
/// assert_eq!(bar, matrix![[0], [0], [0]]);
///
/// let baz = col_vec![0, 1, 2];
/// assert_eq!(baz, matrix![[0], [1], [2]]);
/// ```
#[macro_export]
macro_rules! col_vec {
    [] => {
        $crate::matrix::Matrix::from_col(::std::vec::Vec::new());
    };

    [$elem:expr; $n:expr] => {
        $crate::matrix::Matrix::from_col(::std::vec![$elem; $n])
    };

    [$($elem:expr),+ $(,)?] => {
        $crate::matrix::Matrix::from_col(::std::vec![$($elem),+])
    };
}
