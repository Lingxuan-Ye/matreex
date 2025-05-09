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
/// use matreex::{Matrix, matrix};
///
/// let foo: Matrix<i32> = matrix![];
/// let bar = matrix![[0; 3]; 2];
/// let baz = matrix![[1, 2, 3]; 2];
/// let qux = matrix![[1, 2, 3], [4, 5, 6]];
/// ```
///
/// [`Matrix<T>`]: crate::Matrix
/// [`matrix!`]: crate::matrix!
#[macro_export]
macro_rules! matrix {
    [] => {
        $crate::Matrix::new()
    };

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {
        match $crate::Matrix::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    };

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {{
        extern crate alloc;
        $crate::Matrix::from(alloc::vec![[$($elem),+]; $nrows])
    }};

    [$($row:expr),+ $(,)?] => {
        $crate::Matrix::from([$($row),+])
    };
}

/// Creates a new row vector from literal.
///
/// # Examples
///
/// ```
/// use matreex::{Matrix, matrix, row_vec};
///
/// let foo: Matrix<i32> = row_vec![];
/// assert_eq!(foo.nrows(), 1);
/// assert_eq!(foo.ncols(), 0);
///
/// let bar = row_vec![0; 3];
/// assert_eq!(bar, matrix![[0, 0, 0]]);
///
/// let baz = row_vec![1, 2, 3];
/// assert_eq!(baz, matrix![[1, 2, 3]]);
/// ```
#[macro_export]
macro_rules! row_vec {
    [] => {{
        extern crate alloc;
        $crate::Matrix::from_row(alloc::vec![])
    }};

    [$elem:expr; $n:expr] => {{
        extern crate alloc;
        $crate::Matrix::from_row(alloc::vec![$elem; $n])
    }};

    [$($elem:expr),+ $(,)?] => {{
        extern crate alloc;
        $crate::Matrix::from_row(alloc::vec![$($elem),+])
    }};
}

/// Creates a new column vector from literal.
///
/// # Examples
///
/// ```
/// use matreex::{Matrix, matrix, col_vec};
///
/// let foo: Matrix<i32> = col_vec![];
/// assert_eq!(foo.nrows(), 0);
/// assert_eq!(foo.ncols(), 1);
///
/// let bar = col_vec![0; 3];
/// assert_eq!(bar, matrix![[0], [0], [0]]);
///
/// let baz = col_vec![1, 2, 3];
/// assert_eq!(baz, matrix![[1], [2], [3]]);
/// ```
#[macro_export]
macro_rules! col_vec {
    [] => {{
        extern crate alloc;
        $crate::Matrix::from_col(alloc::vec![])
    }};

    [$elem:expr; $n:expr] => {{
        extern crate alloc;
        $crate::Matrix::from_col(alloc::vec![$elem; $n])
    }};

    [$($elem:expr),+ $(,)?] => {{
        extern crate alloc;
        $crate::Matrix::from_col(alloc::vec![$($elem),+])
    }};
}
