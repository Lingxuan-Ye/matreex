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
