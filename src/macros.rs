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
/// [`Matrix<T>`]: crate::dense::Matrix
/// [`matrix!`]: crate::matrix!
#[macro_export]
macro_rules! matrix {
    [] => {
        $crate::dense::Matrix::<_>::new()
    };

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {
        match $crate::dense::Matrix::<_>::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    };

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {{
        extern crate alloc;
        <$crate::dense::Matrix<_> as $crate::convert::FromRows<_>>::from_rows(alloc::vec![[$($elem),+]; $nrows])
    }};

    [$($row:expr),+ $(,)?] => {
        <$crate::dense::Matrix<_> as $crate::convert::FromRows<_>>::from_rows([$($row),+])
    };
}

/// Creates a new [`RowMajorMatrix<T>`] from literal.
///
/// > And lo, I beheld the elements ride forth,
/// >
/// > their charge laying the foundation for order.
///
/// # Examples
///
/// ```
/// use matreex::{RowMajorMatrix, rmatrix};
///
/// let foo: RowMajorMatrix<i32> = rmatrix![];
/// let bar = rmatrix![[0; 3]; 2];
/// let baz = rmatrix![[1, 2, 3]; 2];
/// let qux = rmatrix![[1, 2, 3], [4, 5, 6]];
/// ```
///
/// [`RowMajorMatrix<T>`]: crate::dense::RowMajorMatrix
/// [`rmatrix!`]: crate::rmatrix!
#[macro_export]
macro_rules! rmatrix {
    [] => {
        $crate::dense::RowMajorMatrix::<_>::new()
    };

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {
        match $crate::dense::RowMajorMatrix::<_>::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    };

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {{
        extern crate alloc;
        <$crate::dense::RowMajorMatrix<_> as $crate::convert::FromRows<_>>::from_rows(alloc::vec![[$($elem),+]; $nrows])
    }};

    [$($row:expr),+ $(,)?] => {
        <$crate::dense::RowMajorMatrix<_> as $crate::convert::FromRows<_>>::from_rows([$($row),+])
    };
}

/// Creates a new [`ColMajorMatrix<T>`] from literal.
///
/// > And lo, a harbinger of static dispatch appeared,
/// >
/// > its ascent shaping the pillars of chaos.
///
/// # Examples
///
/// ```
/// use matreex::{ColMajorMatrix, cmatrix};
///
/// let foo: ColMajorMatrix<i32> = cmatrix![];
/// let bar = cmatrix![[0; 2]; 3];
/// let baz = cmatrix![[1, 2]; 3];
/// let qux = cmatrix![[1, 4], [2, 5], [3, 6]];
/// ```
///
/// [`ColMajorMatrix<T>`]: crate::dense::ColMajorMatrix
/// [`cmatrix!`]: crate::cmatrix!
#[macro_export]
macro_rules! cmatrix {
    [] => {
        $crate::dense::ColMajorMatrix::<_>::new()
    };

    [[$elem:expr; $nrows:expr]; $ncols:expr] => {
        match $crate::dense::ColMajorMatrix::<_>::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    };

    [[$($elem:expr),+ $(,)?]; $ncols:expr] => {{
        extern crate alloc;
        <$crate::dense::ColMajorMatrix<_> as $crate::convert::FromCols<_>>::from_cols(alloc::vec![[$($elem),+]; $ncols])
    }};

    [$($col:expr),+ $(,)?] => {
        <$crate::dense::ColMajorMatrix<_> as $crate::convert::FromCols<_>>::from_cols([$($col),+])
    };
}
