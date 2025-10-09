/// Creates a new [`Matrix<T>`] from a literal.
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
        $crate::rmatrix![]
    };

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {
        $crate::rmatrix![[$elem; $ncols]; $nrows]
    };

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {
        $crate::rmatrix![[$($elem),+]; $nrows]
    };

    [$($row:expr),+ $(,)?] => {
        $crate::rmatrix![$($row),+]
    };
}

/// Creates a new [`Matrix<T, RowMajor>`] from a literal.
///
/// > And lo, I beheld the elements ride forth,
/// >
/// > their charge laying the foundation for order.
///
/// # Examples
///
/// ```
/// use matreex::rmatrix;
/// use matreex::dense::Matrix;
/// use matreex::dense::layout::RowMajor;
///
/// let foo: Matrix<i32, RowMajor> = rmatrix![];
/// let bar = rmatrix![[0; 3]; 2];
/// let baz = rmatrix![[1, 2, 3]; 2];
/// let qux = rmatrix![[1, 2, 3], [4, 5, 6]];
/// ```
///
/// [`Matrix<T, RowMajor>`]: crate::dense::Matrix
/// [`rmatrix!`]: crate::rmatrix!
#[macro_export]
macro_rules! rmatrix {
    [] => {{
        use $crate::dense::Matrix;
        use $crate::dense::layout::RowMajor;

        Matrix::<_, RowMajor>::new()
    }};

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {{
        use $crate::dense::Matrix;
        use $crate::dense::layout::RowMajor;

        match Matrix::<_, RowMajor>::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    }};

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {{
        extern crate alloc;

        use $crate::convert::FromRows;
        use $crate::dense::Matrix;
        use $crate::dense::layout::RowMajor;

        <Matrix::<_, RowMajor> as FromRows<_>>::from_rows(alloc::vec![[$($elem),+]; $nrows])
    }};

    [$($row:expr),+ $(,)?] => {{
        use $crate::convert::FromRows;
        use $crate::dense::Matrix;
        use $crate::dense::layout::RowMajor;

        <Matrix::<_, RowMajor> as FromRows<_>>::from_rows([$($row),+])
    }};
}

/// Creates a new [`Matrix<T, ColMajor>`] from a literal.
///
/// > And lo, a harbinger of static dispatch appeared,
/// >
/// > its ascent shaping the pillars of chaos.
///
/// # Notes
///
/// The input literal still follows visual intuition, that is, the innermost
/// arrays represent rows.
///
/// # Examples
///
/// ```
/// use matreex::cmatrix;
/// use matreex::dense::Matrix;
/// use matreex::dense::layout::ColMajor;
///
/// let foo: Matrix<i32, ColMajor> = cmatrix![];
/// let bar = cmatrix![[0; 3]; 2];
/// let baz = cmatrix![[1, 2, 3]; 2];
/// let qux = cmatrix![[1, 2, 3], [4, 5, 6]];
/// ```
///
/// [`Matrix<T, ColMajor>`]: crate::dense::Matrix
/// [`cmatrix!`]: crate::cmatrix!
#[macro_export]
macro_rules! cmatrix {
    [] => {{
        use $crate::dense::Matrix;
        use $crate::dense::layout::ColMajor;

        Matrix::<_, ColMajor>::new()
    }};

    [[$elem:expr; $nrows:expr]; $ncols:expr] => {{
        use $crate::dense::Matrix;
        use $crate::dense::layout::ColMajor;

        match Matrix::<_, ColMajor>::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    }};

    [[$($elem:expr),+ $(,)?]; $ncols:expr] => {{
        extern crate alloc;

        use $crate::convert::FromRows;
        use $crate::dense::Matrix;
        use $crate::dense::layout::ColMajor;

        <Matrix::<_, ColMajor> as FromRows<_>>::from_rows(alloc::vec![[$($elem),+]; $ncols])
    }};

    [$($col:expr),+ $(,)?] => {{
        use $crate::convert::FromRows;
        use $crate::dense::Matrix;
        use $crate::dense::layout::ColMajor;

        <Matrix::<_, ColMajor> as FromRows<_>>::from_rows([$($col),+])
    }};
}

#[cfg(test)]
#[macro_export]
macro_rules! dispatch_unary {
    { $block:block } => {{
        use $crate::dense::layout::{ColMajor, RowMajor};

        {
            type O = RowMajor;

            $block
        }

        {
            type O = ColMajor;

            $block
        }
    }};
}

#[cfg(test)]
#[macro_export]
macro_rules! dispatch_binary {
    { $block:block } => {{
        use $crate::dense::layout::{ColMajor, RowMajor};

        {
            type O = RowMajor;
            type P = RowMajor;

            $block
        }

        {
            type O = RowMajor;
            type P = ColMajor;

            $block
        }

        {
            type O = ColMajor;
            type P = RowMajor;

            $block
        }

        {
            type O = ColMajor;
            type P = ColMajor;

            $block
        }
    }};
}
