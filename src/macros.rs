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
    [] => {{
        use $crate::Matrix;

        Matrix::<_>::new()
    }};

    [[$elem:expr; $ncols:expr]; $nrows:expr] => {{
        use $crate::Matrix;

        match Matrix::<_>::with_value(($nrows, $ncols), $elem) {
            Err(error) => ::core::panic!("{error}"),
            Ok(matrix) => matrix,
        }
    }};

    [[$($elem:expr),+ $(,)?]; $nrows:expr] => {{
        extern crate alloc;

        use $crate::Matrix;
        use $crate::convert::FromRows;

        <Matrix::<_> as FromRows<_>>::from_rows(alloc::vec![[$($elem),+]; $nrows])
    }};

    [$($row:expr),+ $(,)?] => {{
        use $crate::Matrix;
        use $crate::convert::FromRows;

        <Matrix::<_> as FromRows<_>>::from_rows([$($row),+])
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
