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

// For simplicity, all scalar operations rely on the behavior of `$t`,
// including those performed on references. The `where` clauses prevent
// `(&$t).clone` from returning a reference, which helps avoid misleading
// error messages.

/// Implements scalar addition for [`Matrix<T>`].
///
/// # Notes
///
/// A `scalar` does not have to be a scalar in the mathematical sense.
/// Instead, it can be any type except for [`Matrix<T>`]. However, if
/// you do need to treat some concrete type of [`Matrix<T>`] as a scalar,
/// you can wrap it in a newtype and implement all the necessary trait
/// bounds for it.
///
/// [`Matrix<T>`]: crate::matrix::Matrix
#[macro_export]
macro_rules! impl_scalar_add {
    ($($t:ty)*) => {
        $(
            impl ::std::ops::Add<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element + scalar.clone())
                }
            }

            impl ::std::ops::Add<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element + scalar.clone())
                }
            }

            impl ::std::ops::Add<$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| element.clone() + scalar.clone())
                }
            }

            impl ::std::ops::Add<&$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| element.clone() + scalar.clone())
                }
            }

            impl ::std::ops::Add<$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element.clone() + scalar.clone())
                }
            }

            impl ::std::ops::Add<&$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element.clone() + scalar.clone())
                }
            }

            impl ::std::ops::Add<$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| (*element).clone() + scalar.clone())
                }
            }

            impl ::std::ops::Add<&$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| (*element).clone() + scalar.clone())
                }
            }

            impl ::std::ops::Add<$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() + element)
                }
            }

            impl ::std::ops::Add<&$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() + element.clone())
                }
            }

            impl ::std::ops::Add<$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() + element.clone())
                }
            }

            impl ::std::ops::Add<&$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() + (*element).clone())
                }
            }

            impl ::std::ops::Add<$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() + element)
                }
            }

            impl ::std::ops::Add<&$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() + element.clone())
                }
            }

            impl ::std::ops::Add<$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() + element.clone())
                }
            }

            impl ::std::ops::Add<&$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn add(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() + (*element).clone())
                }
            }

            impl ::std::ops::AddAssign<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn add_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element += scalar.clone());
                }
            }

            impl ::std::ops::AddAssign<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn add_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(rhs, |element, scalar| *element += scalar.clone());
                }
            }
        )*
    }
}

/// Implements scalar subtraction for [`Matrix<T>`].
///
/// # Notes
///
/// A `scalar` does not have to be a scalar in the mathematical sense.
/// Instead, it can be any type except for [`Matrix<T>`]. However, if
/// you do need to treat some concrete type of [`Matrix<T>`] as a scalar,
/// you can wrap it in a newtype and implement all the necessary trait
/// bounds for it.
///
/// [`Matrix<T>`]: crate::matrix::Matrix
#[macro_export]
macro_rules! impl_scalar_sub {
    ($($t:ty)*) => {
        $(
            impl ::std::ops::Sub<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element - scalar.clone())
                }
            }

            impl ::std::ops::Sub<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element - scalar.clone())
                }
            }

            impl ::std::ops::Sub<$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| element.clone() - scalar.clone())
                }
            }

            impl ::std::ops::Sub<&$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| element.clone() - scalar.clone())
                }
            }

            impl ::std::ops::Sub<$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element.clone() - scalar.clone())
                }
            }

            impl ::std::ops::Sub<&$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element.clone() - scalar.clone())
                }
            }

            impl ::std::ops::Sub<$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| (*element).clone() - scalar.clone())
                }
            }

            impl ::std::ops::Sub<&$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| (*element).clone() - scalar.clone())
                }
            }

            impl ::std::ops::Sub<$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() - element)
                }
            }

            impl ::std::ops::Sub<&$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() - element.clone())
                }
            }

            impl ::std::ops::Sub<$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() - element.clone())
                }
            }

            impl ::std::ops::Sub<&$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() - (*element).clone())
                }
            }

            impl ::std::ops::Sub<$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() - element)
                }
            }

            impl ::std::ops::Sub<&$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() - element.clone())
                }
            }

            impl ::std::ops::Sub<$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() - element.clone())
                }
            }

            impl ::std::ops::Sub<&$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn sub(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() - (*element).clone())
                }
            }

            impl ::std::ops::SubAssign<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn sub_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element -= scalar.clone());
                }
            }

            impl ::std::ops::SubAssign<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn sub_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(rhs, |element, scalar| *element -= scalar.clone());
                }
            }
        )*
    }
}

/// Implements scalar multiplication for [`Matrix<T>`].
///
/// # Notes
///
/// A `scalar` does not have to be a scalar in the mathematical sense.
/// Instead, it can be any type except for [`Matrix<T>`]. However, if
/// you do need to treat some concrete type of [`Matrix<T>`] as a scalar,
/// you can wrap it in a newtype and implement all the necessary trait
/// bounds for it.
///
/// [`Matrix<T>`]: crate::matrix::Matrix
#[macro_export]
macro_rules! impl_scalar_mul {
    ($($t:ty)*) => {
        $(
            impl ::std::ops::Mul<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element * scalar.clone())
                }
            }

            impl ::std::ops::Mul<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element * scalar.clone())
                }
            }

            impl ::std::ops::Mul<$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl ::std::ops::Mul<&$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl ::std::ops::Mul<$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl ::std::ops::Mul<&$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl ::std::ops::Mul<$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| (*element).clone() * scalar.clone())
                }
            }

            impl ::std::ops::Mul<&$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| (*element).clone() * scalar.clone())
                }
            }

            impl ::std::ops::Mul<$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() * element)
                }
            }

            impl ::std::ops::Mul<&$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl ::std::ops::Mul<$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl ::std::ops::Mul<&$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() * (*element).clone())
                }
            }

            impl ::std::ops::Mul<$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() * element)
                }
            }

            impl ::std::ops::Mul<&$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl ::std::ops::Mul<$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl ::std::ops::Mul<&$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn mul(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() * (*element).clone())
                }
            }

            impl ::std::ops::MulAssign<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn mul_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= scalar.clone());
                }
            }

            impl ::std::ops::MulAssign<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn mul_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(rhs, |element, scalar| *element *= scalar.clone());
                }
            }
        )*
    }
}

/// Implements scalar division for [`Matrix<T>`].
///
/// # Notes
///
/// A `scalar` does not have to be a scalar in the mathematical sense.
/// Instead, it can be any type except for [`Matrix<T>`]. However, if
/// you do need to treat some concrete type of [`Matrix<T>`] as a scalar,
/// you can wrap it in a newtype and implement all the necessary trait
/// bounds for it.
///
/// [`Matrix<T>`]: crate::matrix::Matrix
#[macro_export]
macro_rules! impl_scalar_div {
    ($($t:ty)*) => {
        $(
            impl ::std::ops::Div<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element / scalar.clone())
                }
            }

            impl ::std::ops::Div<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element / scalar.clone())
                }
            }

            impl ::std::ops::Div<$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| element.clone() / scalar.clone())
                }
            }

            impl ::std::ops::Div<&$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| element.clone() / scalar.clone())
                }
            }

            impl ::std::ops::Div<$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element.clone() / scalar.clone())
                }
            }

            impl ::std::ops::Div<&$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element.clone() / scalar.clone())
                }
            }

            impl ::std::ops::Div<$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| (*element).clone() / scalar.clone())
                }
            }

            impl ::std::ops::Div<&$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| (*element).clone() / scalar.clone())
                }
            }

            impl ::std::ops::Div<$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() / element)
                }
            }

            impl ::std::ops::Div<&$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() / element.clone())
                }
            }

            impl ::std::ops::Div<$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() / element.clone())
                }
            }

            impl ::std::ops::Div<&$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() / (*element).clone())
                }
            }

            impl ::std::ops::Div<$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() / element)
                }
            }

            impl ::std::ops::Div<&$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() / element.clone())
                }
            }

            impl ::std::ops::Div<$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() / element.clone())
                }
            }

            impl ::std::ops::Div<&$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn div(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() / (*element).clone())
                }
            }

            impl ::std::ops::DivAssign<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn div_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element /= scalar.clone());
                }
            }

            impl ::std::ops::DivAssign<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn div_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(rhs, |element, scalar| *element /= scalar.clone());
                }
            }
        )*
    }
}

/// Implements scalar remainder operation for [`Matrix<T>`].
///
/// # Notes
///
/// A `scalar` does not have to be a scalar in the mathematical sense.
/// Instead, it can be any type except for [`Matrix<T>`]. However, if
/// you do need to treat some concrete type of [`Matrix<T>`] as a scalar,
/// you can wrap it in a newtype and implement all the necessary trait
/// bounds for it.
///
/// [`Matrix<T>`]: crate::matrix::Matrix
#[macro_export]
macro_rules! impl_scalar_rem {
    ($($t:ty)*) => {
        $(
            impl ::std::ops::Rem<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element % scalar.clone())
                }
            }

            impl ::std::ops::Rem<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element % scalar.clone())
                }
            }

            impl ::std::ops::Rem<$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| element.clone() % scalar.clone())
                }
            }

            impl ::std::ops::Rem<&$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| element.clone() % scalar.clone())
                }
            }

            impl ::std::ops::Rem<$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element.clone() % scalar.clone())
                }
            }

            impl ::std::ops::Rem<&$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element.clone() % scalar.clone())
                }
            }

            impl ::std::ops::Rem<$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| (*element).clone() % scalar.clone())
                }
            }

            impl ::std::ops::Rem<&$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| (*element).clone() % scalar.clone())
                }
            }

            impl ::std::ops::Rem<$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() % element)
                }
            }

            impl ::std::ops::Rem<&$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() % element.clone())
                }
            }

            impl ::std::ops::Rem<$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() % element.clone())
                }
            }

            impl ::std::ops::Rem<&$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() % (*element).clone())
                }
            }

            impl ::std::ops::Rem<$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() % element)
                }
            }

            impl ::std::ops::Rem<&$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() % element.clone())
                }
            }

            impl ::std::ops::Rem<$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() % element.clone())
                }
            }

            impl ::std::ops::Rem<&$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                #[inline]
                fn rem(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() % (*element).clone())
                }
            }

            impl ::std::ops::RemAssign<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn rem_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element %= scalar.clone());
                }
            }

            impl ::std::ops::RemAssign<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                #[inline]
                fn rem_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(rhs, |element, scalar| *element %= scalar.clone());
                }
            }
        )*
    }
}
