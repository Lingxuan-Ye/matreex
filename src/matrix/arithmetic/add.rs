use super::super::Matrix;
use crate::error::Result;
use crate::impl_scalar_add;
use std::ops::{Add, AddAssign};

impl<L, R, U> Add<Matrix<R>> for Matrix<L>
where
    L: Add<R, Output = U>,
    R: Clone,
{
    type Output = Matrix<U>;

    fn add(self, rhs: Matrix<R>) -> Self::Output {
        self + &rhs
    }
}

impl<L, R, U> Add<&Matrix<R>> for Matrix<L>
where
    L: Add<R, Output = U>,
    R: Clone,
{
    type Output = Matrix<U>;

    fn add(self, rhs: &Matrix<R>) -> Self::Output {
        match self.elementwise_add_consume_self(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Add<Matrix<R>> for &Matrix<L>
where
    L: Add<R, Output = U> + Clone,
    R: Clone,
{
    type Output = Matrix<U>;

    fn add(self, rhs: Matrix<R>) -> Self::Output {
        self + &rhs
    }
}

impl<L, R, U> Add<&Matrix<R>> for &Matrix<L>
where
    L: Add<R, Output = U> + Clone,
    R: Clone,
{
    type Output = Matrix<U>;

    fn add(self, rhs: &Matrix<R>) -> Self::Output {
        match self.elementwise_add(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R> AddAssign<Matrix<R>> for Matrix<L>
where
    L: AddAssign<R>,
    R: Clone,
{
    fn add_assign(&mut self, rhs: Matrix<R>) {
        *self += &rhs
    }
}

impl<L, R> AddAssign<&Matrix<R>> for Matrix<L>
where
    L: AddAssign<R>,
    R: Clone,
{
    fn add_assign(&mut self, rhs: &Matrix<R>) {
        if let Err(error) = self.elementwise_add_assign(rhs) {
            panic!("{error}");
        }
    }
}

impl<L> Matrix<L> {
    /// Performs elementwise addition on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// let result = lhs.elementwise_add(&rhs);
    /// assert_eq!(result, Ok(matrix![[2, 3, 4], [5, 6, 7]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn elementwise_add<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Add<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |(left, right)| left.clone() + right.clone())
    }

    /// Performs elementwise addition on two matrices, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// let result = lhs.elementwise_add_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[2, 3, 4], [5, 6, 7]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn elementwise_add_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Add<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |(left, right)| left + right.clone())
    }

    /// Performs elementwise addition on two matrices, assigning the result
    /// to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// lhs.elementwise_add_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[2, 3, 4], [5, 6, 7]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn elementwise_add_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: AddAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |(left, right)| *left += right.clone())
    }
}

impl_scalar_add! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
