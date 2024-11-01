use super::super::Matrix;
use crate::error::Result;
use crate::impl_scalar_rem;
use std::ops::{Rem, RemAssign};

impl<L> Matrix<L> {
    /// Performs elementwise remainder operation on two matrices.
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
    /// let result = lhs.elementwise_rem(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, 1, 0], [1, 0, 1]]));
    /// ```
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, -1, -2], [-3, -4, -5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_rem(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, -1, 0], [-1, 0, -1]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn elementwise_rem<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Rem<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |left, right| left.clone() % right.clone())
    }

    /// Performs elementwise remainder operation on two matrices,
    /// consuming `self`.
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
    /// let result = lhs.clone().elementwise_rem_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, 1, 0], [1, 0, 1]]));
    /// ```
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, -1, -2], [-3, -4, -5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// let result = lhs.elementwise_rem_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, -1, 0], [-1, 0, -1]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn elementwise_rem_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Rem<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |left, right| left % right.clone())
    }

    /// Performs elementwise remainder operation on two matrices,
    /// assigning the result to `self`.
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
    /// lhs.elementwise_rem_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[0, 1, 0], [1, 0, 1]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn elementwise_rem_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: RemAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |left, right| *left %= right.clone())
    }
}

impl_scalar_rem! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
