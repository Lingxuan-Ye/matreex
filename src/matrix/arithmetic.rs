use super::index::transpose_flattened_index;
use super::iter::VectorIter;
use super::order::Order;
use super::shape::Shape;
use super::Matrix;
use crate::error::{Error, Result};

mod add;
mod div;
mod mul;
mod neg;
mod rem;
mod sub;

impl<L> Matrix<L> {
    /// Ensures that two matrices are conformable for elementwise operations.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, Matrix};
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let lhs = Matrix::<i32>::with_default((2, 3))?;
    ///
    /// let rhs = Matrix::<i32>::with_default((2, 3))?;
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = Matrix::<i32>::with_default((3, 2))?;
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::ShapeNotConformable));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn ensure_elementwise_operation_conformable<R>(&self, rhs: &Matrix<R>) -> Result<&Self> {
        if self.shape().eq(&rhs.shape()) {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }

    /// Performs elementwise operation on two matrices.
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
    /// let result = lhs.elementwise_operation(&rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[2, 3, 4], [5, 6, 7]]));
    /// ```
    pub fn elementwise_operation<R, F, U>(&self, rhs: &Matrix<R>, mut op: F) -> Result<Matrix<U>>
    where
        F: FnMut(&L, &R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        let data = if self.order == rhs.order {
            self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = transpose_flattened_index(index, self.shape);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, consuming `self`.
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
    /// let result = lhs.elementwise_operation_consume_self(&rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[2, 3, 4], [5, 6, 7]]));
    /// ```
    pub fn elementwise_operation_consume_self<R, F, U>(
        self,
        rhs: &Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(L, &R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        let data = if self.order == rhs.order {
            self.data
                .into_iter()
                .zip(rhs.data.iter())
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = transpose_flattened_index(index, self.shape);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, assigning the result
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
    /// lhs.elementwise_operation_assign(&rhs, |x, y| *x += y)?;
    /// assert_eq!(lhs, matrix![[2, 3, 4], [5, 6, 7]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn elementwise_operation_assign<R, F>(
        &mut self,
        rhs: &Matrix<R>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut(&mut L, &R),
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        if self.order == rhs.order {
            self.data
                .iter_mut()
                .zip(rhs.data.iter())
                .for_each(|(left, right)| op(left, right));
        } else {
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = transpose_flattened_index(index, self.shape);
                let right = unsafe { rhs.data.get_unchecked(index) };
                op(left, right)
            });
        }

        Ok(self)
    }
}

impl<L> Matrix<L> {
    /// Ensures that two matrices are conformable for multiplication-like
    /// operation.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, Matrix};
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let lhs = Matrix::<i32>::with_default((2, 3))?;
    ///
    /// let rhs = Matrix::<i32>::with_default((3, 2))?;
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = Matrix::<i32>::with_default((2, 3))?;
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::ShapeNotConformable));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn ensure_multiplication_like_operation_conformable<R>(
        &self,
        rhs: &Matrix<R>,
    ) -> Result<&Self> {
        if self.ncols() == rhs.nrows() {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }

    /// Performs multiplication-like operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// For performance reasons, this method consumes both `self` and `rhs`.
    ///
    /// The closure `op` is guaranteed to receive two non-empty, equal-length
    /// vectors. It should always return a valid value derived from them.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, VectorIter};
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[0, 1], [2, 3], [4, 5]];
    /// let op = |lv: VectorIter<&i32>, rv: VectorIter<&i32>| {
    ///     lv.zip(rv).map(|(x, y)| x * y).reduce(|acc, p| acc + p).unwrap()
    /// };
    /// let result = lhs.multiplication_like_operation(rhs, op);
    /// assert_eq!(result, Ok(matrix![[10, 13], [28, 40]]));
    /// ```
    pub fn multiplication_like_operation<R, F, U>(
        mut self,
        mut rhs: Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(VectorIter<&L>, VectorIter<&R>) -> U,
        U: Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let order = self.order;
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
        let size = shape.size();
        let mut data = Vec::with_capacity(size);

        if self.ncols() == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { order, shape, data });
        }

        self.set_order(Order::RowMajor);
        rhs.set_order(Order::ColMajor);

        match order {
            Order::RowMajor => {
                for row in 0..nrows {
                    for col in 0..ncols {
                        unsafe {
                            data.push(op(
                                Box::new(self.iter_nth_major_axis_vector_unchecked(row)),
                                Box::new(rhs.iter_nth_major_axis_vector_unchecked(col)),
                            ));
                        }
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        unsafe {
                            data.push(op(
                                Box::new(self.iter_nth_major_axis_vector_unchecked(row)),
                                Box::new(rhs.iter_nth_major_axis_vector_unchecked(col)),
                            ));
                        }
                    }
                }
            }
        }

        Ok(Matrix { order, shape, data })
    }
}

impl<T> Matrix<T> {
    /// Performs scalar operation on the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let scalar = 2;
    /// let output = matrix.scalar_operation(&scalar, |x, y| x + y);
    /// assert_eq!(output, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    #[inline]
    pub fn scalar_operation<S, F, U>(&self, scalar: &S, mut op: F) -> Matrix<U>
    where
        F: FnMut(&T, &S) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self
            .data
            .iter()
            .map(|element| op(element, scalar))
            .collect();
        Matrix { order, shape, data }
    }

    /// Performs scalar operation on the matrix, consuming `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let scalar = 2;
    /// let output = matrix.scalar_operation_consume_self(&scalar, |x, y| x + y);
    /// assert_eq!(output, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    #[inline]
    pub fn scalar_operation_consume_self<S, F, U>(self, scalar: &S, mut op: F) -> Matrix<U>
    where
        F: FnMut(T, &S) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self
            .data
            .into_iter()
            .map(|element| op(element, scalar))
            .collect();
        Matrix { order, shape, data }
    }

    /// Performs scalar operation on the matrix, assigning the result
    /// to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let scalar = 2;
    /// matrix.scalar_operation_assign(&scalar, |x, y| *x += y);
    /// assert_eq!(matrix, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    #[inline]
    pub fn scalar_operation_assign<S, F>(&mut self, scalar: &S, mut op: F) -> &mut Self
    where
        F: FnMut(&mut T, &S),
    {
        self.data.iter_mut().for_each(|element| op(element, scalar));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_ensure_elementwise_operation_conformable() {
        let mut lhs = Matrix::<i32>::with_default((2, 3)).unwrap();
        let mut rhs = Matrix::<i32>::with_default((2, 3)).unwrap();

        // default order & default order
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // default order & alternative order
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        lhs.switch_order();

        // alternative order & alternative order
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // alternative order & default order
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::with_default((2, 2)).unwrap();
        let error = lhs
            .ensure_elementwise_operation_conformable(&rhs)
            .unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);

        let rhs = Matrix::<i32>::with_default((3, 2)).unwrap();
        let error = lhs
            .ensure_elementwise_operation_conformable(&rhs)
            .unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);
    }

    #[test]
    fn test_elementwise_operation() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let op = |x: &i32, y: &i32| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // default order & default order
        let output = lhs.elementwise_operation(&rhs, op).unwrap();
        assert_eq!(output, expected);

        rhs.switch_order();

        // default order & alternative order
        let output = lhs.elementwise_operation(&rhs, op).unwrap();
        assert_eq!(output, expected);

        lhs.switch_order();

        // alternative order & alternative order
        let mut output = lhs.elementwise_operation(&rhs, op).unwrap();
        output.switch_order();
        assert_eq!(output, expected);

        rhs.switch_order();

        // alternative order & default order
        let mut output = lhs.elementwise_operation(&rhs, op).unwrap();
        output.switch_order();
        assert_eq!(output, expected);

        let rhs = matrix![[2, 2], [2, 2]];
        let error = lhs.elementwise_operation(&rhs, op).unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);

        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        let error = lhs.elementwise_operation(&rhs, op).unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);
    }

    #[test]
    fn test_elementwise_operation_consume_self() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let op = |x, y: &i32| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // default order & default order
        {
            let lhs = lhs.clone();
            let output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // default order & alternative order
        {
            let lhs = lhs.clone();
            let output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        // alternative order & alternative order
        {
            let lhs = lhs.clone();
            let mut output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // alternative order & default order
        {
            let lhs = lhs.clone();
            let mut output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2]];
            let error = lhs
                .elementwise_operation_consume_self(&rhs, op)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]];
            let error = lhs
                .elementwise_operation_consume_self(&rhs, op)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }
    }

    #[test]
    fn test_elementwise_operation_assign() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let op = |x: &mut i32, y: &i32| *x += y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // default order & default order
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            assert_eq!(lhs, expected);
        }

        rhs.switch_order();

        // default order & alternative order
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            assert_eq!(lhs, expected);
        }

        lhs.switch_order();

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            lhs.switch_order();
            assert_eq!(lhs, expected);
        }

        rhs.switch_order();

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            lhs.switch_order();
            assert_eq!(lhs, expected);
        }

        let unchanged = lhs.clone();

        let rhs = matrix![[2, 2], [2, 2]];
        let error = lhs.elementwise_operation_assign(&rhs, op).unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);
        assert_eq!(lhs, unchanged);

        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        let error = lhs.elementwise_operation_assign(&rhs, op).unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);
        assert_eq!(lhs, unchanged);
    }

    #[test]
    fn test_ensure_multiplication_like_operation_conformable() {
        let mut lhs = Matrix::<i32>::with_default((2, 3)).unwrap();
        let mut rhs = Matrix::<i32>::with_default((3, 2)).unwrap();

        // default order & default order
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // default order & alternative order
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        lhs.switch_order();

        // alternative order & alternative order
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // alternative order & default order
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::with_default((3, 1)).unwrap();
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::with_default((3, 3)).unwrap();
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::with_default((2, 2)).unwrap();
        let error = lhs
            .ensure_multiplication_like_operation_conformable(&rhs)
            .unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);

        let rhs = Matrix::<i32>::with_default((2, 3)).unwrap();
        let error = lhs
            .ensure_multiplication_like_operation_conformable(&rhs)
            .unwrap_err();
        assert_eq!(error, Error::ShapeNotConformable);
    }

    #[test]
    fn test_multiplication_like_operation() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[0, 1], [2, 3], [4, 5]];
        let op = |vl: VectorIter<&i32>, vr: VectorIter<&i32>| {
            vl.zip(vr)
                .map(|(x, y)| x * y)
                .reduce(|acc, p| acc + p)
                .unwrap()
        };
        let expected = matrix![[10, 13], [28, 40]];

        // default order & default order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // default order & alternative order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        // alternative order & alternative order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let mut output = lhs.multiplication_like_operation(rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // alternative order & default order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let mut output = lhs.multiplication_like_operation(rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0], [1], [2]];
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, matrix![[5], [14]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, matrix![[15, 18, 21], [42, 54, 66]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1], [2, 3]];
            let error = lhs.multiplication_like_operation(rhs, op).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1, 3], [4, 5, 6]];
            let error = lhs.multiplication_like_operation(rhs, op).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }
    }

    #[test]
    fn test_scalar_operation() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let scalar = 2;
        let op = |x: &i32, y: &i32| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // default order
        let output = matrix.scalar_operation(&scalar, op);
        assert_eq!(output, expected);

        matrix.switch_order();

        // alternative order
        let mut output = matrix.scalar_operation(&scalar, op);
        output.switch_order();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_scalar_operation_consume_self() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let scalar = 2;
        let op = |x: i32, y: &i32| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // default order
        {
            let matrix = matrix.clone();
            let output = matrix.scalar_operation_consume_self(&scalar, op);
            assert_eq!(output, expected);
        }

        matrix.switch_order();

        // alternative order
        {
            let matrix = matrix.clone();
            let mut output = matrix.scalar_operation_consume_self(&scalar, op);
            output.switch_order();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn test_scalar_operation_assign() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let scalar = 2;
        let op = |x: &mut i32, y: &i32| *x += y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // default order
        {
            let mut matrix = matrix.clone();
            matrix.scalar_operation_assign(&scalar, op);
            assert_eq!(matrix, expected);
        }

        matrix.switch_order();

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.scalar_operation_assign(&scalar, op);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }
}
