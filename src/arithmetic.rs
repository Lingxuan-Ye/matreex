use crate::Matrix;
use crate::error::{Error, Result};
use crate::index::AxisIndex;
use crate::order::Order;
use crate::shape::Shape;

mod add;
mod div;
mod mul;
mod neg;
mod rem;
mod sub;

impl<L> Matrix<L> {
    /// Returns `true` if the matrix is a square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix =matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert!(matrix.is_square());
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert!(!matrix.is_square());
    /// ```
    #[inline]
    pub fn is_square(&self) -> bool {
        let shape = self.shape();
        shape.nrows() == shape.ncols()
    }

    /// Returns `true` if two matrices are conformable for elementwise
    /// operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// assert!(lhs.is_elementwise_operation_conformable(&rhs));
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// assert!(!lhs.is_elementwise_operation_conformable(&rhs));
    /// ```
    #[inline]
    pub fn is_elementwise_operation_conformable<R>(&self, rhs: &Matrix<R>) -> bool {
        if self.order == rhs.order {
            self.shape == rhs.shape
        } else {
            self.major() == rhs.minor() && self.minor() == rhs.major()
        }
    }

    /// Returns `true` if two matrices are conformable for multiplication-like
    /// operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// assert!(!lhs.is_multiplication_like_operation_conformable(&rhs));
    /// ```
    #[inline]
    pub fn is_multiplication_like_operation_conformable<R>(&self, rhs: &Matrix<R>) -> bool {
        self.ncols() == rhs.nrows()
    }
}

impl<L> Matrix<L> {
    /// Ensures that the matrix is a square matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::SquareMatrixRequired`] if the matrix is not a square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let matrix =matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let result = matrix.ensure_square();
    /// assert!(result.is_ok());
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let result = matrix.ensure_square();
    /// assert_eq!(result, Err(Error::SquareMatrixRequired));
    /// ```
    #[inline]
    pub fn ensure_square(&self) -> Result<&Self> {
        if self.is_square() {
            Ok(self)
        } else {
            Err(Error::SquareMatrixRequired)
        }
    }

    /// Ensures that two matrices are conformable for elementwise operations.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::ShapeNotConformable));
    /// ```
    #[inline]
    pub fn ensure_elementwise_operation_conformable<R>(&self, rhs: &Matrix<R>) -> Result<&Self> {
        if self.is_elementwise_operation_conformable(rhs) {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }

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
    /// use matreex::{Error, matrix};
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::ShapeNotConformable));
    /// ```
    #[inline]
    pub fn ensure_multiplication_like_operation_conformable<R>(
        &self,
        rhs: &Matrix<R>,
    ) -> Result<&Self> {
        if self.is_multiplication_like_operation_conformable(rhs) {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }
}

impl<L> Matrix<L> {
    /// Performs elementwise operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_operation(&rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn elementwise_operation<'a, 'b, R, F, U>(
        &'a self,
        rhs: &'b Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(&'a L, &'b R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        let data = if self.order == rhs.order {
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = AxisIndex::from_flattened(index, self.shape)
                        .swap()
                        .to_flattened(rhs.shape);
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
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_operation_consume_self(&rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn elementwise_operation_consume_self<'a, R, F, U>(
        self,
        rhs: &'a Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(L, &'a R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        let data = if self.order == rhs.order {
            self.data
                .into_iter()
                .zip(&rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = AxisIndex::from_flattened(index, self.shape)
                        .swap()
                        .to_flattened(rhs.shape);
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
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// lhs.elementwise_operation_assign(&rhs, |x, y| *x += y)?;
    /// assert_eq!(lhs, matrix![[3, 4, 5], [6, 7, 8]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn elementwise_operation_assign<'a, R, F>(
        &mut self,
        rhs: &'a Matrix<R>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut(&mut L, &'a R),
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        if self.order == rhs.order {
            self.data
                .iter_mut()
                .zip(&rhs.data)
                .for_each(|(left, right)| op(left, right));
        } else {
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = AxisIndex::from_flattened(index, self.shape)
                    .swap()
                    .to_flattened(rhs.shape);
                let right = unsafe { rhs.data.get_unchecked(index) };
                op(left, right)
            });
        }

        Ok(self)
    }
}

impl<L> Matrix<L> {
    /// Performs multiplication-like operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The closure `op` is guaranteed to receive two non-empty, equal-length
    /// slices. It should always return a valid value derived from them.
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// fn dot_product(lhs: &[i32], rhs: &[i32]) -> i32 {
    ///     lhs.iter()
    ///         .zip(rhs)
    ///         .map(|(x, y)| x * y)
    ///         .reduce(|acc, p| acc + p)
    ///         .unwrap()
    /// }
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[1, 2], [3, 4], [5, 6]];
    /// let result = lhs.multiplication_like_operation(rhs, dot_product);
    /// assert_eq!(result, Ok(matrix![[22, 28], [49, 64]]));
    /// ```
    pub fn multiplication_like_operation<R, F, U>(
        mut self,
        mut rhs: Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(&[L], &[R]) -> U,
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
                        let lhs = unsafe { self.get_nth_major_axis_vector(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let lhs = unsafe { self.get_nth_major_axis_vector(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let scalar = 2;
    /// let output = matrix.scalar_operation(&scalar, |x, y| x + y);
    /// assert_eq!(output, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    #[inline]
    pub fn scalar_operation<'a, 'b, S, F, U>(&'a self, scalar: &'b S, mut op: F) -> Matrix<U>
    where
        F: FnMut(&'a T, &'b S) -> U,
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let scalar = 2;
    /// let output = matrix.scalar_operation_consume_self(&scalar, |x, y| x + y);
    /// assert_eq!(output, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    #[inline]
    pub fn scalar_operation_consume_self<'a, S, F, U>(self, scalar: &'a S, mut op: F) -> Matrix<U>
    where
        F: FnMut(T, &'a S) -> U,
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let scalar = 2;
    /// matrix.scalar_operation_assign(&scalar, |x, y| *x += y);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    #[inline]
    pub fn scalar_operation_assign<'a, S, F>(&mut self, scalar: &'a S, mut op: F) -> &mut Self
    where
        F: FnMut(&mut T, &'a S),
    {
        self.data.iter_mut().for_each(|element| op(element, scalar));
        self
    }
}

impl<T> Matrix<T> {
    /// # Safety
    ///
    /// Calling this method when `n >= self.major()` is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline(always)]
    unsafe fn get_nth_major_axis_vector(&self, n: usize) -> &[T] {
        let lower = n * self.major_stride();
        let upper = lower + self.major_stride();
        unsafe { self.data.get_unchecked(lower..upper) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_is_square() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert!(matrix.is_square());

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        assert!(!matrix.is_square());
    }

    #[test]
    fn test_is_elementwise_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];

        // default order & default order
        {
            assert!(lhs.is_elementwise_operation_conformable(&rhs));
            assert!(rhs.is_elementwise_operation_conformable(&lhs));
        }

        // default order & alternative order
        {
            let mut rhs = rhs.clone();
            rhs.switch_order();

            assert!(lhs.is_elementwise_operation_conformable(&rhs));
            assert!(rhs.is_elementwise_operation_conformable(&lhs));
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.switch_order();

            assert!(lhs.is_elementwise_operation_conformable(&rhs));
            assert!(rhs.is_elementwise_operation_conformable(&lhs));
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            assert!(lhs.is_elementwise_operation_conformable(&rhs));
            assert!(rhs.is_elementwise_operation_conformable(&lhs));
        }

        // more test cases

        {
            let rhs = matrix![[2, 2], [2, 2]];

            assert!(!lhs.is_elementwise_operation_conformable(&rhs));
            assert!(!rhs.is_elementwise_operation_conformable(&lhs));
        }

        {
            let rhs = matrix![[2, 2], [2, 2], [2, 2]];

            assert!(!lhs.is_elementwise_operation_conformable(&rhs));
            assert!(!rhs.is_elementwise_operation_conformable(&lhs));
        }

        {
            let rhs = matrix![[2, 2, 2], [2, 2, 2], [2, 2, 2]];

            assert!(!lhs.is_elementwise_operation_conformable(&rhs));
            assert!(!rhs.is_elementwise_operation_conformable(&lhs));
        }
    }

    #[test]
    fn test_is_multiplication_like_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];

        // default order & default order
        {
            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        }

        // default order & alternative order
        {
            let mut rhs = rhs.clone();
            rhs.switch_order();

            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.switch_order();

            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        }

        // more test cases

        {
            let rhs = matrix![[2, 2], [2, 2]];

            assert!(!lhs.is_multiplication_like_operation_conformable(&rhs));
        }

        {
            let rhs = matrix![[2, 2, 2], [2, 2, 2]];

            assert!(!lhs.is_multiplication_like_operation_conformable(&rhs));
        }

        {
            let rhs = matrix![[2, 2, 2], [2, 2, 2], [2, 2, 2]];

            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        }
    }

    #[test]
    fn test_ensure_square() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let result = matrix.ensure_square();
        assert!(result.is_ok());

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let result = matrix.ensure_square();
        assert_eq!(result, Err(Error::SquareMatrixRequired));
    }

    #[test]
    fn test_ensure_elementwise_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];

        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert_eq!(result, Err(Error::ShapeNotConformable));
    }

    #[test]
    fn test_ensure_multiplication_like_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];

        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert_eq!(result, Err(Error::ShapeNotConformable));
    }

    #[test]
    fn test_elementwise_operation() {
        fn add(x: &i32, y: &i32) -> i32 {
            x + y
        }

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order & default order
        {
            let output = lhs.elementwise_operation(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // default order & alternative order
        {
            let mut rhs = rhs.clone();
            rhs.switch_order();

            let output = lhs.elementwise_operation(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.switch_order();

            let output = lhs.elementwise_operation(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & alternative order

        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            let output = lhs.elementwise_operation(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // more test cases

        {
            let rhs = matrix![[2, 2], [2, 2]];

            let error = lhs.elementwise_operation(&rhs, add).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let rhs = matrix![[2, 2], [2, 2], [2, 2]];

            let error = lhs.elementwise_operation(&rhs, add).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        // misuse but should work
        {
            let output = lhs.elementwise_operation(&rhs, |x, _| x).unwrap();
            assert_eq!(output, matrix![[&1, &2, &3], [&4, &5, &6]]);

            let output = lhs.elementwise_operation(&rhs, |_, y| y).unwrap();
            assert_eq!(output, matrix![[&2, &2, &2], [&2, &2, &2]]);

            let output = {
                let rhs = rhs.clone();
                lhs.elementwise_operation(&rhs, |x, _| x).unwrap()
            };
            assert_eq!(output, matrix![[&1, &2, &3], [&4, &5, &6]]);

            let output = {
                let lhs = lhs.clone();
                lhs.elementwise_operation(&rhs, |_, y| y).unwrap()
            };
            assert_eq!(output, matrix![[&2, &2, &2], [&2, &2, &2]]);
        }
    }

    #[test]
    fn test_elementwise_operation_consume_self() {
        fn add(x: i32, y: &i32) -> i32 {
            x + y
        }

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order & default order
        {
            let lhs = lhs.clone();

            let output = lhs.elementwise_operation_consume_self(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // default order & alternative order
        {
            let lhs = lhs.clone();
            let mut rhs = rhs.clone();
            rhs.switch_order();

            let output = lhs.elementwise_operation_consume_self(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.switch_order();

            let output = lhs.elementwise_operation_consume_self(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            let output = lhs.elementwise_operation_consume_self(&rhs, add).unwrap();
            assert_eq!(output, expected);
        }

        // more test cases

        {
            let lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2]];

            let error = lhs
                .elementwise_operation_consume_self(&rhs, add)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]];

            let error = lhs
                .elementwise_operation_consume_self(&rhs, add)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        // misuse but should work
        {
            let lhs = lhs.clone();

            let output = lhs
                .elementwise_operation_consume_self(&rhs, |_, y| y)
                .unwrap();
            assert_eq!(output, matrix![[&2, &2, &2], [&2, &2, &2]]);
        }
    }

    #[test]
    fn test_elementwise_operation_assign() {
        fn add_assign(x: &mut i32, y: &i32) {
            *x += y;
        }

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order & default order
        {
            let mut lhs = lhs.clone();

            lhs.elementwise_operation_assign(&rhs, add_assign).unwrap();
            assert_eq!(lhs, expected);
        }

        // default order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            rhs.switch_order();

            lhs.elementwise_operation_assign(&rhs, add_assign).unwrap();
            assert_eq!(lhs, expected);
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.switch_order();

            lhs.elementwise_operation_assign(&rhs, add_assign).unwrap();
            assert_eq!(lhs, expected);
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            lhs.elementwise_operation_assign(&rhs, add_assign).unwrap();
            assert_eq!(lhs, expected);
        }

        // more test cases

        let unchanged = lhs.clone();

        {
            let mut lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2]];

            let error = lhs
                .elementwise_operation_assign(&rhs, add_assign)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            assert_eq!(lhs, unchanged);
        }

        {
            let mut lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]];

            let error = lhs
                .elementwise_operation_assign(&rhs, add_assign)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            assert_eq!(lhs, unchanged);
        }

        // misuse but should work
        {
            let mut lhs = lhs.map_ref(|x| x);

            lhs.elementwise_operation_assign(&rhs, |x, y| *x = y)
                .unwrap();
            assert_eq!(lhs, matrix![[&2, &2, &2], [&2, &2, &2]]);
        }
    }

    #[test]
    fn test_multiplication_like_operation() {
        fn dot_product(lhs: &[i32], rhs: &[i32]) -> i32 {
            lhs.iter()
                .zip(rhs)
                .map(|(x, y)| x * y)
                .reduce(|acc, p| acc + p)
                .unwrap()
        }

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        let expected = matrix![[22, 28], [49, 64]];

        // default order & default order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();

            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            assert_eq!(output, expected);
        }

        // default order & alternative order
        {
            let lhs = lhs.clone();
            let mut rhs = rhs.clone();
            rhs.switch_order();

            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            let rhs = rhs.clone();
            lhs.switch_order();

            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            assert_eq!(output, expected);
        }

        // more test cases

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1], [2], [3]];

            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            assert_eq!(output, matrix![[14], [32]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            assert_eq!(output, matrix![[30, 36, 42], [66, 81, 96]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2], [3, 4]];

            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2, 3], [4, 5, 6]];

            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }
    }

    #[test]
    fn test_scalar_operation() {
        fn add(x: &i32, y: &i32) -> i32 {
            x + y
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let scalar = 2;
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order
        {
            let output = matrix.scalar_operation(&scalar, add);
            assert_eq!(output, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            let output = matrix.scalar_operation(&scalar, add);
            assert_eq!(output, expected);
        }

        // misuse but should work
        {
            let output = matrix.scalar_operation(&scalar, |x, _| x);
            assert_eq!(output, matrix![[&1, &2, &3], [&4, &5, &6]]);

            let output = matrix.scalar_operation(&scalar, |_, y| y);
            assert_eq!(output, matrix![[&2, &2, &2], [&2, &2, &2]]);

            let output = {
                let scalar = 2;
                matrix.scalar_operation(&scalar, |x, _| x)
            };
            assert_eq!(output, matrix![[&1, &2, &3], [&4, &5, &6]]);

            let output = {
                let matrix = matrix.clone();
                matrix.scalar_operation(&scalar, |_, y| y)
            };
            assert_eq!(output, matrix![[&2, &2, &2], [&2, &2, &2]]);
        }
    }

    #[test]
    fn test_scalar_operation_consume_self() {
        fn add(x: i32, y: &i32) -> i32 {
            x + y
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let scalar = 2;
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order
        {
            let matrix = matrix.clone();

            let output = matrix.scalar_operation_consume_self(&scalar, add);
            assert_eq!(output, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            let output = matrix.scalar_operation_consume_self(&scalar, add);
            assert_eq!(output, expected);
        }

        // misuse but should work
        {
            let matrix = matrix.clone();

            let output = matrix.scalar_operation_consume_self(&scalar, |_, y| y);
            assert_eq!(output, matrix![[&2, &2, &2], [&2, &2, &2]]);
        }
    }

    #[test]
    fn test_scalar_operation_assign() {
        fn add_assign(x: &mut i32, y: &i32) {
            *x += y;
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let scalar = 2;
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order
        {
            let mut matrix = matrix.clone();

            matrix.scalar_operation_assign(&scalar, add_assign);
            assert_eq!(matrix, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.scalar_operation_assign(&scalar, add_assign);
            assert_eq!(matrix, expected);
        }

        // misuse but should work
        {
            let mut matrix = matrix.map_ref(|x| x);

            matrix.scalar_operation_assign(&scalar, |x, y| *x = y);
            assert_eq!(matrix, matrix![[&2, &2, &2], [&2, &2, &2]]);
        }
    }
}
