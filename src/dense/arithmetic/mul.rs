use super::super::Matrix;
use super::super::layout::{ColMajor, Layout, Order, OrderKind, RowMajor};
use crate::error::{Error, Result};
use crate::shape::Shape;
use alloc::vec::Vec;
use core::ops::{Add, Mul, MulAssign};

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    /// Performs multiplication on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.ncols() != rhs.nrows()`.
    /// - [`Error::SizeOverflow`] if the computed size of the output matrix exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.multiply(rhs);
    /// assert_eq!(result, Ok(matrix![[12, 12], [30, 30]]));
    /// ```
    pub fn multiply<R, RO, U>(self, rhs: Matrix<R, RO>) -> Result<Matrix<U, LO>>
    where
        L: Mul<R, Output = U> + Clone,
        R: Clone,
        RO: Order,
        U: Add<Output = U> + Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let inner = self.ncols();
        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);

        if size == 0 {
            return Ok(Matrix { layout, data });
        }

        if inner == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { layout, data });
        }

        let mut lhs = self.with_order::<RowMajor>();
        let mut rhs = rhs.with_order::<ColMajor>();

        unsafe {
            lhs.layout = Layout::default();
            lhs.data.set_len(0);

            rhs.layout = Layout::default();
            rhs.data.set_len(0);
        }

        let lhs_base = lhs.data.as_ptr();
        let rhs_base = rhs.data.as_ptr();

        match LO::KIND {
            OrderKind::RowMajor => unsafe {
                let mut lhs_reset = lhs_base;

                for _ in 1..nrows {
                    let mut rhs_ptr = rhs_base;

                    for _ in 1..ncols {
                        let mut lhs_ptr = lhs_reset;
                        let mut element = (*lhs_ptr).clone() * (*rhs_ptr).clone();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + (*lhs_ptr).clone() * (*rhs_ptr).clone();
                        }
                        data.push(element);
                        rhs_ptr = rhs_ptr.add(1);
                    }

                    {
                        let mut lhs_ptr = lhs_reset;
                        let mut element = lhs_ptr.read() * (*rhs_ptr).clone();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + lhs_ptr.read() * (*rhs_ptr).clone();
                        }
                        data.push(element);
                    }

                    lhs_reset = lhs_reset.add(inner);
                }

                {
                    let mut rhs_ptr = rhs_base;

                    for _ in 1..ncols {
                        let mut lhs_ptr = lhs_reset;
                        let mut element = (*lhs_ptr).clone() * rhs_ptr.read();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + (*lhs_ptr).clone() * rhs_ptr.read();
                        }
                        data.push(element);
                        rhs_ptr = rhs_ptr.add(1);
                    }

                    {
                        let mut lhs_ptr = lhs_reset;
                        let mut element = lhs_ptr.read() * rhs_ptr.read();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + lhs_ptr.read() * rhs_ptr.read();
                        }
                        data.push(element);
                    }
                }
            },

            OrderKind::ColMajor => unsafe {
                let mut rhs_reset = rhs_base;

                for _ in 1..ncols {
                    let mut lhs_ptr = lhs_base;

                    for _ in 1..nrows {
                        let mut rhs_ptr = rhs_reset;
                        let mut element = (*lhs_ptr).clone() * (*rhs_ptr).clone();

                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + (*lhs_ptr).clone() * (*rhs_ptr).clone();
                        }
                        data.push(element);
                        lhs_ptr = lhs_ptr.add(1);
                    }

                    {
                        let mut rhs_ptr = rhs_reset;
                        let mut element = (*lhs_ptr).clone() * rhs_ptr.read();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + (*lhs_ptr).clone() * rhs_ptr.read();
                        }
                        data.push(element);
                    }

                    rhs_reset = rhs_reset.add(inner);
                }

                {
                    let mut lhs_ptr = lhs_base;

                    for _ in 1..nrows {
                        let mut rhs_ptr = rhs_reset;
                        let mut element = lhs_ptr.read() * (*rhs_ptr).clone();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + lhs_ptr.read() * (*rhs_ptr).clone();
                        }
                        data.push(element);
                        lhs_ptr = lhs_ptr.add(1);
                    }

                    {
                        let mut rhs_ptr = rhs_reset;
                        let mut element = lhs_ptr.read() * rhs_ptr.read();
                        for _ in 1..inner {
                            lhs_ptr = lhs_ptr.add(1);
                            rhs_ptr = rhs_ptr.add(1);
                            element = element + lhs_ptr.read() * rhs_ptr.read();
                        }
                        data.push(element);
                    }
                }
            },
        }

        Ok(Matrix { layout, data })
    }

    /// Performs multiplication-like operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.ncols() != rhs.nrows()`.
    /// - [`Error::SizeOverflow`] if the computed size of the output matrix exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The closure `op` is guaranteed to receive two non-empty, equal-length
    /// slices. It should always return a valid value derived from them.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// fn dot_product(lhs: &[i32], rhs: &[i32]) -> i32 {
    ///     lhs.iter()
    ///         .zip(rhs)
    ///         .map(|(lhs, rhs)| lhs * rhs)
    ///         .reduce(|sum, product| sum + product)
    ///         .unwrap()
    /// }
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.multiplication_like_operation(rhs, dot_product);
    /// assert_eq!(result, Ok(matrix![[12, 12], [30, 30]]));
    /// ```
    pub fn multiplication_like_operation<R, RO, F, U>(
        self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        RO: Order,
        F: FnMut(&[L], &[R]) -> U,
        U: Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let inner = self.ncols();
        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);

        if size == 0 {
            return Ok(Matrix { layout, data });
        }

        if inner == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { layout, data });
        }

        let lhs = self.with_order::<RowMajor>();
        let rhs = rhs.with_order::<ColMajor>();

        match LO::KIND {
            OrderKind::RowMajor => {
                for row in 0..nrows {
                    let lhs = unsafe { lhs.get_nth_major_axis_vector_unchecked(row) };
                    for col in 0..ncols {
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }

            OrderKind::ColMajor => {
                for col in 0..ncols {
                    let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                    for row in 0..nrows {
                        let lhs = unsafe { lhs.get_nth_major_axis_vector_unchecked(row) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { layout, data })
    }

    /// Ensures that two matrices are conformable for multiplication-like
    /// operation.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.ncols() != rhs.nrows()`.
    fn ensure_multiplication_like_operation_conformable<R, RO>(
        &self,
        rhs: &Matrix<R, RO>,
    ) -> Result<&Self>
    where
        RO: Order,
    {
        if self.ncols() != rhs.nrows() {
            Err(Error::ShapeNotConformable)
        } else {
            Ok(self)
        }
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Returns a shared reference of the nth major-axis vector, without performing
    /// any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method when `n >= self.major()` is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline(always)]
    unsafe fn get_nth_major_axis_vector_unchecked(&self, n: usize) -> &[T] {
        let stride = self.stride();
        let lower = n * stride.major();
        let upper = lower + stride.major();
        unsafe { self.data.get_unchecked(lower..upper) }
    }
}

impl<L, LO, R, RO, U> Mul<Matrix<R, RO>> for Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.multiply(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Mul<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: &Matrix<R, RO>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<L, LO, R, RO, U> Mul<Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: Matrix<R, RO>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<L, LO, R, RO, U> Mul<&Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: &Matrix<R, RO>) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl<O> Mul<$s> for Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element * *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Mul<$s> for &Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element * *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Mul<Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar * element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Mul<&Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: &Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation(&self, |element, scalar| *scalar * *element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }
        )*
    }
}

macro_rules! impl_primitive_scalar_mul {
    ($($t:ty)*) => {
        $(
            impl_helper! {
                ($t, $t, $t)
                ($t, &$t, $t)
                (&$t, $t, $t)
                (&$t, &$t, $t)
            }

            impl<O> MulAssign<$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn mul_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }

            impl<O> MulAssign<&$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn mul_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_mul! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockL, MockR, MockU, MockZeroSized, Scope};
    use crate::{dispatch_binary, dispatch_unary, matrix};

    #[test]
    fn test_multiply() {
        #[cfg(miri)]
        let lens = [0, 1, 2, 3, 5];
        #[cfg(not(miri))]
        let lens = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19];

        let mut pairs = Vec::with_capacity(lens.len().pow(3));
        for inner in lens {
            for nrows in lens {
                for ncols in lens {
                    let lhs_shape = Shape::new(nrows, inner);
                    let rhs_shape = Shape::new(inner, ncols);
                    pairs.push((lhs_shape, rhs_shape))
                }
            }
        }

        dispatch_binary! {{
            for &(lhs_shape, rhs_shape) in &pairs {
                let lhs = Matrix::<_, O>::from_value(lhs_shape, MockZeroSized::new()).unwrap();
                let rhs = Matrix::<_, P>::from_value(rhs_shape, MockZeroSized::new()).unwrap();
                Scope::with(|scope| {
                    let output = lhs.multiply(rhs).unwrap();
                    let expected_shape = Shape::new(lhs_shape.nrows, rhs_shape.ncols);
                    assert_eq!(output.shape(), expected_shape);
                    let expected_count = Count::expected(lhs_shape, rhs_shape);
                    assert_eq!(scope.init_count(), expected_count.init);
                    assert_eq!(scope.drop_count(), expected_count.drop);
                    assert_eq!(scope.add_count(), expected_count.add);
                    assert_eq!(scope.mul_count(), expected_count.mul);
                });
            }

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2], [3, 4], [5, 6]].with_order::<P>();
            let output = lhs.multiply(rhs).unwrap();
            let expected = matrix![[22, 28], [49, 64]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1], [2], [3]].with_order::<P>();
            let output = lhs.multiply(rhs).unwrap();
            let expected = matrix![[14], [32]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]].with_order::<P>();
            let output = lhs.multiply(rhs).unwrap();
            let expected = matrix![[30, 36, 42], [66, 81, 96]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2], [3, 4]].with_order::<P>();
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<P>();
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[0; 0]; 2].with_order::<O>();
            let rhs = matrix![[0; usize::MAX]; 0].with_order::<P>();
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let lhs = matrix![[0; 0]; 1].with_order::<O>();
            let rhs = matrix![[0; usize::MAX]; 0].with_order::<P>();
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_multiplication_like_operation() {
        fn dot_product(lhs: &[i32], rhs: &[i32]) -> i32 {
            lhs.iter()
                .zip(rhs)
                .map(|(lhs, rhs)| lhs * rhs)
                .reduce(|sum, product| sum + product)
                .unwrap()
        }

        dispatch_binary! {{
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2], [3, 4], [5, 6]].with_order::<P>();
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[22, 28], [49, 64]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1], [2], [3]].with_order::<P>();
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[14], [32]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]].with_order::<P>();
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[30, 36, 42], [66, 81, 96]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2], [3, 4]].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[0; 0]; 2].with_order::<O>();
            let rhs = matrix![[0; usize::MAX]; 0].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let lhs = matrix![[0; 0]; 1].with_order::<O>();
            let rhs = matrix![[0; usize::MAX]; 0].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_mul() {
        dispatch_binary! {{
            let lhs = matrix![
                [MockL(1), MockL(2), MockL(3)],
                [MockL(4), MockL(5), MockL(6)],
            ].with_order::<O>();
            let rhs = matrix![
                [MockR(1), MockR(2)],
                [MockR(3), MockR(4)],
                [MockR(5), MockR(6)],
            ].with_order::<P>();
            let expected = matrix![[MockU(22), MockU(28)], [MockU(49), MockU(64)]];

            {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                let output = lhs * rhs;
                assert_eq!(output, expected);
            }

            {
                let lhs = lhs.clone();
                let output = lhs * &rhs;
                assert_eq!(output, expected);
            }

            {
                let rhs = rhs.clone();
                let output = &lhs * rhs;
                assert_eq!(output, expected);
            }

            {
                let output = &lhs * &rhs;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_mul() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[2, 4, 6], [8, 10, 12]];

            {
                let matrix = matrix.clone();
                let output = matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.clone();
                let output = matrix * &scalar;
                assert_eq!(output, expected);
            }

            {
                let output = &matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let output = &matrix * &scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = matrix * &scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &matrix * &scalar;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_mul_rev() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[2, 4, 6], [8, 10, 12]];

            {
                let matrix = matrix.clone();
                let output = scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.clone();
                let output = &scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let output = scalar * &matrix;
                assert_eq!(output, expected);
            }

            {
                let output = &scalar * &matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = scalar * &matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &scalar * &matrix;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    fn test_primitive_scalar_mul_assign() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[2, 4, 6], [8, 10, 12]];

            {
                let mut matrix = matrix.clone();
                matrix *= scalar;
                assert_eq!(matrix, expected);
            }

            {
                let mut matrix = matrix.clone();
                matrix *= &scalar;
                assert_eq!(matrix, expected);
            }
        }}
    }

    #[derive(Debug)]
    struct Count {
        init: usize,
        drop: usize,
        add: usize,
        mul: usize,
    }

    impl Count {
        fn expected(lhs_shape: Shape, rhs_shape: Shape) -> Self {
            assert_eq!(lhs_shape.ncols, rhs_shape.nrows);

            let inner = lhs_shape.ncols;
            let output_shape = Shape::new(lhs_shape.nrows, rhs_shape.ncols);
            let lhs_size = lhs_shape.size().unwrap();
            let rhs_size = rhs_shape.size().unwrap();
            let output_size = output_shape.size().unwrap();

            let add;
            let mul;
            let default;
            let clone;
            if output_size == 0 {
                add = 0;
                mul = 0;
                default = 0;
                clone = 0;
            } else if inner == 0 {
                add = 0;
                mul = 0;
                default = output_size;
                clone = 0;
            } else {
                add = output_size * (inner - 1);
                mul = output_size * inner;
                default = 0;
                clone = mul * 2 - lhs_size - rhs_size;
            }

            let init = add + mul + default + clone;
            let drop = init + lhs_size + rhs_size - output_size;

            Self {
                init,
                drop,
                add,
                mul,
            }
        }
    }
}
