use super::Matrix;
use super::layout::{Layout, LayoutIndex, Order, OrderKind};
use crate::error::{Error, Result};
use crate::shape::Shape;
use alloc::vec::Vec;
use core::ptr;

mod add;
mod div;
mod mul;
mod neg;
mod rem;
mod sub;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn scalar_operation<'a, 'b, S, F, U>(
        &'a self,
        scalar: &'b S,
        mut op: F,
    ) -> Result<Matrix<U, O>>
    where
        F: FnMut(&'a T, &'b S) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self
            .data
            .iter()
            .map(|element| op(element, scalar))
            .collect();
        Ok(Matrix { layout, data })
    }

    pub fn scalar_operation_consume_self<'a, S, F, U>(
        self,
        scalar: &'a S,
        mut op: F,
    ) -> Result<Matrix<U, O>>
    where
        F: FnMut(T, &'a S) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self
            .data
            .into_iter()
            .map(|element| op(element, scalar))
            .collect();
        Ok(Matrix { layout, data })
    }

    pub fn scalar_operation_assign<'a, S, F>(&mut self, scalar: &'a S, mut op: F) -> &mut Self
    where
        F: FnMut(&mut T, &'a S),
    {
        self.data.iter_mut().for_each(|element| op(element, scalar));
        self
    }
}

impl<L, LO> Matrix<L, LO> where LO: Order {}

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    pub fn elementwise_operation<'a, 'b, R, F, U, RO>(
        &'a self,
        rhs: &'b Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        F: FnMut(&'a L, &'b R) -> U,
        RO: Order,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = LayoutIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    pub fn elementwise_operation_consume_self<'a, R, F, U, RO>(
        self,
        rhs: &'a Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        F: FnMut(L, &'a R) -> U,
        RO: Order,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .into_iter()
                .zip(&rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = LayoutIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    pub fn elementwise_operation_consume_rhs<'a, R, F, U, RO>(
        &'a self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        F: FnMut(&'a L, R) -> U,
        RO: Order,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .iter()
                .zip(rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            let mut rhs = rhs;
            unsafe { rhs.data.set_len(0) };
            let rhs_base = rhs.data.as_ptr();
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = LayoutIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { ptr::read(rhs_base.add(index)) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    pub fn elementwise_operation_consume_both<R, F, U, RO>(
        self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        F: FnMut(L, R) -> U,
        RO: Order,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .into_iter()
                .zip(rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            let mut rhs = rhs;
            unsafe { rhs.data.set_len(0) };
            let rhs_base = rhs.data.as_ptr();
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = LayoutIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { ptr::read(rhs_base.add(index)) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    pub fn elementwise_operation_assign<'a, R, F, RO>(
        &mut self,
        rhs: &'a Matrix<R, RO>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut(&mut L, &'a R),
        RO: Order,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        if LO::KIND == RO::KIND {
            self.data
                .iter_mut()
                .zip(&rhs.data)
                .for_each(|(left, right)| op(left, right));
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = LayoutIndex::from_flattened(index, lhs_stride)
                    .swap()
                    .to_flattened(rhs_stride);
                let right = unsafe { rhs.data.get_unchecked(index) };
                op(left, right)
            });
        }

        Ok(self)
    }

    pub fn elementwise_operation_assign_consume_rhs<R, F, RO>(
        &mut self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut(&mut L, R),
        RO: Order,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        if LO::KIND == RO::KIND {
            self.data
                .iter_mut()
                .zip(rhs.data)
                .for_each(|(left, right)| op(left, right));
        } else {
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = LayoutIndex::from_flattened(index, lhs_stride)
                    .swap()
                    .to_flattened(rhs_stride);
                let right = unsafe { ptr::read(rhs_base.add(index)) };
                op(left, right)
            });
        }

        Ok(self)
    }

    fn ensure_elementwise_operation_conformable<R, RO>(&self, rhs: &Matrix<R, RO>) -> Result<&Self>
    where
        RO: Order,
    {
        if self.shape() == rhs.shape() {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }
}

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    pub fn multiplication_like_operation<R, F, U, RO>(
        self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        F: FnMut(&[L], &[R]) -> U,
        U: Default,
        RO: Order,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);

        if self.ncols() == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { layout, data });
        }

        let lhs = self.into_row_major();
        let rhs = rhs.into_col_major();

        match LO::KIND {
            OrderKind::RowMajor => {
                for row in 0..nrows {
                    for col in 0..ncols {
                        let lhs = unsafe { lhs.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }

            OrderKind::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let lhs = unsafe { lhs.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { layout, data })
    }

    fn ensure_multiplication_like_operation_conformable<R, RO>(
        &self,
        rhs: &Matrix<R, RO>,
    ) -> Result<&Self>
    where
        RO: Order,
    {
        if self.ncols() == rhs.nrows() {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    #[inline(always)]
    unsafe fn get_nth_major_axis_vector_unchecked(&self, n: usize) -> &[T] {
        let stride = self.stride();
        let lower = n * stride.major();
        let upper = lower + stride.major();
        unsafe { self.data.get_unchecked(lower..upper) }
    }
}
