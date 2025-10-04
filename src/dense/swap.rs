use super::Matrix;
use super::layout::{Order, OrderKind};
use crate::error::{Error, Result};
use crate::index::MatrixIndex;
use core::ptr;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn swap<I, J>(&mut self, i: I, j: J) -> Result<&mut Self>
    where
        I: MatrixIndex<Self, Output = T>,
        J: MatrixIndex<Self, Output = T>,
    {
        let x = self.get_mut(i)? as *mut T;
        let y = self.get_mut(j)? as *mut T;

        if x == y {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let x = base.with_addr(x.addr());
        let y = base.with_addr(y.addr());

        unsafe {
            ptr::swap_nonoverlapping(x, y, 1);
        }

        Ok(self)
    }

    pub fn swap_rows(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match O::KIND {
            OrderKind::RowMajor => self.swap_major_axis_vectors(m, n),
            OrderKind::ColMajor => self.swap_minor_axis_vectors(m, n),
        }
    }

    pub fn swap_cols(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match O::KIND {
            OrderKind::RowMajor => self.swap_minor_axis_vectors(m, n),
            OrderKind::ColMajor => self.swap_major_axis_vectors(m, n),
        }
    }

    fn swap_major_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.major() || n >= self.major() {
            return Err(Error::IndexOutOfBounds);
        } else if m == n || self.minor() == 0 {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let index = m * stride.major();
        let jndex = n * stride.major();

        unsafe {
            let x = base.add(index);
            let y = base.add(jndex);
            let count = self.minor() * stride.minor();
            ptr::swap_nonoverlapping(x, y, count);
        }

        Ok(self)
    }

    fn swap_minor_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.minor() || n >= self.minor() {
            return Err(Error::IndexOutOfBounds);
        } else if m == n || self.major() == 0 {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let index = m * stride.minor();
        let jndex = n * stride.minor();

        unsafe {
            let mut x = base.add(index);
            let mut y = base.add(jndex);
            ptr::swap_nonoverlapping(x, y, stride.minor());
            for _ in 1..self.major() {
                x = x.add(stride.major());
                y = y.add(stride.major());
                ptr::swap_nonoverlapping(x, y, stride.minor());
            }
        }

        Ok(self)
    }
}
