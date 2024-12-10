use super::index::Index;
use super::order::Order;
use super::Matrix;
use crate::error::{Error, Result};

impl<T> Matrix<T> {
    /// Swaps the elements at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error};
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.swap((0, 0), (1, 1))?;
    /// assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);
    ///
    /// let result = matrix.swap((0, 0), (2, 2));
    /// assert_eq!(result, Err(Error::IndexOutOfBounds));
    /// # Ok(())
    /// # }
    /// ```
    pub fn swap<I, J>(&mut self, index: I, jndex: J) -> Result<&mut Self>
    where
        I: Index,
        J: Index,
    {
        let (index, jndex) = match self.order {
            Order::RowMajor => {
                if index.row() >= self.major()
                    || index.col() >= self.minor()
                    || jndex.row() >= self.major()
                    || jndex.col() >= self.minor()
                {
                    return Err(Error::IndexOutOfBounds);
                }
                (
                    index.row() * self.major_stride() + index.col(),
                    jndex.row() * self.major_stride() + jndex.col(),
                )
            }
            Order::ColMajor => {
                if index.row() >= self.minor()
                    || index.col() >= self.major()
                    || jndex.row() >= self.minor()
                    || jndex.col() >= self.major()
                {
                    return Err(Error::IndexOutOfBounds);
                }
                (
                    index.col() * self.minor_stride() + index.row(),
                    jndex.col() * self.minor_stride() + jndex.row(),
                )
            }
        };
        self.data.swap(index, jndex);
        Ok(self)
    }

    pub fn swap_rows(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match self.order {
            Order::RowMajor => self.swap_major_axis_vectors(m, n),
            Order::ColMajor => self.swap_minor_axis_vectors(m, n),
        }
    }

    pub fn swap_cols(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match self.order {
            Order::RowMajor => self.swap_minor_axis_vectors(m, n),
            Order::ColMajor => self.swap_major_axis_vectors(m, n),
        }
    }
}

impl<T> Matrix<T> {
    fn swap_major_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.major() || n >= self.major() {
            return Err(Error::IndexOutOfBounds);
        }
        let mut index = m * self.major_stride();
        let mut jndex = n * self.major_stride();
        for _ in 0..self.minor() {
            self.data.swap(index, jndex);
            index += 1;
            jndex += 1;
        }
        Ok(self)
    }

    fn swap_minor_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.minor() || n >= self.minor() {
            return Err(Error::IndexOutOfBounds);
        }
        let mut index = m;
        let mut jndex = n;
        for _ in 0..self.major() {
            self.data.swap(index, jndex);
            index += self.major_stride();
            jndex += self.major_stride();
        }
        Ok(self)
    }
}
