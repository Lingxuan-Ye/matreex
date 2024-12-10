use super::index::Index;
use super::order::Order;
use super::Matrix;
use crate::error::{Error, Result};

impl<T> Matrix<T> {
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
}
