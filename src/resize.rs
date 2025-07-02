use crate::Matrix;
use crate::error::Result;
use crate::shape::{AsShape, AxisShape};

impl<T> Matrix<T> {
    /// Resizes the matrix to the specified shape.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// Reducing the size does not automatically shrink the capacity.
    /// This choice is made to avoid potential reallocation. Consider
    /// explicitly calling [`shrink_to_fit`] if needed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::{Order, matrix};
    ///
    /// # fn main() -> Result<()> {
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let mut row_major = matrix.clone();
    /// row_major.set_order(Order::RowMajor);
    /// row_major.resize((2, 2))?;
    /// assert_eq!(row_major, matrix![[1, 2], [3, 4]]);
    /// row_major.resize((2, 3))?;
    /// assert_eq!(row_major, matrix![[1, 2, 3], [4, 0, 0]]);
    ///
    /// let mut col_major = matrix.clone();
    /// col_major.set_order(Order::ColMajor);
    /// col_major.resize((2, 2))?;
    /// assert_eq!(col_major, matrix![[1, 2], [4, 5]]);
    /// col_major.resize((2, 3))?;
    /// assert_eq!(col_major, matrix![[1, 2, 0], [4, 5, 0]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    /// [`shrink_to_fit`]: Matrix::shrink_to_fit
    pub fn resize<S>(&mut self, shape: S) -> Result<&mut Self>
    where
        T: Default,
        S: AsShape,
    {
        let shape = AxisShape::from_shape(shape, self.order);
        let size = shape.size::<T>()?;
        self.shape = shape;
        self.data.resize_with(size, T::default);
        Ok(self)
    }
}
