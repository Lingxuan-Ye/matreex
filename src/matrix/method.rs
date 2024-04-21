use super::dimension::Dimension;
use super::Matrix;
use crate::error::{Error, Result};
use crate::shape::{Shape, TryIntoShape};
use crate::MemoryLayout;

impl<T> Matrix<T> {
    pub fn shape(&self) -> Shape {
        self.dimension.to_shape(self.layout)
    }

    pub fn nrows(&self) -> usize {
        self.dimension.get_nrows(self.layout)
    }

    pub fn ncols(&self) -> usize {
        self.dimension.get_ncols(self.layout)
    }

    pub fn size(&self) -> usize {
        self.dimension.size()
    }

    pub fn layout(&self) -> MemoryLayout {
        self.layout
    }
}

impl<T> Matrix<T> {
    pub fn transpose(&mut self) -> &mut Self {
        self.layout = !self.layout;
        self
    }

    pub fn switch_layout(&mut self) -> &mut Self {
        // should rearrange self.data when implement it
        unimplemented!();
    }

    pub fn set_layout(&mut self, layout: MemoryLayout) -> &mut Self {
        if layout != self.layout {
            self.switch_layout();
        }
        self
    }

    pub fn reshape<S: TryIntoShape>(&mut self, shape: S) -> Result<&mut Self> {
        let shape = shape.try_into_shape()?;
        if shape.size() != self.data.len() {
            return Err(Error::SizeMismatch);
        }
        self.dimension = Dimension::from_shape(shape, self.layout);
        Ok(self)
    }
}

impl<T: Default> Matrix<T> {
    pub fn resize<S: TryIntoShape>(&mut self, shape: S) -> Result<&mut Self> {
        let shape = shape.try_into_shape()?;
        let size = Self::check_size(&shape)?;
        self.data.resize_with(size, Default::default);
        self.dimension = Dimension::from_shape(shape, self.layout);
        Ok(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_reshape() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.reshape((2, 3)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.reshape((3, 2)).unwrap();
        assert_eq!(matrix, matrix![[0, 1], [2, 3], [4, 5]]);

        matrix.reshape((1, 6)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2, 3, 4, 5]]);

        matrix.reshape((6, 1)).unwrap();
        assert_eq!(matrix, matrix![[0], [1], [2], [3], [4], [5]]);

        matrix.reshape((2, 3)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        assert_eq!(
            matrix.reshape((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        assert_eq!(matrix.reshape((2, 2)).unwrap_err(), Error::SizeMismatch);
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_resize() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.resize((2, 3)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.resize((2, 2)).unwrap();
        assert_eq!(matrix, matrix![[0, 1], [2, 3]]);

        matrix.resize((3, 3)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 0, 0], [0, 0, 0]]);

        matrix.resize((2, 3)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 0, 0]]);

        assert_eq!(
            matrix.resize((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 0, 0]]);
    }
}