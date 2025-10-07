use super::Matrix;
use super::layout::{Layout, Order};
use crate::error::Result;
use crate::index::Index;
use crate::shape::AsShape;
use alloc::vec::Vec;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn new() -> Self {
        let layout = Layout::default();
        let data = Vec::new();
        Self { layout, data }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let layout = Layout::default();
        let data = Vec::with_capacity(capacity);
        Self { layout, data }
    }

    pub fn with_default<S>(shape: S) -> Result<Self>
    where
        S: AsShape,
        T: Default,
    {
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.resize_with(size, T::default);
        Ok(Self { layout, data })
    }

    pub fn with_value<S>(shape: S, value: T) -> Result<Self>
    where
        S: AsShape,
        T: Clone,
    {
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let data = alloc::vec![value; size];
        Ok(Self { layout, data })
    }

    pub fn with_initializer<S, F>(shape: S, mut initializer: F) -> Result<Self>
    where
        S: AsShape,
        F: FnMut(Index) -> T,
    {
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let stride = layout.stride();
        let mut data = Vec::with_capacity(size);
        for index in 0..size {
            let index = Index::from_flattened::<O>(index, stride);
            let element = initializer(index);
            data.push(element);
        }
        Ok(Self { layout, data })
    }
}

impl<T, O> Default for Matrix<T, O>
where
    O: Order,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::index::Index;
    use crate::shape::Shape;
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_new() {
        dispatch_unary! {{
            let matrix = Matrix::<u8, O>::new();
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
        }}
    }

    #[test]
    fn test_with_capacity() {
        dispatch_unary! {{
            let matrix = Matrix::<u8, O>::with_capacity(10);
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
            assert!(matrix.capacity() >= 10);
        }}
    }

    #[test]
    fn test_with_default() {
        dispatch_unary! {{
            let shape = Shape::new(2, 3);
            let matrix = Matrix::<u8, O>::with_default(shape).unwrap();
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(&matrix, &expected);

            let shape = Shape::new(2, usize::MAX);
            let error = Matrix::<u8, O>::with_default(shape).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let shape = Shape::new(1, isize::MAX as usize + 1);
            let error = Matrix::<u8, O>::with_default(shape).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);

            // Unable to cover.
            // let shape = Shape::new(1, isize::MAX as usize + 1);
            // ssert!(Matrix::<(), O>::with_default(shape).is_ok());
        }}
    }

    #[test]
    fn test_with_value() {
        dispatch_unary! {{
            let shape = Shape::new(2, 3);
            let matrix = Matrix::<u8, O>::with_value(shape, 0).unwrap();
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(&matrix, &expected);

            let shape = Shape::new(2, usize::MAX);
            let error = Matrix::<u8, O>::with_value(shape, 0).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let shape = Shape::new(1, isize::MAX as usize + 1);
            let error = Matrix::<u8, O>::with_value(shape, 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);

            let shape = Shape::new(1, isize::MAX as usize + 1);
            assert!(Matrix::<(), O>::with_value(shape, ()).is_ok());
        }}
    }

    #[test]
    fn test_with_initializer() {
        dispatch_unary! {{
            let shape = Shape::new(2, 3);
            let matrix = Matrix::<Index, O>::with_initializer(shape, |index| index).unwrap();
            let expected = matrix![
                [Index::new(0, 0), Index::new(0, 1), Index::new(0, 2)],
                [Index::new(1, 0), Index::new(1, 1), Index::new(1, 2)],
            ];
            assert_eq!(&matrix, &expected);

            // Assert no panic from unflattening indices occurs.
            let shape = Shape::new(2, 0);
            let matrix = Matrix::<Index, O>::with_initializer(shape, |index| index).unwrap();
            let expected = matrix![[Index::default(); 0]; 2];
            assert_eq!(&matrix, &expected);

            // Assert no panic from unflattening indices occurs.
            let shape = Shape::new(0, 3);
            let matrix = Matrix::<Index, O>::with_initializer(shape, |index| index).unwrap();
            let expected = matrix![[Index::default(); 3]; 0];
            assert_eq!(&matrix, &expected);

            let shape = Shape::new(2, usize::MAX);
            let error = Matrix::<u8, O>::with_initializer(shape, |_| 0).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let shape = Shape::new(1, isize::MAX as usize+ 1);
            let error = Matrix::<u8, O>::with_initializer(shape, |_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);

            // Unable to cover.
            // let shape = Shape::new(1, isize::MAX as usize + 1);
            // assert!(Matrix::<(), O>::with_initializer(shape, |_| ()).is_ok());
        }}
    }

    #[test]
    fn test_default() {
        dispatch_unary! {{
            let matrix = Matrix::<u8, O>::default();
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
        }}
    }
}
