use super::Matrix;
use super::layout::{Layout, Order};
use crate::error::Result;
use crate::index::Index;
use crate::shape::AsShape;
use alloc::vec;
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
        let data = vec![value; size];
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
