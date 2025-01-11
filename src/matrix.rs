//! Defines [`Matrix<T>`] and all its related components.

use self::index::transpose_flattened_index;
use self::order::Order;
use self::shape::{AxisShape, Shape};
use crate::error::{Error, Result};
use std::cmp::min;
use std::mem::size_of;

pub mod index;
pub mod iter;
pub mod order;
pub mod shape;

mod arithmetic;
mod construct;
mod convert;
mod fmt;
mod swap;

#[cfg(feature = "rayon")]
mod parallel;

/// [`Matrix<T>`] means matrix.
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    order: Order,
    shape: AxisShape,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Returns the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Order};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.order(), Order::default());
    /// ```
    #[inline]
    pub fn order(&self) -> Order {
        self.order
    }

    /// Returns the shape of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Shape};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let shape = matrix.shape();
    /// assert_eq!(shape.nrows(), 2);
    /// assert_eq!(shape.ncols(), 3);
    /// ```
    #[inline]
    pub fn shape(&self) -> Shape {
        self.shape.to_shape(self.order)
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.nrows(), 2);
    /// ```
    #[inline]
    pub fn nrows(&self) -> usize {
        self.shape.nrows(self.order)
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.ncols(), 3);
    /// ```
    #[inline]
    pub fn ncols(&self) -> usize {
        self.shape.ncols(self.order)
    }

    /// Returns the total number of elements in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.size(), 6);
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the matrix contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    ///
    /// let matrix = Matrix::<i32>::new();
    /// assert!(matrix.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the capacity of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    ///
    /// let mut matrix = Matrix::<i32>::with_capacity(10);
    /// assert!(matrix.capacity() >= 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns the length of the major axis.
    fn major(&self) -> usize {
        self.shape.major()
    }

    /// Returns the length of the minor axis.
    fn minor(&self) -> usize {
        self.shape.minor()
    }

    /// Returns the stride of the major axis.
    fn major_stride(&self) -> usize {
        self.shape.major_stride()
    }

    /// Returns the stride of the minor axis.
    #[allow(dead_code)]
    const fn minor_stride(&self) -> usize {
        self.shape.minor_stride()
    }
}

impl<T> Matrix<T> {
    /// Transposes the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix.transpose();
    ///
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 3);
    ///
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 1);
    /// assert_eq!(matrix[(1, 1)], 4);
    ///
    /// // row 2
    /// assert_eq!(matrix[(2, 0)], 2);
    /// assert_eq!(matrix[(2, 1)], 5);
    /// ```
    pub fn transpose(&mut self) -> &mut Self {
        let size = self.size();
        let mut visited = vec![false; size];

        for index in 0..size {
            if visited[index] {
                continue;
            }
            let mut current = index;
            while !visited[current] {
                visited[current] = true;
                let next = transpose_flattened_index(current, self.shape);
                self.data.swap(index, next);
                current = next;
            }
        }

        self.shape.transpose();
        self
    }

    /// Switches the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let order = matrix.order();
    ///
    /// matrix.switch_order();
    /// assert_ne!(matrix.order(), order);
    ///
    /// matrix.switch_order();
    /// assert_eq!(matrix.order(), order);
    /// ```
    #[inline]
    pub fn switch_order(&mut self) -> &mut Self {
        self.transpose();
        self.order.switch();
        self
    }

    /// Switches the order of the matrix without rearranging the underlying
    /// data. As a result, the matrix appears transposed when accessed.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let order = matrix.order();
    /// matrix.switch_order_without_rearrangement();
    /// assert_ne!(matrix.order(), order);
    ///
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 3);
    ///
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 1);
    /// assert_eq!(matrix[(1, 1)], 4);
    ///
    /// // row 2
    /// assert_eq!(matrix[(2, 0)], 2);
    /// assert_eq!(matrix[(2, 1)], 5);
    /// ```
    #[inline]
    pub fn switch_order_without_rearrangement(&mut self) -> &mut Self {
        self.order.switch();
        self
    }

    /// Sets the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Order};
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.set_order(Order::RowMajor);
    /// assert_eq!(matrix.order(), Order::RowMajor);
    ///
    /// matrix.set_order(Order::ColMajor);
    /// assert_eq!(matrix.order(), Order::ColMajor);
    /// ```
    #[inline]
    pub fn set_order(&mut self, order: Order) -> &mut Self {
        if order != self.order {
            self.switch_order();
        }
        self
    }

    /// Sets the order of the matrix without rearranging the underlying
    /// data. As a result, when the order is changed, the matrix appears
    /// transposed when accessed.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let mut order = matrix.order();
    /// order.switch();
    /// matrix.set_order_without_rearrangement(order);
    /// assert_eq!(matrix.order(), order);
    ///
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 3);
    ///
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 1);
    /// assert_eq!(matrix[(1, 1)], 4);
    ///
    /// // row 2
    /// assert_eq!(matrix[(2, 0)], 2);
    /// assert_eq!(matrix[(2, 1)], 5);
    /// ```
    #[inline]
    pub fn set_order_without_rearrangement(&mut self, order: Order) -> &mut Self {
        if order != self.order {
            self.switch_order_without_rearrangement();
        }
        self
    }

    /// Resizes the matrix to the specified shape.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if total bytes stored exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// Reducing the size does not automatically shrink the capacity.
    /// This choice is made to avoid potential reallocation. Consider
    /// explicitly calling [`Matrix::shrink_to_fit`] if needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.resize((2, 2))?;
    /// assert_eq!(matrix, matrix![[0, 1], [2, 3]]);
    ///
    /// matrix.resize((2, 3))?;
    /// assert_eq!(matrix, matrix![[0, 1, 2], [3, 0, 0]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn resize<S>(&mut self, shape: S) -> Result<&mut Self>
    where
        T: Default,
        S: Into<Shape>,
    {
        let shape = shape.into().try_to_axis_shape(self.order)?;
        let size = Self::check_size(shape.size())?;
        self.shape = shape;
        self.data.resize_with(size, T::default);
        Ok(self)
    }

    /// Reshapes the matrix to the specified shape.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeMismatch`] if the size of the new shape does not
    ///   match the current size of the matrix.
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
    /// matrix.reshape((3, 2))?;
    /// assert_eq!(matrix, matrix![[0, 1], [2, 3], [4, 5]]);
    ///
    /// let result = matrix.reshape((2, 2));
    /// assert_eq!(result, Err(Error::SizeMismatch));
    /// # Ok(())
    /// # }
    /// ```
    pub fn reshape<S>(&mut self, shape: S) -> Result<&mut Self>
    where
        S: Into<Shape>,
    {
        let Ok(shape) = shape.into().try_to_axis_shape(self.order) else {
            return Err(Error::SizeMismatch);
        };
        if self.size() != shape.size() {
            return Err(Error::SizeMismatch);
        }
        self.shape = shape;
        Ok(self)
    }

    /// Shrinks the capacity of the matrix as much as possible.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert!(matrix.capacity() >= 6);
    ///
    /// matrix.resize((1, 3))?;
    /// assert!(matrix.capacity() >= 6);
    ///
    /// matrix.shrink_to_fit();
    /// assert!(matrix.capacity() >= 3);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) -> &mut Self {
        self.data.shrink_to_fit();
        self
    }

    /// Shrinks the capacity of the matrix with a lower bound.
    ///
    /// The capacity will remain at least as large as both the size
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit,
    /// this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert!(matrix.capacity() >= 6);
    ///
    /// matrix.resize((1, 3))?;
    /// assert!(matrix.capacity() >= 6);
    ///
    /// matrix.shrink_to(4);
    /// assert!(matrix.capacity() >= 4);
    ///
    /// matrix.shrink_to(0);
    /// assert!(matrix.capacity() >= 3);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) -> &mut Self {
        self.data.shrink_to(min_capacity);
        self
    }

    /// Overwrites the overlapping part of this matrix with `source`,
    /// leaving the non-overlapping part unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 0, 0], [0, 0, 0]];
    /// let source = matrix![[1, 1], [1, 1], [1, 1]];
    /// matrix.overwrite(&source);
    /// assert_eq!(matrix, matrix![[1, 1, 0], [1, 1, 0]]);
    /// ```
    pub fn overwrite(&mut self, source: &Self) -> &mut Self
    where
        T: Clone,
    {
        if self.order == source.order {
            let major = min(self.major(), source.major());
            let minor = min(self.minor(), source.minor());
            for i in 0..major {
                let self_lower = i * self.major_stride();
                let self_upper = self_lower + minor;
                let source_lower = i * source.major_stride();
                let source_upper = source_lower + minor;
                unsafe {
                    self.data
                        .get_unchecked_mut(self_lower..self_upper)
                        .clone_from_slice(source.data.get_unchecked(source_lower..source_upper));
                }
            }
        } else {
            let major = min(self.major(), source.minor());
            let minor = min(self.minor(), source.major());
            for i in 0..major {
                let self_lower = i * self.major_stride();
                let self_upper = self_lower + minor;
                unsafe {
                    self.data
                        .get_unchecked_mut(self_lower..self_upper)
                        .iter_mut()
                        .zip(source.iter_nth_minor_axis_vector_unchecked(i))
                        .for_each(|(x, y)| *x = y.clone());
                }
            }
        }
        self
    }

    /// Applies a closure to each element of the matrix,
    /// modifying the matrix in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix.apply(|x| *x += 1);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    pub fn apply<F>(&mut self, f: F) -> &mut Self
    where
        F: FnMut(&mut T),
    {
        self.data.iter_mut().for_each(f);
        self
    }

    /// Applies a closure to each element of the matrix,
    /// returning a new matrix with the results.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];
    /// let matrix_f64 = matrix_i32.map(|x| x as f64);
    /// assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// ```
    #[inline]
    pub fn map<U, F>(self, f: F) -> Matrix<U>
    where
        F: FnMut(T) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self.data.into_iter().map(f).collect();
        Matrix { order, shape, data }
    }

    #[inline]
    pub fn map_ref<U, F>(&self, f: F) -> Matrix<U>
    where
        F: FnMut(&T) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self.data.iter().map(f).collect();
        Matrix { order, shape, data }
    }
}

impl<T> Matrix<T> {
    fn check_size(size: usize) -> Result<usize> {
        // see more info at https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.with_capacity
        const MAX: usize = isize::MAX as usize;
        match size_of::<T>().checked_mul(size) {
            Some(0..=MAX) => Ok(size),
            _ => Err(Error::CapacityOverflow),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_transpose() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let order = matrix.order;
        let major = matrix.major();
        let minor = matrix.minor();

        matrix.transpose();

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 3);

            // row 1
            assert_eq!(matrix[(1, 0)], 1);
            assert_eq!(matrix[(1, 1)], 4);

            // row 2
            assert_eq!(matrix[(2, 0)], 2);
            assert_eq!(matrix[(2, 1)], 5);

            assert_eq!(matrix.order, order);
            assert_eq!(matrix.major(), minor);
            assert_eq!(matrix.minor(), major);
        }

        matrix.transpose();

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_eq!(matrix.order, order);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }
    }

    #[test]
    fn test_switch_order() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let order = matrix.order;
        let major = matrix.major();
        let minor = matrix.minor();

        matrix.switch_order();

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_ne!(matrix.order, order);
            assert_eq!(matrix.major(), minor);
            assert_eq!(matrix.minor(), major);
        }

        matrix.switch_order();

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_eq!(matrix.order, order);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }
    }

    #[test]
    fn test_switch_order_without_rearrangement() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let order = matrix.order;
        let major = matrix.major();
        let minor = matrix.minor();

        matrix.switch_order_without_rearrangement();

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 3);

            // row 1
            assert_eq!(matrix[(1, 0)], 1);
            assert_eq!(matrix[(1, 1)], 4);

            // row 2
            assert_eq!(matrix[(2, 0)], 2);
            assert_eq!(matrix[(2, 1)], 5);

            assert_ne!(matrix.order, order);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }

        matrix.switch_order_without_rearrangement();

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_eq!(matrix.order, order);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }
    }

    #[test]
    fn test_set_order() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let major = matrix.major();
        let minor = matrix.minor();

        matrix.set_order(Order::RowMajor);

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_eq!(matrix.order, Order::RowMajor);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }

        matrix.set_order(Order::ColMajor);

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_eq!(matrix.order, Order::ColMajor);
            assert_eq!(matrix.major(), minor);
            assert_eq!(matrix.minor(), major);
        }
    }

    #[test]
    fn test_set_order_without_rearrangement() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let mut order = matrix.order();
        let major = matrix.major();
        let minor = matrix.minor();

        matrix.set_order_without_rearrangement(order);

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 1);
            assert_eq!(matrix[(0, 2)], 2);

            // row 1
            assert_eq!(matrix[(1, 0)], 3);
            assert_eq!(matrix[(1, 1)], 4);
            assert_eq!(matrix[(1, 2)], 5);

            assert_eq!(matrix.order, order);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }

        order.switch();
        matrix.set_order_without_rearrangement(order);

        {
            // row 0
            assert_eq!(matrix[(0, 0)], 0);
            assert_eq!(matrix[(0, 1)], 3);

            // row 1
            assert_eq!(matrix[(1, 0)], 1);
            assert_eq!(matrix[(1, 1)], 4);

            // row 2
            assert_eq!(matrix[(2, 0)], 2);
            assert_eq!(matrix[(2, 1)], 5);

            assert_eq!(matrix.order, order);
            assert_eq!(matrix.major(), major);
            assert_eq!(matrix.minor(), minor);
        }
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

        matrix.resize((2, 0)).unwrap();
        assert_eq!(matrix, matrix![[], []]);

        let unchanged = matrix.clone();

        let error = matrix.resize((usize::MAX, 2)).unwrap_err();
        assert_eq!(error, Error::SizeOverflow);
        assert_eq!(matrix, unchanged);

        let error = matrix.resize((isize::MAX as usize + 1, 1)).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);
        assert_eq!(matrix, unchanged);
    }

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

        let unchanged = matrix.clone();

        let error = matrix.reshape((2, 2)).unwrap_err();
        assert_eq!(error, Error::SizeMismatch);
        assert_eq!(matrix, unchanged);

        let error = matrix.reshape((usize::MAX, 2)).unwrap_err();
        assert_eq!(error, Error::SizeMismatch);
        assert_eq!(matrix, unchanged);

        let error = matrix.reshape((isize::MAX as usize + 1, 1)).unwrap_err();
        assert_eq!(error, Error::SizeMismatch);
        assert_eq!(matrix, unchanged);
    }

    #[test]
    fn test_overwrite() {
        let blank = matrix![[0, 0, 0], [0, 0, 0]];

        {
            let mut source = matrix![[1, 2]];

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 0], [0, 0, 0]]);

            source.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 0], [0, 0, 0]]);
        }

        {
            let mut source = matrix![[1, 2], [3, 4]];

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);

            source.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);
        }

        {
            let mut source = matrix![[1, 2], [3, 4], [5, 6]];

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);

            source.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);
        }

        {
            let mut source = matrix![[1, 2, 3]];

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);

            source.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);
        }

        {
            let mut source = matrix![[1, 2, 3, 4]];

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);

            source.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite(&source);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);
        }
    }

    #[test]
    fn test_apply() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.apply(|x| *x += 2);
        assert_eq!(matrix, matrix![[2, 3, 4], [5, 6, 7]]);

        matrix.switch_order();

        matrix.apply(|x| *x -= 2);
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_map() {
        let matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];

        let mut matrix_f64 = matrix_i32.map(|x| x as f64);
        assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        matrix_f64.switch_order();

        let mut matrix_i32 = matrix_f64.map(|x| x as i32);
        matrix_i32.switch_order();
        assert_eq!(matrix_i32, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_map_ref() {
        let mut matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];

        let matrix_f64 = matrix_i32.map_ref(|x| *x as f64);
        assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        matrix_i32.switch_order();

        let mut matrix_f64 = matrix_i32.map_ref(|x| *x as f64);
        matrix_f64.switch_order();
        assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    }
}
