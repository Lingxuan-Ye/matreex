//! A simple matrix implementation.
//!
//! # Quick Start
//!
//! First, we need to import [`matrix!`].
//!
//! ```
//! use matreex::matrix;
//! ```
//!
//! ## Addition
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[1, 2, 3], [4, 5, 6]];
//! let rhs = matrix![[2, 2, 2], [2, 2, 2]];
//! assert_eq!(lhs + rhs, matrix![[3, 4, 5], [6, 7, 8]]);
//! ```
//!
//! ## Subtraction
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[1, 2, 3], [4, 5, 6]];
//! let rhs = matrix![[2, 2, 2], [2, 2, 2]];
//! assert_eq!(lhs - rhs, matrix![[-1, 0, 1], [2, 3, 4]]);
//! ```
//!
//! ## Multiplication
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[1, 2, 3], [4, 5, 6]];
//! let rhs = matrix![[1, 2], [3, 4], [5, 6]];
//! assert_eq!(lhs * rhs, matrix![[22, 28], [49, 64]]);
//! ```
//!
//! ## Division
//!
//! ```compile_fail
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
//! assert_eq!(lhs / rhs, matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);
//! ```
//!
//! Wait, matrix division isn't well-defined, remember? It won't compile.
//! But don't worry, you might just need to perform elementwise division:
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
//! assert_eq!(lhs.elementwise_div(&rhs), Ok(matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]));
//! ```
//!
//! Or scalar division:
//!
//! ```
//! # use matreex::matrix;
//! #
//! let matrix = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! assert_eq!(matrix / 2.0, matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);
//!
//! let matrix = matrix![[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]];
//! assert_eq!(2.0 / matrix, matrix![[2.0, 1.0, 0.5], [0.25, 0.125, 0.0625]]);
//! ```
//!
//! Or maybe the inverse of a matrix?
//!
//! Nah, we don't have that yet.
//!
//! # FAQs
//!
//! ## Why named `matreex`?
//!
//! Hmm ... Who knows? Could be a name conflict.
//!
//! ## Is it `no_std` compatible?
//!
//! This crate is `no_std` compatible if the `parallel` feature is not enabled.

#![no_std]

extern crate alloc;

pub use self::error::{Error, Result};
pub use self::index::{Index, WrappingIndex};
pub use self::order::Order;
pub use self::shape::Shape;

use self::index::AxisIndex;
use self::shape::AxisShape;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp;
use core::ptr;

pub mod error;
pub mod index;
pub mod iter;
pub mod order;
pub mod shape;

#[cfg(feature = "parallel")]
pub mod parallel;

mod arithmetic;
mod construct;
mod convert;
mod eq;
mod fmt;
mod hash;
mod macros;
mod swap;

#[cfg(feature = "serde")]
mod deserialize;

/// [`Matrix<T>`] means matrix.
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Clone)]
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
    /// use matreex::{Order, matrix};
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(matrix.order(), Order::RowMajor);
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
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
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
    ///
    /// It always returns `1`.
    fn minor_stride(&self) -> usize {
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// matrix.transpose();
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    ///
    /// matrix.transpose();
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    pub fn transpose(&mut self) -> &mut Self {
        if size_of::<T>() == 0 {
            self.shape.transpose();
            return self;
        }

        let base = self.data.as_mut_ptr();
        let old_shape = self.shape;
        self.shape.transpose();
        let new_shape = self.shape;
        let size = self.size();
        let mut visited = vec![false; size];

        for index in 0..size {
            let mut current = index;
            loop {
                let state = unsafe { visited.get_unchecked_mut(current) };
                if *state {
                    break;
                }
                *state = true;
                let next = AxisIndex::from_flattened(current, old_shape)
                    .swap()
                    .to_flattened(new_shape);
                unsafe {
                    let x = base.add(index);
                    let y = base.add(next);
                    ptr::swap(x, y);
                }
                current = next;
            }
        }

        self
    }

    /// Switches the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let order = matrix.order();
    ///
    /// matrix.switch_order();
    /// assert_ne!(matrix.order(), order);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    ///
    /// matrix.switch_order();
    /// assert_eq!(matrix.order(), order);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let order = matrix.order();
    ///
    /// matrix.switch_order_without_rearrangement();
    /// assert_ne!(matrix.order(), order);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    ///
    /// matrix.switch_order_without_rearrangement();
    /// assert_eq!(matrix.order(), order);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
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
    /// use matreex::{Order, matrix};
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// matrix.set_order(Order::RowMajor);
    /// assert_eq!(matrix.order(), Order::RowMajor);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    ///
    /// matrix.set_order(Order::ColMajor);
    /// assert_eq!(matrix.order(), Order::ColMajor);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
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
    /// use matreex::{Order, matrix};
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.set_order(Order::RowMajor);
    ///
    /// matrix.set_order_without_rearrangement(Order::RowMajor);
    /// assert_eq!(matrix.order(), Order::RowMajor);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    ///
    /// matrix.set_order_without_rearrangement(Order::ColMajor);
    /// assert_eq!(matrix.order(), Order::ColMajor);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    /// ```
    #[inline]
    pub fn set_order_without_rearrangement(&mut self, order: Order) -> &mut Self {
        if order != self.order {
            self.switch_order_without_rearrangement();
        }
        self
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
    /// # use matreex::Result;
    /// use matreex::{Order, matrix};
    ///
    /// # fn main() -> Result<()> {
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let mut row_major = matrix.clone();
    /// row_major.set_order(Order::RowMajor);
    /// row_major.reshape((3, 2))?;
    /// assert_eq!(row_major, matrix![[1, 2], [3, 4], [5, 6]]);
    ///
    /// let mut col_major = matrix.clone();
    /// col_major.set_order(Order::ColMajor);
    /// col_major.reshape((3, 2))?;
    /// assert_eq!(col_major, matrix![[1, 5], [4, 3], [2, 6]]);
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
    /// [`shrink_to_fit`]: Matrix::shrink_to_fit
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

    /// Shrinks the capacity of the matrix as much as possible.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
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
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
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

    /// Returns `true` if the matrix contains an element with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert!(matrix.contains(&5));
    /// assert!(!matrix.contains(&10));
    /// ```
    #[inline]
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.data.contains(value)
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
            let major = cmp::min(self.major(), source.major());
            let minor = cmp::min(self.minor(), source.minor());
            for i in 0..major {
                let self_lower = i * self.major_stride();
                let self_upper = self_lower + minor * self.minor_stride();
                let source_lower = i * source.major_stride();
                let source_upper = source_lower + minor * self.minor_stride();
                unsafe {
                    self.data
                        .get_unchecked_mut(self_lower..self_upper)
                        .clone_from_slice(source.data.get_unchecked(source_lower..source_upper));
                }
            }
        } else {
            let major = cmp::min(self.major(), source.minor());
            let minor = cmp::min(self.minor(), source.major());
            for i in 0..major {
                let self_lower = i * self.major_stride();
                let self_upper = self_lower + minor * self.minor_stride();
                unsafe {
                    self.data
                        .get_unchecked_mut(self_lower..self_upper)
                        .iter_mut()
                        .zip(source.data.iter().skip(i).step_by(source.major_stride()))
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.apply(|element| *element += 2);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
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
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let result = matrix.map(|element| element as f64);
    /// assert_eq!(result, Ok(matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    /// ```
    #[inline]
    pub fn map<U, F>(self, f: F) -> Result<Matrix<U>>
    where
        F: FnMut(T) -> U,
    {
        Matrix::<U>::check_size(self.size())?;

        let order = self.order;
        let shape = self.shape;
        let data = self.data.into_iter().map(f).collect();

        Ok(Matrix { order, shape, data })
    }

    /// Applies a closure to each element of the matrix,
    /// returning a new matrix with the results.
    ///
    /// This method is similar to [`map`] but passes references to the
    /// elements instead of taking ownership of them.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let result = matrix.map_ref(|element| *element as f64);
    /// assert_eq!(result, Ok(matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    /// ```
    ///
    /// [`map`]: Matrix::map
    #[inline]
    pub fn map_ref<'a, U, F>(&'a self, f: F) -> Result<Matrix<U>>
    where
        F: FnMut(&'a T) -> U,
    {
        Matrix::<U>::check_size(self.size())?;

        let order = self.order;
        let shape = self.shape;
        let data = self.data.iter().map(f).collect();

        Ok(Matrix { order, shape, data })
    }

    /// Clears the matrix, removing all elements.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.clear();
    /// assert_eq!(matrix.nrows(), 0);
    /// assert_eq!(matrix.ncols(), 0);
    /// assert!(matrix.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) -> &mut Self {
        self.shape = AxisShape::default();
        self.data.clear();
        self
    }
}

impl<T> Matrix<T> {
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    fn check_size(size: usize) -> Result<usize> {
        if size_of::<T>().saturating_mul(size) > isize::MAX as usize {
            Err(Error::CapacityOverflow)
        } else {
            Ok(size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        matrix.transpose();
        assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);

        matrix.transpose();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        // testing `Matrix::transpose` in a different order is pointless
        // since `Matrix::switch_order` depends on this method
    }

    #[test]
    fn test_switch_order() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let order = matrix.order;

        matrix.switch_order();
        assert_ne!(matrix.order, order);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        assert_eq!(matrix.order, order);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    }

    #[test]
    fn test_switch_order_without_rearrangement() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let order = matrix.order;

        matrix.switch_order_without_rearrangement();
        assert_ne!(matrix.order, order);
        assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);

        matrix.switch_order_without_rearrangement();
        assert_eq!(matrix.order, order);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    }

    #[test]
    fn test_set_order() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        matrix.set_order(Order::RowMajor);
        assert_eq!(matrix.order(), Order::RowMajor);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.set_order(Order::ColMajor);
        assert_eq!(matrix.order(), Order::ColMajor);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    }

    #[test]
    fn test_set_order_without_rearrangement() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        matrix.set_order(Order::RowMajor);

        matrix.set_order_without_rearrangement(Order::RowMajor);
        assert_eq!(matrix.order(), Order::RowMajor);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.set_order_without_rearrangement(Order::ColMajor);
        assert_eq!(matrix.order(), Order::ColMajor);
        assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    }

    #[test]
    fn test_reshape() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];

        // row-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::RowMajor);

            matrix.reshape((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

            matrix.reshape((3, 2)).unwrap();
            assert_eq!(matrix, matrix![[1, 2], [3, 4], [5, 6]]);

            matrix.reshape((1, 6)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3, 4, 5, 6]]);

            matrix.reshape((6, 1)).unwrap();
            assert_eq!(matrix, matrix![[1], [2], [3], [4], [5], [6]]);

            matrix.reshape((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
        }

        // col-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::ColMajor);

            matrix.reshape((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

            matrix.reshape((3, 2)).unwrap();
            assert_eq!(matrix, matrix![[1, 5], [4, 3], [2, 6]]);

            matrix.reshape((1, 6)).unwrap();
            assert_eq!(matrix, matrix![[1, 4, 2, 5, 3, 6]]);

            matrix.reshape((6, 1)).unwrap();
            assert_eq!(matrix, matrix![[1], [4], [2], [5], [3], [6]]);

            matrix.reshape((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
        }

        // errors
        {
            let mut matrix = matrix.clone();
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
    }

    #[test]
    fn test_resize() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];

        // row-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::RowMajor);

            matrix.resize((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

            matrix.resize((2, 2)).unwrap();
            assert_eq!(matrix, matrix![[1, 2], [3, 4]]);

            matrix.resize((3, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 0, 0], [0, 0, 0]]);

            matrix.resize((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 0, 0]]);

            matrix.resize((2, 0)).unwrap();
            assert_eq!(matrix, matrix![[], []]);
        }

        // col-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::ColMajor);

            matrix.resize((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

            matrix.resize((2, 2)).unwrap();
            assert_eq!(matrix, matrix![[1, 2], [4, 5]]);

            matrix.resize((3, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 5, 0], [4, 0, 0], [2, 0, 0]]);

            matrix.resize((2, 3)).unwrap();
            assert_eq!(matrix, matrix![[1, 2, 0], [4, 5, 0]]);

            matrix.resize((2, 0)).unwrap();
            assert_eq!(matrix, matrix![[], []]);
        }

        // errors
        {
            let mut matrix = matrix.clone();
            let unchanged = matrix.clone();

            let error = matrix.resize((usize::MAX, 2)).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
            assert_eq!(matrix, unchanged);

            let error = matrix.resize((isize::MAX as usize + 1, 1)).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
            assert_eq!(matrix, unchanged);
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        assert!(matrix.capacity() >= 6);

        matrix.resize((1, 3)).unwrap();
        assert!(matrix.capacity() >= 6);

        matrix.shrink_to_fit();
        assert!(matrix.capacity() >= 3);
    }

    #[test]
    fn test_shrink_to() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        assert!(matrix.capacity() >= 6);

        matrix.resize((1, 3)).unwrap();
        assert!(matrix.capacity() >= 6);

        matrix.shrink_to(4);
        assert!(matrix.capacity() >= 4);

        matrix.shrink_to(0);
        assert!(matrix.capacity() >= 3);
    }

    #[test]
    fn test_contains() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        assert!(!matrix.contains(&0));
        assert!(matrix.contains(&1));
        assert!(matrix.contains(&2));
        assert!(matrix.contains(&3));
        assert!(matrix.contains(&4));
        assert!(matrix.contains(&5));
        assert!(matrix.contains(&6));
    }

    #[test]
    fn test_overwrite() {
        fn test_helper(destination: Matrix<i32>, source: Matrix<i32>, expected: Matrix<i32>) {
            // row-major & row-major
            {
                let mut destination = destination.clone();
                let mut source = source.clone();
                destination.set_order(Order::RowMajor);
                source.set_order(Order::RowMajor);

                destination.overwrite(&source);
                assert_eq!(destination, expected);
            }

            // row-major & col-major
            {
                let mut destination = destination.clone();
                let mut source = source.clone();
                destination.set_order(Order::RowMajor);
                source.set_order(Order::ColMajor);

                destination.overwrite(&source);
                assert_eq!(destination, expected);
            }

            // col-major & row-major
            {
                let mut destination = destination.clone();
                let mut source = source.clone();
                destination.set_order(Order::ColMajor);
                source.set_order(Order::RowMajor);

                destination.overwrite(&source);
                assert_eq!(destination, expected);
            }

            // col-major & col-major
            {
                let mut destination = destination.clone();
                let mut source = source.clone();
                destination.set_order(Order::ColMajor);
                source.set_order(Order::ColMajor);

                destination.overwrite(&source);
                assert_eq!(destination, expected);
            }
        }

        let destination = matrix![[0, 0, 0], [0, 0, 0]];

        {
            let source = matrix![[1, 2]];
            let expected = matrix![[1, 2, 0], [0, 0, 0]];

            test_helper(destination.clone(), source, expected);
        }

        {
            let source = matrix![[1, 2], [3, 4]];
            let expected = matrix![[1, 2, 0], [3, 4, 0]];

            test_helper(destination.clone(), source, expected);
        }

        {
            let source = matrix![[1, 2], [3, 4], [5, 6]];
            let expected = matrix![[1, 2, 0], [3, 4, 0]];

            test_helper(destination.clone(), source, expected);
        }

        {
            let source = matrix![[1, 2, 3]];
            let expected = matrix![[1, 2, 3], [0, 0, 0]];

            test_helper(destination.clone(), source, expected);
        }

        {
            let source = matrix![[1, 2, 3, 4]];
            let expected = matrix![[1, 2, 3], [0, 0, 0]];

            test_helper(destination.clone(), source, expected);
        }
    }

    #[test]
    fn test_apply() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // row-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::RowMajor);

            matrix.apply(|element| *element += 2);
            assert_eq!(matrix, expected);
        }

        // col-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::ColMajor);

            matrix.apply(|element| *element += 2);
            assert_eq!(matrix, expected);
        }
    }

    #[test]
    fn test_map() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // row-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::RowMajor);

            let output = matrix.map(|element| element as f64).unwrap();
            assert_eq!(output, expected);
        }

        // col-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::ColMajor);

            let output = matrix.map(|element| element as f64).unwrap();
            assert_eq!(output, expected);
        }

        // errors
        {
            let matrix = matrix![[(); usize::MAX]; 1];

            let error = matrix.map(|_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }
    }

    #[test]
    fn test_map_ref() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // row-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::RowMajor);

            let output = matrix.map_ref(|element| *element as f64).unwrap();
            assert_eq!(output, expected);
        }

        // col-major
        {
            let mut matrix = matrix.clone();
            matrix.set_order(Order::ColMajor);

            let output = matrix.map_ref(|element| *element as f64).unwrap();
            assert_eq!(output, expected);
        }

        // to matrix of references
        {
            let output = matrix.map_ref(|element| element).unwrap();
            assert_eq!(output, matrix![[&1, &2, &3], [&4, &5, &6]]);
        }

        // errors
        {
            let matrix = matrix![[(); usize::MAX]; 1];

            let error = matrix.map_ref(|_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }
    }

    #[test]
    fn test_clear() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        matrix.clear();
        assert_eq!(matrix.nrows(), 0);
        assert_eq!(matrix.ncols(), 0);
        assert!(matrix.is_empty());
    }
}
