//! This module defines [`Matrix<T>`] and all its related components.

use self::iter::VectorIter;
use self::order::Order;
use self::shape::{AxisShape, Shape, ShapeLike};
use crate::error::{Error, Result};

pub mod index;
pub mod iter;
pub mod order;
pub mod shape;

mod arithmetic;
mod conversion;
mod default;
mod fmt;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// [`Matrix<T>`] means ... matrix.
///
/// Instead of using constructor methods, you might prefer to create a
/// matrix with the [`matrix!`] macro:
///
/// ```
/// use matreex::matrix;
///
/// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
/// ```
///
/// [`matrix!`]: crate::matrix!
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    order: Order,
    shape: AxisShape,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Creates a new [`Matrix<T>`] instance with default values.
    ///
    /// # Panics
    ///
    /// Panics if size exceeds [`usize::MAX`], or total bytes stored
    /// exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Matrix};
    ///
    /// let matrix = Matrix::new((2, 3));
    /// assert_eq!(matrix, matrix![[0, 0, 0], [0, 0, 0]]);
    /// ```
    ///
    /// ```should_panic
    /// # use matreex::Matrix;
    /// #
    /// let matrix = Matrix::<u8>::new((usize::MAX, 2));
    /// ```
    ///
    /// ```should_panic
    /// # use matreex::Matrix;
    /// #
    /// let matrix = Matrix::<u8>::new((isize::MAX as usize + 1, 1));
    /// ```
    pub fn new<S: ShapeLike>(shape: S) -> Self
    where
        T: Default,
    {
        match Self::build(shape) {
            Err(error) => panic!("{error}"),
            Ok(matrix) => matrix,
        }
    }

    /// Builds a new [`Matrix<T>`] instance with default values.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityExceeded`] if total bytes stored exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error, Matrix};
    ///
    /// let result = Matrix::build((2, 3));
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    ///
    /// let result = Matrix::<u8>::build((usize::MAX, 2));
    /// assert_eq!(result, Err(Error::SizeOverflow));
    ///
    /// let result = Matrix::<u8>::build((isize::MAX as usize + 1, 1));
    /// assert_eq!(result, Err(Error::CapacityExceeded));
    /// ```
    pub fn build<S: ShapeLike>(shape: S) -> Result<Self>
    where
        T: Default,
    {
        let order = Order::default();
        let shape = AxisShape::try_from_shape(shape, order)?;
        let size = Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(size);
        data.resize_with(size, T::default);
        Ok(Self { order, shape, data })
    }

    /// Creates an empty [`Matrix<T>`] instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    ///
    /// let matrix = Matrix::<u8>::empty();
    /// assert_eq!(matrix.nrows(), 0);
    /// assert_eq!(matrix.ncols(), 0);
    /// assert!(matrix.is_empty());
    /// ```
    pub fn empty() -> Self {
        Self::default()
    }
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
    /// assert_eq!(matrix.shape(), Shape::new(2, 3));
    /// ```
    pub fn shape(&self) -> Shape {
        self.shape.interpret(self.order)
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Shape};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.nrows(), 2);
    /// ```
    pub fn nrows(&self) -> usize {
        self.shape.interpret_nrows(self.order)
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Shape};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.ncols(), 3);
    /// ```
    pub fn ncols(&self) -> usize {
        self.shape.interpret_ncols(self.order)
    }

    /// Returns the total number of elements in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Shape};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the matrix contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Matrix, Shape};
    ///
    /// let matrix: Matrix<u8> = matrix![];
    /// assert!(matrix.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the capacity of the matrix.
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
    /// matrix.resize((1, 10))?;
    /// assert!(matrix.capacity() >= 10);
    /// # Ok(())
    /// # }
    /// ```
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
                let next = Self::transpose_flattened_index(current, self.shape);
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
    /// use matreex::{matrix, Order};
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let mut expected = Order::default();
    /// assert_eq!(matrix.order(), expected);
    ///
    /// matrix.switch_order();
    /// expected.switch();
    /// assert_eq!(matrix.order(), expected);
    ///
    /// matrix.switch_order();
    /// expected.switch();
    /// assert_eq!(matrix.order(), expected);
    /// ```
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
    /// use matreex::{matrix, Order};
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let mut expected = Order::default();
    /// assert_eq!(matrix.order(), expected);
    ///
    /// matrix.switch_order_without_rearrangement();
    /// expected.switch();
    /// assert_eq!(matrix.order(), expected);
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 3);
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 1);
    /// assert_eq!(matrix[(1, 1)], 4);
    /// // row 2
    /// assert_eq!(matrix[(2, 0)], 2);
    /// assert_eq!(matrix[(2, 1)], 5);
    ///
    /// matrix.switch_order_without_rearrangement();
    /// expected.switch();
    /// assert_eq!(matrix.order(), expected);
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 1);
    /// assert_eq!(matrix[(0, 2)], 2);
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 3);
    /// assert_eq!(matrix[(1, 1)], 4);
    /// assert_eq!(matrix[(1, 2)], 5);
    /// ```
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
    /// assert_eq!(matrix.order(), Order::default());
    ///
    /// matrix.set_order(Order::RowMajor);
    /// assert_eq!(matrix.order(), Order::RowMajor);
    ///
    /// matrix.set_order(Order::ColMajor);
    /// assert_eq!(matrix.order(), Order::ColMajor);
    /// ```
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
    /// use matreex::{matrix, Order};
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.order(), Order::default());
    ///
    /// matrix.set_order_without_rearrangement(Order::RowMajor);
    /// assert_eq!(matrix.order(), Order::RowMajor);
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 1);
    /// assert_eq!(matrix[(0, 2)], 2);
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 3);
    /// assert_eq!(matrix[(1, 1)], 4);
    /// assert_eq!(matrix[(1, 2)], 5);
    ///
    /// matrix.set_order_without_rearrangement(Order::ColMajor);
    /// assert_eq!(matrix.order(), Order::ColMajor);
    /// // row 0
    /// assert_eq!(matrix[(0, 0)], 0);
    /// assert_eq!(matrix[(0, 1)], 3);
    /// // row 1
    /// assert_eq!(matrix[(1, 0)], 1);
    /// assert_eq!(matrix[(1, 1)], 4);
    /// // row 2
    /// assert_eq!(matrix[(2, 0)], 2);
    /// assert_eq!(matrix[(2, 1)], 5);
    /// ```
    pub fn set_order_without_rearrangement(&mut self, order: Order) -> &mut Self {
        if order != self.order {
            self.switch_order_without_rearrangement();
        }
        self
    }

    /// Resizes the matrix to the specified shape.
    ///
    /// # Notes
    ///
    /// Reducing the size does not automatically shrink the capacity.
    /// This choice is made to avoid potential reallocation.
    /// Consider explicitly calling [`Matrix::shrink_to_fit`]
    /// if needed.
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
    pub fn resize<S: ShapeLike>(&mut self, shape: S) -> Result<&mut Self>
    where
        T: Default,
    {
        let shape = AxisShape::try_from_shape(shape, self.order)?;
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
    /// match the current size of the matrix.
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
    pub fn reshape<S: ShapeLike>(&mut self, shape: S) -> Result<&mut Self> {
        match shape.size() {
            Ok(size) if (self.size() == size) => {
                self.shape = AxisShape::from_shape_unchecked(shape, self.order);
                Ok(self)
            }
            _ => Err(Error::SizeMismatch),
        }
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
    /// matrix.shrink_to_fit();
    /// assert!(matrix.capacity() >= 3);
    /// # Ok(())
    /// # }
    /// ```
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
    /// matrix.shrink_to(4);
    /// assert!(matrix.capacity() >= 4);
    /// matrix.shrink_to(0);
    /// assert!(matrix.capacity() >= 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shrink_to(&mut self, min_capacity: usize) -> &mut Self {
        self.data.shrink_to(min_capacity);
        self
    }

    /// Overwrites the overlapping part of this matrix with another one,
    /// leaving the non-overlapping part unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Order};
    ///
    /// let mut matrix = matrix![[0, 0, 0], [0, 0, 0]];
    /// let other = matrix![[1, 1], [1, 1], [1, 1]];
    ///
    /// matrix.overwrite_with(&other);
    /// assert_eq!(matrix, matrix![[1, 1, 0], [1, 1, 0]]);
    /// ```
    pub fn overwrite_with(&mut self, other: &Self) -> &mut Self
    where
        T: Clone,
    {
        if self.order == other.order {
            let major = std::cmp::min(self.major(), other.major());
            let minor = std::cmp::min(self.minor(), other.minor());
            for i in 0..major {
                let self_lower = i * self.major_stride();
                let self_upper = self_lower + minor;
                let other_lower = i * other.major_stride();
                let other_upper = other_lower + minor;
                self.data[self_lower..self_upper]
                    .clone_from_slice(&other.data[other_lower..other_upper]);
            }
        } else {
            let major = std::cmp::min(self.major(), other.minor());
            let minor = std::cmp::min(self.minor(), other.major());
            for i in 0..major {
                let self_lower = i * self.major_stride();
                let self_upper = self_lower + minor;
                self.data[self_lower..self_upper]
                    .iter_mut()
                    .zip(other.iter_nth_minor_axis_vector_unchecked(i))
                    .for_each(|(x, y)| *x = y.clone());
            }
        }
        self
    }
}

impl<T> Matrix<T> {
    /// Applies a closure to each element of the matrix,
    /// modifying the matrix in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.apply(|x| *x += 1);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
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
    ///
    /// let matrix_f64 = matrix_i32.map(|x| x as f64);
    /// assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// ```
    pub fn map<U, F>(self, f: F) -> Matrix<U>
    where
        F: FnMut(T) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self.data.into_iter().map(f).collect();
        Matrix { order, shape, data }
    }
}

#[cfg(feature = "rayon")]
impl<T> Matrix<T>
where
    T: Sync + Send,
{
    /// Applies a closure to each element of the matrix in parallel,
    /// modifying the matrix in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.par_apply(|x| *x += 1);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    pub fn par_apply<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&mut T) + Sync + Send,
    {
        self.data.par_iter_mut().for_each(f);
        self
    }

    /// Applies a closure to each element of the matrix in parallel,
    /// returning a new matrix with the results.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// let matrix_f64 = matrix_i32.par_map(|x| x as f64);
    /// assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// ```
    pub fn par_map<U, F>(self, f: F) -> Matrix<U>
    where
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self.data.into_par_iter().map(f).collect();
        Matrix { order, shape, data }
    }
}

impl<L> Matrix<L> {
    /// Ensures that two matrices are conformable for elementwise operations.
    ///
    /// # Errors
    ///
    /// - [`Error::NotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, Matrix};
    ///
    /// let lhs = Matrix::<i32>::new((2, 3));
    ///
    /// let rhs = Matrix::<i32>::new((2, 3));
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = Matrix::<i32>::new((3, 2));
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::NotConformable));
    /// ```
    pub fn ensure_elementwise_operation_conformable<R>(&self, rhs: &Matrix<R>) -> Result<&Self> {
        if self.shape() != rhs.shape() {
            Err(Error::NotConformable)
        } else {
            Ok(self)
        }
    }

    /// Performs elementwise operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::NotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// let result = lhs.elementwise_operation(&rhs, |(x, y)| x + y);
    /// assert_eq!(result, Ok(matrix![[2, 3, 4], [5, 6, 7]]));
    /// ```
    pub fn elementwise_operation<R, F, U>(&self, rhs: &Matrix<R>, mut op: F) -> Result<Matrix<U>>
    where
        F: FnMut((&L, &R)) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        let data = if self.order == rhs.order {
            self.data.iter().zip(rhs.data.iter()).map(op).collect()
        } else {
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = Self::transpose_flattened_index(index, self.shape);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op((left, right))
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::NotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// let result = lhs.elementwise_operation_consume_self(&rhs, |(x, y)| x + y);
    /// assert_eq!(result, Ok(matrix![[2, 3, 4], [5, 6, 7]]));
    /// ```
    pub fn elementwise_operation_consume_self<R, F, U>(
        self,
        rhs: &Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut((L, &R)) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        let data = if self.order == rhs.order {
            self.data.into_iter().zip(rhs.data.iter()).map(op).collect()
        } else {
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = Self::transpose_flattened_index(index, self.shape);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op((left, right))
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, assigning the result
    /// to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::NotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    ///
    /// lhs.elementwise_operation_assign(&rhs, |(x, y)| *x += y)?;
    /// assert_eq!(lhs, matrix![[2, 3, 4], [5, 6, 7]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn elementwise_operation_assign<R, F>(
        &mut self,
        rhs: &Matrix<R>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut((&mut L, &R)),
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        if self.order == rhs.order {
            self.data.iter_mut().zip(rhs.data.iter()).for_each(op);
        } else {
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = Self::transpose_flattened_index(index, self.shape);
                let right = unsafe { rhs.data.get_unchecked(index) };
                op((left, right))
            });
        }

        Ok(self)
    }
}

impl<L> Matrix<L> {
    /// Ensures that two matrices are conformable for multiplication-like
    /// operation.
    ///
    /// # Errors
    ///
    /// - [`Error::NotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, Matrix};
    ///
    /// let lhs = Matrix::<i32>::new((2, 3));
    ///
    /// let rhs = Matrix::<i32>::new((3, 2));
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = Matrix::<i32>::new((2, 3));
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::NotConformable));
    /// ```
    pub fn ensure_multiplication_like_operation_conformable<R>(
        &self,
        rhs: &Matrix<R>,
    ) -> Result<&Self> {
        if self.ncols() != rhs.nrows() {
            Err(Error::NotConformable)
        } else {
            Ok(self)
        }
    }

    /// Performs multiplication-like operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::NotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// For performance reasons, this method consumes both `self` and `rhs`.
    ///
    /// The closure `op` is guaranteed to receive two non-empty, equal-length
    /// vectors. It should always return a valid value derived from them.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, VectorIter};
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[0, 1], [2, 3], [4, 5]];
    /// let op = |vl: VectorIter<&i32>, vr: VectorIter<&i32>| {
    ///     vl.zip(vr).map(|(x, y)| x * y).reduce(|acc, p| acc + p).unwrap()
    /// };
    ///
    /// let result = lhs.multiplication_like_operation(rhs, op);
    /// assert_eq!(result, Ok(matrix![[10, 13], [28, 40]]));
    /// ```
    pub fn multiplication_like_operation<R, F, U>(
        mut self,
        mut rhs: Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(VectorIter<&L>, VectorIter<&R>) -> U,
        U: Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let order = self.order;
        let shape = AxisShape::try_from_shape(Shape::new(nrows, ncols), order)?;
        let size = shape.size();
        let mut data = Vec::with_capacity(size);

        if self.ncols() == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { order, shape, data });
        }

        self.set_order(Order::RowMajor);
        rhs.set_order(Order::ColMajor);

        match order {
            Order::RowMajor => {
                for row in 0..nrows {
                    for col in 0..ncols {
                        let element = op(
                            unsafe { Box::new(self.iter_nth_major_axis_vector_unchecked(row)) },
                            unsafe { Box::new(rhs.iter_nth_major_axis_vector_unchecked(col)) },
                        );
                        data.push(element);
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let element = op(
                            unsafe { Box::new(self.iter_nth_major_axis_vector_unchecked(row)) },
                            unsafe { Box::new(rhs.iter_nth_major_axis_vector_unchecked(col)) },
                        );
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { order, shape, data })
    }
}

impl<T> Matrix<T> {
    /// Performs scalar operation on the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let scalar = 2;
    ///
    /// let result = matrix.scalar_operation(&scalar, |x, y| x + y);
    /// assert_eq!(result, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    pub fn scalar_operation<S, F, U>(&self, scalar: &S, mut op: F) -> Matrix<U>
    where
        F: FnMut(&T, &S) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self
            .data
            .iter()
            .map(|element| op(element, scalar))
            .collect();
        Matrix { order, shape, data }
    }

    /// Performs scalar operation on the matrix, consuming `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let scalar = 2;
    ///
    /// let result = matrix.scalar_operation_consume_self(&scalar, |x, y| x + y);
    /// assert_eq!(result, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    pub fn scalar_operation_consume_self<S, F, U>(self, scalar: &S, mut op: F) -> Matrix<U>
    where
        F: FnMut(T, &S) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self
            .data
            .into_iter()
            .map(|element| op(element, scalar))
            .collect();
        Matrix { order, shape, data }
    }

    /// Performs scalar operation on the matrix, assigning the result
    /// to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let scalar = 2;
    ///
    /// matrix.scalar_operation_assign(&scalar, |x, y| *x += y);
    /// assert_eq!(matrix, matrix![[2, 3, 4], [5, 6, 7]]);
    /// ```
    pub fn scalar_operation_assign<S, F>(&mut self, scalar: &S, mut op: F) -> &mut Self
    where
        F: FnMut(&mut T, &S),
    {
        self.data.iter_mut().for_each(|element| op(element, scalar));
        self
    }
}

impl<T> Matrix<T> {
    fn check_size(size: usize) -> Result<usize> {
        // see more info at https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.with_capacity
        const MAX: usize = isize::MAX as usize;
        match std::mem::size_of::<T>().checked_mul(size) {
            Some(0..=MAX) => Ok(size),
            _ => Err(Error::CapacityExceeded),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    // All tests are based on the assumption that
    // the default order is `RowMajor`.

    #[test]
    fn test_new() {
        let expected = matrix![[0, 0, 0], [0, 0, 0]];
        assert_eq!(Matrix::new((2, 3)), expected);
        assert_ne!(Matrix::new((3, 2)), expected);
    }

    #[test]
    fn test_build() {
        let expected = matrix![[0, 0, 0], [0, 0, 0]];
        assert_eq!(Matrix::build((2, 3)).unwrap(), expected);
        assert_ne!(Matrix::build((3, 2)).unwrap(), expected);

        assert_eq!(
            Matrix::<u8>::build((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(
            Matrix::<u8>::build((isize::MAX as usize + 1, 1)).unwrap_err(),
            Error::CapacityExceeded
        );

        assert_eq!(
            Matrix::<i32>::build((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(
            Matrix::<i32>::build((isize::MAX as usize / 4 + 1, 1)).unwrap_err(),
            Error::CapacityExceeded
        );

        // The following test cases for zero-sized types are impractical to
        // run in debug mode, and since `#[cfg(not(debug_assertions))]` does
        // not strictly match release mode, these tests are commented out.

        // assert_eq!(
        //     Matrix::<()>::build((usize::MAX, 2)).unwrap_err(),
        //     Error::SizeOverflow
        // );
        // assert!(Matrix::<()>::build((isize::MAX as usize + 1, 1)).is_ok());

        // #[derive(Debug, Default)]
        // struct Foo;
        // assert_eq!(
        //     Matrix::<Foo>::build((usize::MAX, 2)).unwrap_err(),
        //     Error::SizeOverflow
        // );
        // assert!(Matrix::<Foo>::build((isize::MAX as usize + 1, 1)).is_ok());
    }

    #[test]
    fn test_empty() {
        let matrix = Matrix::<i32>::empty();
        assert_eq!(matrix.order, Order::RowMajor);
        assert_eq!(matrix.nrows(), 0);
        assert_eq!(matrix.ncols(), 0);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_transpose() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        assert_eq!(matrix.order, Order::RowMajor);
        assert_eq!(matrix.major(), 2);
        assert_eq!(matrix.minor(), 3);

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

            assert_eq!(matrix.order, Order::RowMajor);
            assert_eq!(matrix.major(), 3);
            assert_eq!(matrix.minor(), 2);
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

            assert_eq!(matrix.order, Order::RowMajor);
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
        }
    }

    #[test]
    fn test_switch_order() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        assert_eq!(matrix.order, Order::RowMajor);
        assert_eq!(matrix.major(), 2);
        assert_eq!(matrix.minor(), 3);

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

            assert_eq!(matrix.order, Order::ColMajor);
            assert_eq!(matrix.major(), 3);
            assert_eq!(matrix.minor(), 2);
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

            assert_eq!(matrix.order, Order::RowMajor);
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
        }
    }

    #[test]
    fn test_switch_order_without_rearrangement() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        assert_eq!(matrix.order, Order::RowMajor);
        assert_eq!(matrix.major(), 2);
        assert_eq!(matrix.minor(), 3);

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

            assert_eq!(matrix.order, Order::ColMajor);
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
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

            assert_eq!(matrix.order, Order::RowMajor);
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
        }
    }

    #[test]
    fn test_set_order() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        assert_eq!(matrix.order, Order::RowMajor);
        assert_eq!(matrix.major(), 2);
        assert_eq!(matrix.minor(), 3);

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
            assert_eq!(matrix.major(), 3);
            assert_eq!(matrix.minor(), 2);
        }

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
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
        }
    }

    #[test]
    fn test_set_order_without_rearrangement() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        assert_eq!(matrix.order, Order::RowMajor);
        assert_eq!(matrix.major(), 2);
        assert_eq!(matrix.minor(), 3);

        matrix.set_order_without_rearrangement(Order::RowMajor);

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
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
        }

        matrix.set_order_without_rearrangement(Order::ColMajor);

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

            assert_eq!(matrix.order, Order::ColMajor);
            assert_eq!(matrix.major(), 2);
            assert_eq!(matrix.minor(), 3);
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
        assert_eq!(error, Error::CapacityExceeded);
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
    fn test_overwrite_with() {
        let blank = matrix![[0, 0, 0], [0, 0, 0]];

        {
            let mut other = matrix![[1, 2]];

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 0], [0, 0, 0]]);

            other.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 0], [0, 0, 0]]);
        }

        {
            let mut other = matrix![[1, 2], [3, 4]];

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);

            other.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);
        }

        {
            let mut other = matrix![[1, 2], [3, 4], [5, 6]];

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);

            other.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 0], [3, 4, 0]]);
        }

        {
            let mut other = matrix![[1, 2, 3]];

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);

            other.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);
        }

        {
            let mut other = matrix![[1, 2, 3, 4]];

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
            assert_eq!(matrix, matrix![[1, 2, 3], [0, 0, 0]]);

            other.switch_order();

            let mut matrix = blank.clone();
            matrix.overwrite_with(&other);
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

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_apply() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.par_apply(|x| *x += 2);
        assert_eq!(matrix, matrix![[2, 3, 4], [5, 6, 7]]);

        matrix.switch_order();

        matrix.par_apply(|x| *x -= 2);
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_map() {
        let matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];

        let mut matrix_f64 = matrix_i32.par_map(|x| x as f64);
        assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        matrix_f64.switch_order();

        let mut matrix_i32 = matrix_f64.par_map(|x| x as i32);
        matrix_i32.switch_order();
        assert_eq!(matrix_i32, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_ensure_elementwise_operation_conformable() {
        let mut lhs = Matrix::<i32>::new((2, 3));
        let mut rhs = Matrix::<i32>::new((2, 3));

        // RowMajor & RowMajor
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // RowMajor & ColMajor
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        lhs.switch_order();

        // ColMajor & ColMajor
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // ColMajor & RowMajor
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::new((2, 2));
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert_eq!(result, Err(Error::NotConformable));

        let rhs = Matrix::<i32>::new((3, 2));
        let result = lhs.ensure_elementwise_operation_conformable(&rhs);
        assert_eq!(result, Err(Error::NotConformable));
    }

    #[test]
    fn test_elementwise_operation() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let op = |(x, y): (&i32, &i32)| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // RowMajor & RowMajor
        let output = lhs.elementwise_operation(&rhs, op).unwrap();
        assert_eq!(output, expected);

        rhs.switch_order();

        // RowMajor & ColMajor
        let output = lhs.elementwise_operation(&rhs, op).unwrap();
        assert_eq!(output, expected);

        lhs.switch_order();

        // ColMajor & ColMajor
        let mut output = lhs.elementwise_operation(&rhs, op).unwrap();
        output.switch_order();
        assert_eq!(output, expected);

        rhs.switch_order();

        // ColMajor & RowMajor
        let mut output = lhs.elementwise_operation(&rhs, op).unwrap();
        output.switch_order();
        assert_eq!(output, expected);

        let rhs = matrix![[2, 2], [2, 2]];
        let error = lhs.elementwise_operation(&rhs, op).unwrap_err();
        assert_eq!(error, Error::NotConformable);

        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        let error = lhs.elementwise_operation(&rhs, op).unwrap_err();
        assert_eq!(error, Error::NotConformable);
    }

    #[test]
    fn test_elementwise_operation_consume_self() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let op = |(x, y): (i32, &i32)| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // RowMajor & RowMajor
        {
            let lhs = lhs.clone();
            let output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // RowMajor & ColMajor
        {
            let lhs = lhs.clone();
            let output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        // ColMajor & ColMajor
        {
            let lhs = lhs.clone();
            let mut output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // ColMajor & RowMajor
        {
            let lhs = lhs.clone();
            let mut output = lhs.elementwise_operation_consume_self(&rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2]];
            let error = lhs
                .elementwise_operation_consume_self(&rhs, op)
                .unwrap_err();
            assert_eq!(error, Error::NotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]];
            let error = lhs
                .elementwise_operation_consume_self(&rhs, op)
                .unwrap_err();
            assert_eq!(error, Error::NotConformable);
        }
    }

    #[test]
    fn test_elementwise_operation_assign() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let op = |(x, y): (&mut i32, &i32)| *x += y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // RowMajor & RowMajor
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            assert_eq!(lhs, expected);
        }

        rhs.switch_order();

        // RowMajor & ColMajor
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            assert_eq!(lhs, expected);
        }

        lhs.switch_order();

        // ColMajor & ColMajor
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            lhs.switch_order();
            assert_eq!(lhs, expected);
        }

        rhs.switch_order();

        // ColMajor & RowMajor
        {
            let mut lhs = lhs.clone();
            lhs.elementwise_operation_assign(&rhs, op).unwrap();
            lhs.switch_order();
            assert_eq!(lhs, expected);
        }

        let unchanged = lhs.clone();

        let rhs = matrix![[2, 2], [2, 2]];
        let error = lhs.elementwise_operation_assign(&rhs, op).unwrap_err();
        assert_eq!(error, Error::NotConformable);
        assert_eq!(lhs, unchanged);

        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        let error = lhs.elementwise_operation_assign(&rhs, op).unwrap_err();
        assert_eq!(error, Error::NotConformable);
        assert_eq!(lhs, unchanged);
    }

    #[test]
    fn test_ensure_multiplication_like_operation_conformable() {
        let mut lhs = Matrix::<i32>::new((2, 3));
        let mut rhs = Matrix::<i32>::new((3, 2));

        // RowMajor & RowMajor
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // RowMajor & ColMajor
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        lhs.switch_order();

        // ColMajor & ColMajor
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        rhs.switch_order();

        // ColMajor & RowMajor
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::new((3, 1));
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::new((3, 3));
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert!(result.is_ok());

        let rhs = Matrix::<i32>::new((2, 2));
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert_eq!(result, Err(Error::NotConformable));

        let rhs = Matrix::<i32>::new((2, 3));
        let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
        assert_eq!(result, Err(Error::NotConformable));
    }

    #[test]
    fn test_multiplication_like_operation() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[0, 1], [2, 3], [4, 5]];
        let op = |vl: VectorIter<&i32>, vr: VectorIter<&i32>| {
            vl.zip(vr)
                .map(|(x, y)| x * y)
                .reduce(|acc, p| acc + p)
                .unwrap()
        };
        let expected = matrix![[10, 13], [28, 40]];

        // RowMajor & RowMajor
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // RowMajor & ColMajor
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        // ColMajor & ColMajor
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let mut output = lhs.multiplication_like_operation(rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // ColMajor & RowMajor
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let mut output = lhs.multiplication_like_operation(rhs, op).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0], [1], [2]];
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, matrix![[5], [14]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
            let output = lhs.multiplication_like_operation(rhs, op).unwrap();
            assert_eq!(output, matrix![[15, 18, 21], [42, 54, 66]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1], [2, 3]];
            let error = lhs.multiplication_like_operation(rhs, op).unwrap_err();
            assert_eq!(error, Error::NotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1, 3], [4, 5, 6]];
            let error = lhs.multiplication_like_operation(rhs, op).unwrap_err();
            assert_eq!(error, Error::NotConformable);
        }
    }

    #[test]
    fn test_scalar_operation() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let scalar = 2;
        let op = |x: &i32, y: &i32| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // RowMajor
        let output = matrix.scalar_operation(&scalar, op);
        assert_eq!(output, expected);

        matrix.switch_order();

        // ColMajor
        let mut output = matrix.scalar_operation(&scalar, op);
        output.switch_order();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_scalar_operation_consume_self() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let scalar = 2;
        let op = |x: i32, y: &i32| x + y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // RowMajor
        {
            let matrix = matrix.clone();
            let output = matrix.scalar_operation_consume_self(&scalar, op);
            assert_eq!(output, expected);
        }

        matrix.switch_order();

        // ColMajor
        {
            let matrix = matrix.clone();
            let mut output = matrix.scalar_operation_consume_self(&scalar, op);
            output.switch_order();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn test_scalar_operation_assign() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let scalar = 2;
        let op = |x: &mut i32, y: &i32| *x += y;
        let expected = matrix![[2, 3, 4], [5, 6, 7]];

        // RowMajor
        {
            let mut matrix = matrix.clone();
            matrix.scalar_operation_assign(&scalar, op);
            assert_eq!(matrix, expected);
        }

        matrix.switch_order();

        // ColMajor
        {
            let mut matrix = matrix.clone();
            matrix.scalar_operation_assign(&scalar, op);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }
}
