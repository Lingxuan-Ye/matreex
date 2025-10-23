use super::Matrix;
use super::layout::Order;
use crate::error::Result;
use crate::index::Index;
use rayon::prelude::*;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Applies a closure to each element in parallel, modifying the matrix
    /// in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.par_apply(|element| *element += 2);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn par_apply<F>(&mut self, f: F) -> &mut Self
    where
        T: Send,
        F: Fn(&mut T) + Sync + Send,
    {
        self.data.par_iter_mut().for_each(f);
        self
    }

    /// Applies a closure to each element in parallel, returning a new matrix with
    /// the results.
    ///
    /// See [`par_map_ref`] for a non-consuming version.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let result = matrix.par_map(|element| element as f64);
    /// assert_eq!(result, Ok(matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    /// ```
    ///
    /// [`par_map_ref`]: Matrix::par_map_ref
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn par_map<F, U>(self, f: F) -> Result<Matrix<U, O>>
    where
        T: Send,
        F: Fn(T) -> U + Sync + Send,
        U: Send,
    {
        let layout = self.layout.cast()?;
        let data = self.data.into_par_iter().map(f).collect();
        Ok(Matrix { layout, data })
    }

    /// Applies a closure to each element in parallel, returning a new matrix with
    /// the results.
    ///
    /// See [`par_map`] for a consuming version.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let result = matrix.par_map_ref(|element| *element as f64);
    /// assert_eq!(result, Ok(matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    /// ```
    ///
    /// [`par_map`]: Matrix::par_map
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn par_map_ref<'a, F, U>(&'a self, f: F) -> Result<Matrix<U, O>>
    where
        T: Sync,
        F: Fn(&'a T) -> U + Sync + Send,
        U: Send,
    {
        let layout = self.layout.cast()?;
        let data = self.data.par_iter().map(f).collect();
        Ok(Matrix { layout, data })
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Returns a parallel iterator over the elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use matreex::parallel::*;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let sum = matrix.par_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 21);
    /// ```
    pub fn par_iter_elements(&self) -> impl ParallelIterator<Item = &T>
    where
        T: Sync,
    {
        self.data.par_iter()
    }

    /// Returns a parallel iterator that allows modifying each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use matreex::parallel::*;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .par_iter_elements_mut()
    ///     .for_each(|element| *element += 2);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn par_iter_elements_mut(&mut self) -> impl ParallelIterator<Item = &mut T>
    where
        T: Send,
    {
        self.data.par_iter_mut()
    }

    /// Creates a parallel consuming iterator, that is, one that moves each element
    /// out of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use matreex::parallel::*;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let sum = matrix.into_par_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 21);
    /// ```
    pub fn into_par_iter_elements(self) -> impl ParallelIterator<Item = T>
    where
        T: Send,
    {
        self.data.into_par_iter()
    }

    /// Returns a parallel iterator over the elements along with their indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use matreex::parallel::*;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .par_iter_elements_with_index()
    ///     .for_each(|(index, element)| {
    ///         assert_eq!(element, &matrix[index]);
    ///     });
    /// ```
    pub fn par_iter_elements_with_index(&self) -> impl ParallelIterator<Item = (Index, &T)>
    where
        T: Sync,
    {
        let stride = self.stride();
        self.data
            .par_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }

    /// Returns a parallel iterator that allows modifying each element along with
    /// its index.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use matreex::parallel::*;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .par_iter_elements_mut_with_index()
    ///     .for_each(|(index, element)| {
    ///         *element += index.row as i32 + index.col as i32;
    ///     });
    /// assert_eq!(matrix, matrix![[1, 3, 5], [5, 7, 9]]);
    /// ```
    pub fn par_iter_elements_mut_with_index(
        &mut self,
    ) -> impl ParallelIterator<Item = (Index, &mut T)>
    where
        T: Send,
    {
        let stride = self.stride();
        self.data
            .par_iter_mut()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }

    /// Creates a parallel consuming iterator, that is, one that moves each element
    /// out of the matrix along with its index.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use matreex::parallel::*;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .clone()
    ///     .into_par_iter_elements_with_index()
    ///     .for_each(|(index, element)| {
    ///         assert_eq!(element, matrix[index]);
    ///     });
    /// ```
    pub fn into_par_iter_elements_with_index(self) -> impl ParallelIterator<Item = (Index, T)>
    where
        T: Send,
    {
        let stride = self.stride();
        self.data
            .into_par_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_par_apply() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.par_apply(|element| *element += 2);
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    fn test_par_map() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = matrix.par_map(|element| element as f64).unwrap();
            let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
            assert_eq!(output, expected);

            let matrix = matrix![[(); usize::MAX]; 1].with_order::<O>();
            let error = matrix.par_map(|_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_par_map_ref() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = matrix.par_map_ref(|element| *element as f64).unwrap();
            let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
            assert_eq!(output, expected);

            // Map to matrix of references.
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = matrix.par_map_ref(|element| element).unwrap();
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            assert_eq!(output, expected);

            let matrix = matrix![[(); usize::MAX]; 1].with_order::<O>();
            let error = matrix.par_map_ref(|_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_par_iter_elements() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let sum = matrix.par_iter_elements().sum::<i32>();
            let expected = 21;
            assert_eq!(sum, expected);
        }}
    }

    #[test]
    fn test_par_iter_elements_mut() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix
                .par_iter_elements_mut()
                .for_each(|element| *element += 2);
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    fn test_into_par_iter_elements() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let sum = matrix.into_par_iter_elements().sum::<i32>();
            let expected = 21;
            assert_eq!(sum, expected);
        }}
    }

    #[test]
    fn test_par_iter_elements_with_index() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix
                .par_iter_elements_with_index()
                .for_each(|(index, element)| {
                    assert_eq!(element, &matrix[index]);
                });

            // Assert no panic from unflattening indices occurs.
            let matrix = matrix![[0; 0]; 2].with_order::<O>();
            matrix.par_iter_elements_with_index().for_each(|_| ());

            // Assert no panic from unflattening indices occurs.
            let matrix = matrix![[0; 3]; 0].with_order::<O>();
            matrix.par_iter_elements_with_index().for_each(|_| ());
        }}
    }

    #[test]
    fn test_par_iter_elements_mut_with_index() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix
                .par_iter_elements_mut_with_index()
                .for_each(|(index, element)| {
                    *element += index.row as i32 + index.col as i32;
                });
            let expected = matrix![[1, 3, 5], [5, 7, 9]];
            assert_eq!(matrix, expected);

            // Assert no panic from unflattening indices occurs.
            let mut matrix = matrix![[0; 0]; 2].with_order::<O>();
            matrix.par_iter_elements_mut_with_index().for_each(|_| ());

            // Assert no panic from unflattening indices occurs.
            let mut matrix = matrix![[0; 3]; 0].with_order::<O>();
            matrix.par_iter_elements_mut_with_index().for_each(|_| ());
        }}
    }

    #[test]
    fn test_into_par_iter_elements_with_index() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix
                .clone()
                .into_par_iter_elements_with_index()
                .for_each(|(index, element)| {
                    assert_eq!(element, matrix[index]);
                });

            // Assert no panic from unflattening indices occurs.
            let matrix = matrix![[0; 0]; 2].with_order::<O>();
            matrix.into_par_iter_elements_with_index().for_each(|_| ());

            // Assert no panic from unflattening indices occurs.
            let matrix = matrix![[0; 3]; 0].with_order::<O>();
            matrix.into_par_iter_elements_with_index().for_each(|_| ());
        }}
    }
}
