use super::index::Index;
use super::Matrix;
use rayon::prelude::*;

impl<T> Matrix<T> {
    /// Applies a closure to each element of the matrix in parallel,
    /// modifying the matrix in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix.par_apply(|x| *x += 1);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    pub fn par_apply<F>(&mut self, f: F) -> &mut Self
    where
        T: Send,
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
    /// let matrix_f64 = matrix_i32.par_map(|x| x as f64);
    /// assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// ```
    #[inline]
    pub fn par_map<U, F>(self, f: F) -> Matrix<U>
    where
        T: Send,
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self.data.into_par_iter().map(f).collect();
        Matrix { order, shape, data }
    }

    /// Applies a closure to each element of the matrix in parallel,
    /// returning a new matrix with the results.
    ///
    /// This method is similar to [`par_map`] but passes references to the
    /// elements instead of taking ownership of them.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];
    /// let matrix_f64 = matrix_i32.par_map_ref(|x| *x as f64);
    /// assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    /// ```
    ///
    /// [`par_map`]: Matrix::par_map
    #[inline]
    pub fn par_map_ref<U, F>(&self, f: F) -> Matrix<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        let order = self.order;
        let shape = self.shape;
        let data = self.data.par_iter().map(f).collect();
        Matrix { order, shape, data }
    }
}

impl<T> Matrix<T> {
    /// Returns a parallel iterator over the elements of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use rayon::prelude::*;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let sum = matrix.par_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 15);
    /// ```
    #[inline]
    pub fn par_iter_elements(&self) -> impl ParallelIterator<Item = &T>
    where
        T: Sync,
    {
        self.data.par_iter()
    }

    /// Returns a parallel iterator that allows modifying each element
    /// of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use rayon::prelude::*;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix
    ///     .par_iter_elements_mut()
    ///     .for_each(|element| *element += 1);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    pub fn par_iter_elements_mut(&mut self) -> impl ParallelIterator<Item = &mut T>
    where
        T: Send,
    {
        self.data.par_iter_mut()
    }

    /// Creates a parallel consuming iterator, that is, one that moves each
    /// element out of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use rayon::prelude::*;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let sum = matrix.into_par_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 15);
    /// ```
    #[inline]
    pub fn into_par_iter_elements(self) -> impl ParallelIterator<Item = T>
    where
        T: Send,
    {
        self.data.into_par_iter()
    }

    /// Returns a parallel iterator over the elements of the matrix along with
    /// their indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use rayon::prelude::*;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
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
        self.data.par_iter().enumerate().map(|(index, element)| {
            let index = Index::from_flattened(index, self.order, self.shape);
            (index, element)
        })
    }

    /// Returns a parallel iterator that allows modifying each element
    /// of the matrix along with its index.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Index};
    /// use rayon::prelude::*;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix
    ///     .par_iter_elements_mut_with_index()
    ///     .for_each(|(index, element)| {
    ///         *element += index.row as i32 + index.col as i32;
    ///     });
    /// assert_eq!(matrix, matrix![[0, 2, 4], [4, 6, 8]]);
    /// ```
    pub fn par_iter_elements_mut_with_index(
        &mut self,
    ) -> impl ParallelIterator<Item = (Index, &mut T)>
    where
        T: Send,
    {
        self.data
            .par_iter_mut()
            .enumerate()
            .map(|(index, element)| {
                let index = Index::from_flattened(index, self.order, self.shape);
                (index, element)
            })
    }

    /// Creates a parallel consuming iterator, that is, one that moves each
    /// element out of the matrix along with its index.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// use rayon::prelude::*;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
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
        self.data
            .into_par_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened(index, self.order, self.shape);
                (index, element)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

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
    fn test_par_map_ref() {
        let mut matrix_i32 = matrix![[0, 1, 2], [3, 4, 5]];

        let matrix_f64 = matrix_i32.par_map_ref(|x| *x as f64);
        assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        matrix_i32.switch_order();

        let mut matrix_f64 = matrix_i32.par_map_ref(|x| *x as f64);
        matrix_f64.switch_order();
        assert_eq!(matrix_f64, matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    }

    #[test]
    fn test_par_iter_elements() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        let sum = matrix.par_iter_elements().sum::<i32>();
        assert_eq!(sum, 15);

        matrix.switch_order();

        let sum = matrix.par_iter_elements().sum::<i32>();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_par_iter_elements_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix
            .par_iter_elements_mut()
            .for_each(|element| *element += 1);
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();

        matrix
            .par_iter_elements_mut()
            .for_each(|element| *element -= 1);
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_into_par_iter_elements() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        let sum = matrix.clone().into_par_iter_elements().sum::<i32>();
        assert_eq!(sum, 15);

        matrix.switch_order();

        let sum = matrix.clone().into_par_iter_elements().sum::<i32>();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_par_iter_elements_with_index() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix
            .par_iter_elements_with_index()
            .for_each(|(index, element)| {
                assert_eq!(element, &matrix[index]);
            });

        matrix.switch_order();

        matrix
            .par_iter_elements_with_index()
            .for_each(|(index, element)| {
                assert_eq!(element, &matrix[index]);
            });
    }

    #[test]
    fn test_par_iter_elements_mut_with_index() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix
            .par_iter_elements_mut_with_index()
            .for_each(|(index, element)| {
                *element += index.row as i32 + index.col as i32;
            });
        assert_eq!(matrix, matrix![[0, 2, 4], [4, 6, 8]]);

        matrix.switch_order();

        matrix
            .par_iter_elements_mut_with_index()
            .for_each(|(index, element)| {
                *element -= index.row as i32 + index.col as i32;
            });
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_into_par_iter_elements_with_index() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix
            .clone()
            .into_par_iter_elements_with_index()
            .for_each(|(index, element)| {
                assert_eq!(element, matrix[index]);
            });

        matrix.switch_order();

        matrix
            .clone()
            .into_par_iter_elements_with_index()
            .for_each(|(index, element)| {
                assert_eq!(element, matrix[index]);
            });
    }
}
