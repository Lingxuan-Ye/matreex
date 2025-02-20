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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.par_apply(|x| *x += 2);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
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
    /// let matrix_i32 = matrix![[1, 2, 3], [4, 5, 6]];
    /// let matrix_f64 = matrix_i32.par_map(|x| x as f64);
    /// assert_eq!(matrix_f64, matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
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
    /// let matrix_i32 = matrix![[1, 2, 3], [4, 5, 6]];
    /// let matrix_f64 = matrix_i32.par_map_ref(|x| *x as f64);
    /// assert_eq!(matrix_f64, matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// ```
    ///
    /// [`par_map`]: Matrix::par_map
    #[inline]
    pub fn par_map_ref<'a, U, F>(&'a self, f: F) -> Matrix<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&'a T) -> U + Sync + Send,
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let sum = matrix.par_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 21);
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .par_iter_elements_mut()
    ///     .for_each(|element| *element += 2);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
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
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let sum = matrix.into_par_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 21);
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
    /// use matreex::matrix;
    /// use rayon::prelude::*;
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
        fn add_two(x: &mut i32) {
            *x += 2;
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order
        {
            let mut matrix = matrix.clone();

            matrix.par_apply(add_two);
            assert_eq!(matrix, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.par_apply(add_two);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }

    #[test]
    fn test_par_map() {
        fn to_f64(x: i32) -> f64 {
            x as f64
        }

        let matrix_i32 = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // default order
        {
            let matrix_i32 = matrix_i32.clone();

            let matrix_f64 = matrix_i32.par_map(to_f64);
            assert_eq!(matrix_f64, expected);
        }

        // alternative order
        {
            let mut matrix_i32 = matrix_i32.clone();
            matrix_i32.switch_order();

            let mut matrix_f64 = matrix_i32.par_map(to_f64);
            matrix_f64.switch_order();
            assert_eq!(matrix_f64, expected);
        }
    }

    #[test]
    fn test_par_map_ref() {
        fn to_f64(x: &i32) -> f64 {
            *x as f64
        }

        let matrix_i32 = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // default order
        {
            let matrix_f64 = matrix_i32.par_map_ref(to_f64);
            assert_eq!(matrix_f64, expected);
        }

        // alternative order
        {
            let mut matrix_i32 = matrix_i32.clone();
            matrix_i32.switch_order();

            let mut matrix_f64 = matrix_i32.par_map_ref(to_f64);
            matrix_f64.switch_order();
            assert_eq!(matrix_f64, expected);
        }

        // to matrix of references
        {
            let matrix_ref = matrix_i32.par_map_ref(|x| x);
            assert_eq!(matrix_ref, matrix![[&1, &2, &3], [&4, &5, &6]]);
        }
    }

    #[test]
    fn test_par_iter_elements() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = 21;

        // default order
        {
            let sum = matrix.par_iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            let sum = matrix.par_iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn test_par_iter_elements_mut() {
        fn add_two(x: &mut i32) {
            *x += 2;
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order
        {
            let mut matrix = matrix.clone();

            matrix.par_iter_elements_mut().for_each(add_two);
            assert_eq!(matrix, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.par_iter_elements_mut().for_each(add_two);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }

    #[test]
    fn test_into_par_iter_elements() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = 21;

        // default order
        {
            let matrix = matrix.clone();

            let sum = matrix.into_par_iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            let sum = matrix.into_par_iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn test_par_iter_elements_with_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let test_index = |(index, element)| {
            assert_eq!(element, &matrix[index]);
        };

        // default order
        {
            matrix.par_iter_elements_with_index().for_each(test_index);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.par_iter_elements_with_index().for_each(test_index);
        }
    }

    #[test]
    fn test_par_iter_elements_mut_with_index() {
        fn add_index((index, element): (Index, &mut i32)) {
            *element += index.row as i32 + index.col as i32;
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[1, 3, 5], [5, 7, 9]];

        // default order
        {
            let mut matrix = matrix.clone();

            matrix
                .par_iter_elements_mut_with_index()
                .for_each(add_index);
            assert_eq!(matrix, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix
                .par_iter_elements_mut_with_index()
                .for_each(add_index);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }

    #[test]
    fn test_into_par_iter_elements_with_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let test_index = |(index, element)| {
            assert_eq!(element, matrix[index]);
        };

        // default order
        {
            let matrix = matrix.clone();

            matrix
                .into_par_iter_elements_with_index()
                .for_each(test_index);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix
                .into_par_iter_elements_with_index()
                .for_each(test_index);
        }
    }
}
