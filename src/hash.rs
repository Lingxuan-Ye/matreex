use crate::Matrix;
use core::hash::{Hash, Hasher};

impl<T> Hash for Matrix<T>
where
    T: Hash,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.shape().hash(state);
        for row in self.iter_rows() {
            for element in row {
                element.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::matrix;
    use crate::order::Order;
    use std::hash::{DefaultHasher, Hash, Hasher};

    #[test]
    fn test_hash() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut matrix = matrix.clone();

            matrix.set_order(Order::RowMajor);
            let row_major_hash = {
                let mut hasher = DefaultHasher::new();
                matrix.hash(&mut hasher);
                hasher.finish()
            };

            matrix.set_order(Order::ColMajor);
            let col_major_hash = {
                let mut hasher = DefaultHasher::new();
                matrix.hash(&mut hasher);
                hasher.finish()
            };

            assert_eq!(row_major_hash, col_major_hash);
        }

        {
            let mut matrix = matrix.clone();

            matrix.order = Order::RowMajor;
            let row_major_hash = {
                let mut hasher = DefaultHasher::new();
                matrix.hash(&mut hasher);
                hasher.finish()
            };

            matrix.order = Order::ColMajor;
            let col_major_hash = {
                let mut hasher = DefaultHasher::new();
                matrix.hash(&mut hasher);
                hasher.finish()
            };

            assert_ne!(row_major_hash, col_major_hash);
        }
    }
}
