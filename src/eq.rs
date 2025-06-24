use crate::Matrix;
use crate::index::AxisIndex;

impl<T> PartialEq for Matrix<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.order == other.order {
            self.shape == other.shape && self.data == other.data
        } else if self.major() == other.minor() && self.minor() == other.major() {
            self.data.iter().enumerate().all(|(index, left)| {
                let index = AxisIndex::from_flattened(index, self.stride())
                    .swap()
                    .to_flattened(other.stride());
                let right = unsafe { other.data.get_unchecked(index) };
                left == right
            })
        } else {
            false
        }
    }
}

impl<T> Eq for Matrix<T> where T: Eq {}

#[cfg(test)]
mod tests {
    use crate::matrix;
    use crate::testkit;

    #[test]
    fn test_eq() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert_eq!(lhs, rhs);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert_ne!(lhs, rhs);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert_ne!(lhs, rhs);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert_ne!(lhs, rhs);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert_ne!(lhs, rhs);
        });
    }
}
