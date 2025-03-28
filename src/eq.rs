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
                let index = AxisIndex::from_flattened(index, self.shape)
                    .swap()
                    .to_flattened(other.shape);
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

    #[test]
    fn test_eq() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2, 3], [4, 5, 6]];

        // default order & default order
        {
            assert_eq!(lhs.order, rhs.order);
            assert_eq!(lhs.shape, rhs.shape);
            assert_eq!(lhs.data, rhs.data);
            assert_eq!(lhs, rhs);
        }

        // default order & alternative order
        {
            let mut rhs = rhs.clone();
            rhs.switch_order();

            assert_ne!(lhs.order, rhs.order);
            assert_ne!(lhs.shape, rhs.shape);
            assert_ne!(lhs.data, rhs.data);
            assert_eq!(lhs, rhs);
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            lhs.switch_order();

            assert_ne!(lhs.order, rhs.order);
            assert_ne!(lhs.shape, rhs.shape);
            assert_ne!(lhs.data, rhs.data);
            assert_eq!(lhs, rhs);
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            assert_eq!(lhs.order, rhs.order);
            assert_eq!(lhs.shape, rhs.shape);
            assert_eq!(lhs.data, rhs.data);
            assert_eq!(lhs, rhs);
        }

        // more test cases

        {
            let rhs = matrix![[1, 2], [3, 4]];
            assert_ne!(lhs, rhs);
        }

        {
            let rhs = matrix![[1, 2], [3, 4], [5, 6]];
            assert_ne!(lhs, rhs);
        }

        {
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
            assert_ne!(lhs, rhs);
        }

        {
            let rhs = matrix![[2, 2, 2], [2, 2, 2]];
            assert_ne!(lhs, rhs);
        }
    }
}
