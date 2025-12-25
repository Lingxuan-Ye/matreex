use super::Matrix;
use super::layout::Order;
use crate::index::Index;

impl<L, LO, R, RO> PartialEq<Matrix<R, RO>> for Matrix<L, LO>
where
    L: PartialEq<R>,
    LO: Order,
    RO: Order,
{
    fn eq(&self, other: &Matrix<R, RO>) -> bool {
        if self.shape() != other.shape() {
            false
        } else if LO::KIND == RO::KIND {
            self.data == other.data
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = other.stride();
            self.data.iter().enumerate().all(|(index, lhs)| {
                let index =
                    Index::from_flattened::<LO>(index, lhs_stride).to_flattened::<RO>(rhs_stride);
                let rhs = unsafe { other.data.get_unchecked(index) };
                lhs == rhs
            })
        }
    }
}

impl<T, O> Eq for Matrix<T, O>
where
    T: PartialEq,
    O: Order,
{
}

#[cfg(test)]
mod tests {
    use super::super::layout::{ColMajor, Layout, RowMajor};
    use super::*;
    use crate::mock::{MockL, MockR};
    use crate::shape::Shape;
    use alloc::vec;

    #[test]
    fn test_eq() {
        let lhs: Matrix<MockL<i32>, RowMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockL(1), MockL(2), MockL(3), MockL(4), MockL(5), MockL(6)];
            Matrix { layout, data }
        };
        let rhs: Matrix<MockR<i32>, RowMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockR(1), MockR(2), MockR(3), MockR(4), MockR(5), MockR(6)];
            Matrix { layout, data }
        };
        assert_eq!(lhs, rhs);

        let lhs: Matrix<MockL<i32>, RowMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockL(1), MockL(2), MockL(3), MockL(4), MockL(5), MockL(6)];
            Matrix { layout, data }
        };
        let rhs: Matrix<MockR<i32>, ColMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockR(1), MockR(4), MockR(2), MockR(5), MockR(3), MockR(6)];
            Matrix { layout, data }
        };
        assert_eq!(lhs, rhs);

        let lhs: Matrix<MockL<i32>, ColMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockL(1), MockL(4), MockL(2), MockL(5), MockL(3), MockL(6)];
            Matrix { layout, data }
        };
        let rhs: Matrix<MockR<i32>, RowMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockR(1), MockR(2), MockR(3), MockR(4), MockR(5), MockR(6)];
            Matrix { layout, data }
        };
        assert_eq!(lhs, rhs);

        let lhs: Matrix<MockL<i32>, ColMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockL(1), MockL(4), MockL(2), MockL(5), MockL(3), MockL(6)];
            Matrix { layout, data }
        };
        let rhs: Matrix<MockR<i32>, ColMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![MockR(1), MockR(4), MockR(2), MockR(5), MockR(3), MockR(6)];
            Matrix { layout, data }
        };
        assert_eq!(lhs, rhs);
    }
}
