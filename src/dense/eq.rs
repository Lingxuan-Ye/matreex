use super::Matrix;
use super::layout::Order;
use crate::index::Index;

impl<L, R, LO, RO> PartialEq<Matrix<R, RO>> for Matrix<L, LO>
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
            self.data.iter().enumerate().all(|(index, left)| {
                let index =
                    Index::from_flattened::<LO>(index, lhs_stride).to_flattened::<RO>(rhs_stride);
                let right = unsafe { other.data.get_unchecked(index) };
                left == right
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
