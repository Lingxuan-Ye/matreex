use super::layout::{Order, OrderKind, Stride};
use crate::index::Index;

impl Index {
    pub(super) fn from_flattened<O>(index: usize, stride: Stride) -> Self
    where
        O: Order,
    {
        let major = index / stride.major();
        let minor = (index % stride.major()) / stride.minor();
        match O::KIND {
            OrderKind::RowMajor => Self::new(major, minor),
            OrderKind::ColMajor => Self::new(minor, major),
        }
    }

    pub(super) fn to_flattened<O>(self, stride: Stride) -> usize
    where
        O: Order,
    {
        let (major, minor) = match O::KIND {
            OrderKind::RowMajor => (self.row, self.col),
            OrderKind::ColMajor => (self.col, self.row),
        };
        major * stride.major() + minor * stride.minor()
    }
}
