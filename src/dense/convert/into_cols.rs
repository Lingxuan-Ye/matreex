use super::super::Matrix;
use super::super::layout::{Order, OrderKind};
use crate::convert::IntoCols;
use crate::index::Index;
use alloc::boxed::Box;
use alloc::vec::Vec;

impl<T, O> IntoCols<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn into_cols(self) -> Box<[Box<[T]>]> {
        IntoCols::<Box<[Vec<T>]>>::into_cols(self)
            .into_iter()
            .map(|col| col.into_boxed_slice())
            .collect()
    }
}

impl<T, O> IntoCols<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    fn into_cols(self) -> Vec<Box<[T]>> {
        IntoCols::<Box<[Box<[T]>]>>::into_cols(self).into_vec()
    }
}

impl<T, O> IntoCols<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    fn into_cols(self) -> Box<[Vec<T>]> {
        let shape = self.shape();
        let nrows = shape.nrows();
        let ncols = shape.ncols();
        let stride = self.stride();

        match O::KIND {
            OrderKind::RowMajor => {
                let mut output: Box<_> = (0..ncols).map(|_| Vec::with_capacity(nrows)).collect();
                self.data
                    .into_iter()
                    .enumerate()
                    .for_each(|(index, element)| unsafe {
                        let index = Index::from_flattened::<O>(index, stride);
                        output.get_unchecked_mut(index.col).push(element)
                    });
                output
            }

            OrderKind::ColMajor => {
                let mut iter = self.data.into_iter();
                (0..ncols)
                    .map(|_| iter.by_ref().take(nrows).collect())
                    .collect()
            }
        }
    }
}

impl<T, O> IntoCols<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    fn into_cols(self) -> Vec<Vec<T>> {
        IntoCols::<Box<[Vec<T>]>>::into_cols(self).into_vec()
    }
}
