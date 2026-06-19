use super::super::Matrix;
use super::super::layout::{Order, OrderKind};
use crate::convert::IntoCols;
use crate::index::Index;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

impl<T, O> IntoCols<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn into_cols(self) -> Box<[Box<[T]>]> {
        let shape = self.shape();

        match O::KIND {
            OrderKind::RowMajor => {
                let mut output: Box<[Box<[MaybeUninit<T>]>]> = (0..shape.ncols)
                    .map(|_| Box::new_uninit_slice(shape.nrows))
                    .collect();
                let mut index = Index::new(0, 0);
                for element in self.data {
                    if index.col == shape.ncols {
                        index.col = 0;
                        index.row += 1;
                    }
                    unsafe {
                        output
                            .get_unchecked_mut(index.col)
                            .get_unchecked_mut(index.row)
                            .write(element);
                    }
                    index.col += 1;
                }
                output
                    .into_iter()
                    .map(|col| unsafe { col.assume_init() })
                    .collect()
            }

            OrderKind::ColMajor => {
                let mut iter = self.data.into_iter();
                (0..shape.ncols)
                    .map(|_| iter.by_ref().take(shape.nrows).collect())
                    .collect()
            }
        }
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

        match O::KIND {
            OrderKind::RowMajor => {
                let mut output: Box<[Vec<T>]> = (0..shape.ncols)
                    .map(|_| Vec::with_capacity(shape.nrows))
                    .collect();
                let mut index = Index::new(0, 0);
                for element in self.data {
                    if index.col == shape.ncols {
                        index.col = 0;
                        index.row += 1;
                    }
                    unsafe {
                        output
                            .get_unchecked_mut(index.col)
                            .as_mut_ptr()
                            .add(index.row)
                            .write(element);
                    }
                    index.col += 1;
                }
                for col in &mut output {
                    unsafe {
                        col.set_len(shape.nrows);
                    }
                }
                output
            }

            OrderKind::ColMajor => {
                let mut iter = self.data.into_iter();
                (0..shape.ncols)
                    .map(|_| iter.by_ref().take(shape.nrows).collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::TryFromRows;
    use crate::dispatch_unary;
    use crate::error::Result;
    use alloc::vec;

    #[test]
    fn test_into_cols() -> Result<()> {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::try_from_rows([[1, 4], [2, 5], [3, 6]])?;
            let output: Box<[Box<[i32]>]> = matrix.into_cols();
            let expected: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 4], [2, 5], [3, 6]])?;
            let output: Vec<Box<[i32]>> = matrix.into_cols();
            let expected: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 4], [2, 5], [3, 6]])?;
            let output: Box<[Vec<i32>]> = matrix.into_cols();
            let expected: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 4], [2, 5], [3, 6]])?;
            let output: Vec<Vec<i32>> = matrix.into_cols();
            let expected: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            assert_eq!(output, expected);
        }}

        Ok(())
    }
}
