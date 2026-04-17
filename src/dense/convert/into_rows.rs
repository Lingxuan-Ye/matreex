use super::super::Matrix;
use super::super::layout::{Order, OrderKind};
use crate::convert::IntoRows;
use crate::index::Index;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

impl<T, O> IntoRows<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn into_rows(self) -> Box<[Box<[T]>]> {
        let shape = self.shape();

        match O::KIND {
            OrderKind::RowMajor => {
                let mut iter = self.data.into_iter();
                (0..shape.nrows)
                    .map(|_| iter.by_ref().take(shape.ncols).collect())
                    .collect()
            }

            OrderKind::ColMajor => {
                let mut output: Box<[Box<[MaybeUninit<T>]>]> = (0..shape.nrows)
                    .map(|_| Box::new_uninit_slice(shape.ncols))
                    .collect();
                let mut index = Index::new(0, 0);
                for element in self.data {
                    if index.row == shape.nrows {
                        index.row = 0;
                        index.col += 1;
                    }
                    unsafe {
                        output
                            .get_unchecked_mut(index.row)
                            .get_unchecked_mut(index.col)
                            .write(element);
                    }
                    index.row += 1;
                }
                let ptr = Box::into_raw(output) as *mut [Box<[T]>];
                unsafe { Box::from_raw(ptr) }
            }
        }
    }
}

impl<T, O> IntoRows<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    fn into_rows(self) -> Vec<Box<[T]>> {
        IntoRows::<Box<[Box<[T]>]>>::into_rows(self).into_vec()
    }
}

impl<T, O> IntoRows<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    fn into_rows(self) -> Box<[Vec<T>]> {
        let shape = self.shape();

        match O::KIND {
            OrderKind::RowMajor => {
                let mut iter = self.data.into_iter();
                (0..shape.nrows)
                    .map(|_| iter.by_ref().take(shape.ncols).collect())
                    .collect()
            }

            OrderKind::ColMajor => {
                let mut output: Box<[Vec<T>]> = (0..shape.nrows)
                    .map(|_| Vec::with_capacity(shape.ncols))
                    .collect();
                let mut index = Index::new(0, 0);
                for element in self.data {
                    if index.row == shape.nrows {
                        index.row = 0;
                        index.col += 1;
                    }
                    unsafe {
                        output
                            .get_unchecked_mut(index.row)
                            .as_mut_ptr()
                            .add(index.col)
                            .write(element);
                    }
                    index.row += 1;
                }
                for row in &mut output {
                    unsafe {
                        row.set_len(shape.ncols);
                    }
                }
                output
            }
        }
    }
}

impl<T, O> IntoRows<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    fn into_rows(self) -> Vec<Vec<T>> {
        IntoRows::<Box<[Vec<T>]>>::into_rows(self).into_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::TryFromRows;
    use crate::dispatch_unary;
    use alloc::vec;

    #[test]
    fn test_into_rows() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Box<[Box<[i32]>]> = matrix.into_rows();
            let expected: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Vec<Box<[i32]>> = matrix.into_rows();
            let expected: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Box<[Vec<i32>]> = matrix.into_rows();
            let expected: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Vec<Vec<i32>> = matrix.into_rows();
            let expected: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            assert_eq!(output, expected);
        }}
    }
}
