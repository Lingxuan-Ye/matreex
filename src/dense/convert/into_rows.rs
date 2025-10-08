use super::super::Matrix;
use super::super::layout::{Order, OrderKind};
use crate::convert::IntoRows;
use crate::index::Index;
use alloc::boxed::Box;
use alloc::vec::Vec;

impl<T, O> IntoRows<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn into_rows(self) -> Box<[Box<[T]>]> {
        IntoRows::<Box<[Vec<T>]>>::into_rows(self)
            .into_iter()
            .map(|row| row.into_boxed_slice())
            .collect()
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
        let nrows = shape.nrows();
        let ncols = shape.ncols();
        let stride = self.stride();

        match O::KIND {
            OrderKind::RowMajor => {
                let mut iter = self.data.into_iter();
                (0..nrows)
                    .map(|_| iter.by_ref().take(ncols).collect())
                    .collect()
            }

            OrderKind::ColMajor => {
                let mut output: Box<_> = (0..nrows).map(|_| Vec::with_capacity(ncols)).collect();
                self.data
                    .into_iter()
                    .enumerate()
                    .for_each(|(index, element)| unsafe {
                        let index = Index::from_flattened::<O>(index, stride);
                        output.get_unchecked_mut(index.row).push(element)
                    });
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
    use crate::convert::FromRows;
    use crate::dispatch_unary;
    use alloc::vec;

    #[test]
    fn test_into_rows() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::from_rows([[1, 2, 3], [4, 5, 6]]);
            let expected: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output: Box<[Box<[i32]>]> = matrix.into_rows();
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::from_rows([[1, 2, 3], [4, 5, 6]]);
            let expected: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output: Vec<Box<[i32]>> = matrix.into_rows();
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::from_rows([[1, 2, 3], [4, 5, 6]]);
            let expected: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output: Box<[Vec<i32>]> = matrix.into_rows();
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::from_rows([[1, 2, 3], [4, 5, 6]]);
            let expected: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let output: Vec<Vec<i32>> = matrix.into_rows();
            assert_eq!(output, expected);
        }}
    }
}
