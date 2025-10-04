use super::Matrix;
use super::layout::Order;
use crate::convert::{FromRowIterator, FromRows, TryFromRows};
use crate::error::{Error, Result};
use alloc::boxed::Box;
use alloc::vec::Vec;

mod from_cols;
mod from_rows;

impl<T, O, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, O>
where
    O: Order,
{
    fn from(value: [[T; C]; R]) -> Self {
        Self::from_rows(value)
    }
}

impl<T, O, const R: usize, const C: usize> From<Box<[[T; C]; R]>> for Matrix<T, O>
where
    O: Order,
{
    fn from(value: Box<[[T; C]; R]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, O, const C: usize> From<Box<[[T; C]]>> for Matrix<T, O>
where
    O: Order,
{
    fn from(value: Box<[[T; C]]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, O, const C: usize> From<Vec<[T; C]>> for Matrix<T, O>
where
    O: Order,
{
    fn from(value: Vec<[T; C]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, O, const R: usize, const C: usize> TryFrom<[Box<[T; C]>; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: [Box<[T; C]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize, const C: usize> TryFrom<Box<[Box<[T; C]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Box<[Box<[T; C]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const C: usize> TryFrom<Box<[Box<[T; C]>]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Box<[Box<[T; C]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const C: usize> TryFrom<Vec<Box<[T; C]>>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Vec<Box<[T; C]>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<[Box<[T]>; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: [Box<[T]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<Box<[Box<[T]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<[Vec<T>; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: [Vec<T>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<Box<[Vec<T>; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, V> FromIterator<V> for Matrix<T, O>
where
    O: Order,
    V: IntoIterator<Item = T>,
{
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        Self::from_row_iter(iter)
    }
}
