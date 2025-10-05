//! Traits for matrix conversion.

use crate::error::Result;

/// A trait for matrix conversion from a sequence of rows.
pub trait FromRows<S>: Sized {
    /// Converts from a sequence of rows.
    fn from_rows(value: S) -> Self;
}

/// A trait for fallible matrix conversion from a sequence of rows.
pub trait TryFromRows<S>: Sized {
    /// Attempts to convert from a sequence of rows.
    fn try_from_rows(value: S) -> Result<Self>;
}

/// A trait for matrix conversion from an iterator over rows.
pub trait FromRowIterator<R, T>: Sized
where
    R: IntoIterator<Item = T>,
{
    /// Converts from an iterator over rows.
    fn from_row_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = R>;
}

impl<M, S> TryFromRows<S> for M
where
    M: FromRows<S>,
{
    fn try_from_rows(value: S) -> Result<Self> {
        Ok(Self::from_rows(value))
    }
}

/// A trait for matrix conversion from a sequence of columns.
pub trait FromCols<S>: Sized {
    /// Converts from a sequence of columns.
    fn from_cols(value: S) -> Self;
}

/// A trait for fallible matrix conversion from a sequence of columns.
pub trait TryFromCols<S>: Sized {
    /// Attempts to convert from a sequence of columns.
    fn try_from_cols(value: S) -> Result<Self>;
}

/// A trait for matrix conversion from an iterator over columns.
pub trait FromColIterator<C, T>: Sized
where
    C: IntoIterator<Item = T>,
{
    /// Converts from an iterator over columns.
    fn from_col_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = C>;
}

impl<M, S> TryFromCols<S> for M
where
    M: FromCols<S>,
{
    fn try_from_cols(value: S) -> Result<Self> {
        Ok(Self::from_cols(value))
    }
}

/// A trait for matrix conversion to a sequence of rows.
pub trait IntoRows<S> {
    /// Converts to a sequence of rows.
    fn into_rows(self) -> S;
}

/// A trait for matrix conversion to a sequence of columns.
pub trait IntoCols<S> {
    /// Converts to a sequence of columns.
    fn into_cols(self) -> S;
}
