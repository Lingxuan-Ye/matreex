//! Traits for matrix conversion.

use crate::error::Result;

/// A trait for conversion from a sequence of rows.
pub trait FromRows<T>: Sized {
    /// Converts from a sequence of rows.
    fn from_rows(value: T) -> Self;
}

/// A trait for fallible conversion from a sequence of rows.
pub trait TryFromRows<T>: Sized {
    /// Attempts to convert from a sequence of rows.
    fn try_from_rows(value: T) -> Result<Self>;
}

/// A trait for conversion from an iterator over rows.
pub trait FromRowIterator<T, V>: Sized
where
    V: IntoIterator<Item = T>,
{
    /// Converts from an iterator over rows.
    fn from_row_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>;
}

/// A trait for conversion from a sequence of columns.
pub trait FromCols<T>: Sized {
    /// Converts from a sequence of columns.
    fn from_cols(value: T) -> Self;
}

/// A trait for fallible conversion from a sequence of columns.
pub trait TryFromCols<T>: Sized {
    /// Attempts to convert from a sequence of columns.
    fn try_from_cols(value: T) -> Result<Self>;
}

/// A trait for conversion from an iterator over columns.
pub trait FromColIterator<T, V>: Sized
where
    V: IntoIterator<Item = T>,
{
    /// Converts from an iterator over columns.
    fn from_col_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>;
}
