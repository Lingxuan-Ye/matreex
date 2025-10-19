//! Traits for matrix conversion.

use crate::error::Result;

/// A trait for matrix conversion from a sequence of rows.
pub trait TryFromRows<S>: Sized {
    /// Attempts to convert from a sequence of rows.
    fn try_from_rows(value: S) -> Result<Self>;
}

/// A trait for matrix conversion from an iterator over rows.
pub trait FromRowIterator<R, T>
where
    R: IntoIterator<Item = T>,
{
    /// Converts from an iterator over rows.
    fn from_row_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = R>;
}

/// A trait for matrix conversion from a sequence of columns.
pub trait TryFromCols<S>: Sized {
    /// Attempts to convert from a sequence of columns.
    fn try_from_cols(value: S) -> Result<Self>;
}

/// A trait for matrix conversion from an iterator over columns.
pub trait FromColIterator<C, T>
where
    C: IntoIterator<Item = T>,
{
    /// Converts from an iterator over columns.
    fn from_col_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = C>;
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
