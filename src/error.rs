//! This module serves for error handling.

/// An enum for error types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Error when matrix size exceeds [`usize::MAX`], which is, in fact
    /// pointless, since a matrix can store at most [`isize::MAX`] bytes.
    SizeOverflow,

    /// Error when attempting to reshape matrix to a different size.
    SizeMismatch,

    /// Error when total bytes exceeds [`isize::MAX`]. Refer to
    /// [`Vec::with_capacity`] for more information.
    CapacityExceeded,

    /// Error for inconsistent row or column length.
    LengthInconsistent,

    /// Error for accessing an index out of bounds.
    IndexOutOfBounds,

    /// Error when shape is not conformable.
    NotConformable,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let content = match self {
            Self::SizeOverflow => "size overflows",
            Self::SizeMismatch => "size does not match",
            Self::CapacityExceeded => "capacity exceeds",
            Self::LengthInconsistent => "length inconsistent",
            Self::IndexOutOfBounds => "index out of bounds",
            Self::NotConformable => "shape not conformable",
        };
        write!(f, "{content}")
    }
}

impl std::error::Error for Error {}

/// An alias for [`core::result::Result`].
pub type Result<T> = core::result::Result<T, Error>;
