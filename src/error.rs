//! Error handling for the crate.

/// An alias for [`core::result::Result`].
pub type Result<T> = core::result::Result<T, Error>;

/// An enum for error types.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Error {
    /// Error when the size exceeds [`usize::MAX`].
    SizeOverflow,

    /// Error when the size of the shape does not match the length of the
    /// underlying data.
    SizeMismatch,

    /// Error when attempting to allocate more than [`isize::MAX`] bytes
    /// of memory.
    ///
    /// Refer to [`vec`] and *[The Rustonomicon]* for more information.
    ///
    /// [`vec`]: mod@alloc::vec
    /// [The Rustonomicon]: https://doc.rust-lang.org/stable/nomicon/vec/vec-alloc.html#allocating-memory
    CapacityOverflow,

    /// Error when converting to a matrix from rows or columns with
    /// inconsistent lengths.
    LengthInconsistent,

    /// Error for accessing an index out of bounds.
    IndexOutOfBounds,

    /// Error when a square matrix is required but the current one does
    /// not satisfy this requirement.
    SquareMatrixRequired,

    /// Error when the shapes of two matrices are not conformable for the
    /// intended operation.
    ShapeNotConformable,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let content = match self {
            Self::SizeOverflow => "size overflow",
            Self::SizeMismatch => "size mismatch",
            Self::CapacityOverflow => "capacity overflow",
            Self::LengthInconsistent => "length inconsistent",
            Self::IndexOutOfBounds => "index out of bounds",
            Self::SquareMatrixRequired => "square matrix required",
            Self::ShapeNotConformable => "shape not conformable",
        };
        f.write_str(content)
    }
}

impl core::error::Error for Error {}
