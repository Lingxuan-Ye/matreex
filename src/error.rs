//! Error handling for the crate.

/// An alias for [`std::result::Result`].
pub type Result<T> = std::result::Result<T, Error>;

/// An enum for error types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Error when matrix size exceeds [`usize::MAX`], which is, in fact
    /// pointless, since a matrix can only store up to [`isize::MAX`] bytes
    /// of data.
    SizeOverflow,

    /// Error when the size of the shape does not match the length of the
    /// underlying data.
    ///
    /// Ensuring this equality is crucial because if the size exceeds the
    /// length, indexing into the matrix may result in out-of-bounds memory
    /// access, leading to *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    SizeMismatch,

    /// Error when the capacity exceeds [`isize::MAX`].
    ///
    /// Refer to [`Vec::with_capacity`] for more information.
    CapacityOverflow,

    /// Error when creating a matrix from a sequence of vectors with
    /// inconsistent lengths.
    LengthInconsistent,

    /// Error for accessing an index out of bounds.
    IndexOutOfBounds,

    /// Error when a square matrix is required but the current one does not
    /// satisfy this requirement.
    SquareMatrixRequired,

    /// Error when the shapes of two matrices are not conformable for the
    /// intended operation.
    ShapeNotConformable,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = match self {
            Self::SizeOverflow => "size overflow",
            Self::SizeMismatch => "size mismatch",
            Self::CapacityOverflow => "capacity overflow",
            Self::LengthInconsistent => "length inconsistent",
            Self::IndexOutOfBounds => "index out of bounds",
            Self::SquareMatrixRequired => "square matrix required",
            Self::ShapeNotConformable => "shape not conformable",
        };
        write!(f, "{content}")
    }
}

impl std::error::Error for Error {}
