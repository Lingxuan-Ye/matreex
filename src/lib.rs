#![no_std]

extern crate alloc;

pub use self::error::{Error, Result};
pub use self::index::{Index, WrappingIndex};
pub use self::shape::Shape;

use self::dense::layout::RowMajor;

pub mod convert;
pub mod dense;
pub mod error;
pub mod index;
pub mod shape;

#[cfg(feature = "parallel")]
pub mod parallel;

mod macros;

/// An alias for [`dense::Matrix<T, RowMajor>`].
///
/// This provides a better experience for type inference than giving the type
/// parameter for the storage order a default of [`RowMajor`].
pub type Matrix<T> = self::dense::Matrix<T, RowMajor>;
