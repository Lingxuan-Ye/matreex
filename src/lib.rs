#![no_std]

extern crate alloc;

pub use self::dense::layout::{ColMajor, RowMajor};
pub use self::dense::{ColMajorMatrix, Matrix, RowMajorMatrix};
pub use self::error::{Error, Result};
pub use self::index::{Index, WrappingIndex};
pub use self::shape::Shape;

pub mod convert;
pub mod dense;
pub mod error;
pub mod index;
pub mod shape;

#[cfg(feature = "parallel")]
pub mod parallel;
