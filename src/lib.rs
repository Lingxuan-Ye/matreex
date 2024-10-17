//! A simple matrix implementation.
//!
//! # Quick Start
//!
//! ```
//! use matreex::matrix;
//!
//! let lhs = matrix![[0, 1, 2], [3, 4, 5]];
//! let rhs = matrix![[0, 1], [2, 3], [4, 5]];
//!
//! assert_eq!(lhs * rhs, matrix![[10, 13], [28, 40]]);
//! ```
//!
//! # FAQs
//!
//! ## Why named `matreex`?
//!
//! Hmm ... Who knows? Could be a name conflict.

pub use self::error::{Error, Result};
pub use self::matrix::index::Index;
pub use self::matrix::iter::{MatrixIter, VectorIter};
pub use self::matrix::order::Order;
pub use self::matrix::shape::Shape;
pub use self::matrix::Matrix;

pub mod error;
pub mod matrix;

mod macros;
