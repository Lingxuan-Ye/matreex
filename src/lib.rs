//! A simple matrix implementation.
//!
//! # Quick Start
//!
//! First, we need to import `matrix!`.
//!
//! ```
//! use matreex::matrix;
//! ```
//!
//! ## Addition
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[0, 1, 2], [3, 4, 5]];
//! let rhs = matrix![[2, 2, 2], [2, 2, 2]];
//! assert_eq!(lhs + rhs, matrix![[2, 3, 4], [5, 6, 7]]);
//! ```
//!
//! ## Subtraction
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[0, 1, 2], [3, 4, 5]];
//! let rhs = matrix![[2, 2, 2], [2, 2, 2]];
//! assert_eq!(lhs - rhs, matrix![[-2, -1, 0], [1, 2, 3]]);
//! ```
//!
//! ## Multiplication
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[0, 1, 2], [3, 4, 5]];
//! let rhs = matrix![[0, 1], [2, 3], [4, 5]];
//! assert_eq!(lhs * rhs, matrix![[10, 13], [28, 40]]);
//! ```
//!
//! ## Division
//!
//! ```compile_fail
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
//! let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
//! assert_eq!(lhs / rhs, matrix![[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]);
//! ```
//!
//! Wait, matrix division isn't well-defined, remember? It won't compile. But
//! don't worry, you might just need to perform elementwise division:
//!
//! ```
//! # use matreex::matrix;
//! #
//! let lhs = matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
//! let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
//! assert_eq!(lhs.elementwise_div(&rhs), Ok(matrix![[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]));
//! ```
//!
//! Or scalar division:
//!
//! ```
//! # use matreex::matrix;
//! #
//! let matrix = matrix![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
//! assert_eq!(matrix / 2.0, matrix![[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]);
//!
//! let matrix = matrix![[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]];
//! assert_eq!(2.0 / matrix, matrix![[2.0, 1.0, 0.5], [0.25, 0.125, 0.0625]]);
//! ```
//!
//! Or maybe the inverse of a matrix?
//!
//! Nah, we don't have that yet.
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
