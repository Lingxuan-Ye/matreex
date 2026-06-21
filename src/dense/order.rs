//! Types for matrix storage order.

/// A marker type representing row-major order.
///
/// In this order, elements are stored row by row, with consecutive
/// elements of a row stored contiguously in memory.
///
/// For column-major order see [`ColMajor`].
#[derive(Debug)]
pub struct RowMajor;

/// A marker type representing column-major order.
///
/// In this order, elements are stored column by column, with consecutive
/// elements of a column stored contiguously in memory.
///
/// For row-major order see [`RowMajor`].
#[derive(Debug)]
pub struct ColMajor;

/// A sealed trait restricting allowed storage orders.
///
/// The allowed orders are:
///
/// - [`RowMajor`]
/// - [`ColMajor`]
pub trait Order: internal::Sealed {
    #[doc(hidden)]
    const KIND: OrderKind;
}

impl Order for RowMajor {
    const KIND: OrderKind = OrderKind::RowMajor;
}

impl Order for ColMajor {
    const KIND: OrderKind = OrderKind::ColMajor;
}

#[doc(hidden)]
#[derive(Debug, PartialEq)]
pub enum OrderKind {
    RowMajor,
    ColMajor,
}

mod internal {
    pub trait Sealed {}

    impl Sealed for super::RowMajor {}
    impl Sealed for super::ColMajor {}
}
