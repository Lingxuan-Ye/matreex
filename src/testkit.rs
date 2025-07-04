//! Provides helper functions for tests.
//!
//! # Principles for Writing Tests
//!
//! - Prefer explicit to implicit.
//! - Prefer immutable to mutable.
//! - Prefer stateless to stateful.
//! - Prefer ownership to shared references.

use crate::Matrix;
use crate::order::Order;
use core::fmt::Debug;

/// Asserts two matrices are strictly equal, that is, they are identical
/// in all fields.
pub(crate) fn assert_strict_eq<T>(input_0: &Matrix<T>, input_1: &Matrix<T>)
where
    T: Debug + PartialEq,
{
    assert_eq!(input_0.order, input_1.order);
    assert_eq!(input_0.shape, input_1.shape);
    assert_eq!(input_0.data, input_1.data);
}

/// Asserts two matrices are loosely equal, that is, they are of the
/// same shape and all elements at corresponding positions are equal,
/// regardless of internal representations.
///
/// Prefer this to [`assert_eq!`] when writing tests.
pub(crate) fn assert_loose_eq<T>(input_0: &Matrix<T>, input_1: &Matrix<T>)
where
    T: Debug + PartialEq,
{
    assert_eq!(input_0, input_1);
}

/// Calls `testcase` with the input matrix for each order.
///
/// This method relies on the correctness of [`Matrix::set_order`].
pub(crate) fn for_each_order_unary<T, F>(input: Matrix<T>, testcase: F)
where
    T: Clone,
    F: Fn(Matrix<T>),
{
    // row-major
    {
        let mut input = input.clone();
        input.set_order(Order::RowMajor);
        testcase(input)
    }

    // col-major
    {
        let mut input = input;
        input.set_order(Order::ColMajor);
        testcase(input)
    }
}

/// Calls `testcase` with the input matrices for each combination of orders.
///
/// This method relies on the correctness of [`Matrix::set_order`].
pub(crate) fn for_each_order_binary<T, U, F>(input_0: Matrix<T>, input_1: Matrix<U>, testcase: F)
where
    T: Clone,
    U: Clone,
    F: Fn(Matrix<T>, Matrix<U>),
{
    // row-major & row-major
    {
        let mut input_0 = input_0.clone();
        let mut input_1 = input_1.clone();
        input_0.set_order(Order::RowMajor);
        input_1.set_order(Order::RowMajor);
        testcase(input_0, input_1)
    }

    // row-major & col-major
    {
        let mut input_0 = input_0.clone();
        let mut input_1 = input_1.clone();
        input_0.set_order(Order::RowMajor);
        input_1.set_order(Order::ColMajor);
        testcase(input_0, input_1)
    }

    // col-major & row-major
    {
        let mut input_0 = input_0.clone();
        let mut input_1 = input_1.clone();
        input_0.set_order(Order::ColMajor);
        input_1.set_order(Order::RowMajor);
        testcase(input_0, input_1)
    }

    // col-major & col-major
    {
        let mut input_0 = input_0;
        let mut input_1 = input_1;
        input_0.set_order(Order::ColMajor);
        input_1.set_order(Order::ColMajor);
        testcase(input_0, input_1)
    }
}
