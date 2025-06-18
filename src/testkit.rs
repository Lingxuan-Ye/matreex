//! Provides helper functions and mock types for tests.
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

pub(crate) mod mock {
    use core::ops::Neg;
    use core::ops::{Add, AddAssign};
    use core::ops::{Div, DivAssign};
    use core::ops::{Mul, MulAssign};
    use core::ops::{Rem, RemAssign};
    use core::ops::{Sub, SubAssign};

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub(crate) struct MockL(pub(crate) i32);

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub(crate) struct MockR(pub(crate) i32);

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub(crate) struct MockT(pub(crate) i32);

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub(crate) struct MockU(pub(crate) i32);

    impl Add<MockR> for MockL {
        type Output = MockU;

        fn add(self, rhs: MockR) -> Self::Output {
            MockU(self.0 + rhs.0)
        }
    }

    impl AddAssign<MockR> for MockL {
        fn add_assign(&mut self, rhs: MockR) {
            self.0 += rhs.0;
        }
    }

    impl Sub<MockR> for MockL {
        type Output = MockU;

        fn sub(self, rhs: MockR) -> Self::Output {
            MockU(self.0 - rhs.0)
        }
    }

    impl SubAssign<MockR> for MockL {
        fn sub_assign(&mut self, rhs: MockR) {
            self.0 -= rhs.0;
        }
    }

    impl Mul<MockR> for MockL {
        type Output = MockU;

        fn mul(self, rhs: MockR) -> Self::Output {
            MockU(self.0 * rhs.0)
        }
    }

    impl MulAssign<MockR> for MockL {
        fn mul_assign(&mut self, rhs: MockR) {
            self.0 *= rhs.0;
        }
    }

    impl Div<MockR> for MockL {
        type Output = MockU;

        fn div(self, rhs: MockR) -> Self::Output {
            MockU(self.0 / rhs.0)
        }
    }

    impl DivAssign<MockR> for MockL {
        fn div_assign(&mut self, rhs: MockR) {
            self.0 /= rhs.0;
        }
    }

    impl Rem<MockR> for MockL {
        type Output = MockU;

        fn rem(self, rhs: MockR) -> Self::Output {
            MockU(self.0 % rhs.0)
        }
    }

    impl RemAssign<MockR> for MockL {
        fn rem_assign(&mut self, rhs: MockR) {
            self.0 %= rhs.0;
        }
    }

    impl Neg for MockT {
        type Output = MockU;

        fn neg(self) -> Self::Output {
            MockU(-self.0)
        }
    }

    impl Add for MockU {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }
}
