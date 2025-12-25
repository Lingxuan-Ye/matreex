extern crate std;

use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::thread_local;

thread_local! {
    static IN_SCOPE: Cell<bool> = const { Cell::new(false) };

    static INIT_COUNT: Cell<usize> = const { Cell::new(0) };
    static DROP_COUNT: Cell<usize> = const { Cell::new(0) };

    static ADD_COUNT: Cell<usize> = const { Cell::new(0) };
    static MUL_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[derive(Debug)]
pub(crate) struct Scope(PhantomData<*const ()>);

#[derive(Debug, PartialEq)]
pub(crate) struct MockZeroSized(());

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MockL<T>(pub(crate) T);

#[derive(Debug, Clone)]
pub(crate) struct MockR<T>(pub(crate) T);

#[derive(Debug, Clone)]
pub(crate) struct MockT<T>(pub(crate) T);

#[derive(Debug, Default, PartialEq)]
pub(crate) struct MockU<T>(pub(crate) T);

impl Scope {
    pub(crate) fn with<F>(f: F)
    where
        F: FnOnce(&Scope),
    {
        if IN_SCOPE.replace(true) {
            panic!("nested scope is not allowed");
        }

        f(&Self(PhantomData));

        IN_SCOPE.set(false);

        INIT_COUNT.set(0);
        DROP_COUNT.set(0);

        ADD_COUNT.set(0);
        MUL_COUNT.set(0);
    }

    pub(crate) fn init_count(&self) -> usize {
        INIT_COUNT.get()
    }

    pub(crate) fn drop_count(&self) -> usize {
        DROP_COUNT.get()
    }

    pub(crate) fn add_count(&self) -> usize {
        ADD_COUNT.get()
    }

    pub(crate) fn mul_count(&self) -> usize {
        MUL_COUNT.get()
    }
}

impl MockZeroSized {
    pub(crate) fn new() -> Self {
        if IN_SCOPE.get() {
            INIT_COUNT.with(|cell| cell.update(|count| count + 1));
        }
        Self(())
    }
}

impl Default for MockZeroSized {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MockZeroSized {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Drop for MockZeroSized {
    fn drop(&mut self) {
        if IN_SCOPE.try_with(Cell::get) == Ok(true) {
            let _ = DROP_COUNT.try_with(|cell| cell.update(|count| count + 1));
        }
    }
}

impl Add for MockZeroSized {
    type Output = Self;

    fn add(self, _: Self) -> Self::Output {
        if IN_SCOPE.get() {
            ADD_COUNT.with(|cell| cell.update(|count| count + 1));
        }
        Self::new()
    }
}

impl Mul for MockZeroSized {
    type Output = Self;

    fn mul(self, _: Self) -> Self::Output {
        if IN_SCOPE.get() {
            MUL_COUNT.with(|cell| cell.update(|count| count + 1));
        }
        Self::new()
    }
}

impl<L, R> PartialEq<MockR<R>> for MockL<L>
where
    L: PartialEq<R>,
{
    fn eq(&self, other: &MockR<R>) -> bool {
        self.0 == other.0
    }
}

impl<L, R, U> Add<MockR<R>> for MockL<L>
where
    L: Add<R, Output = U>,
{
    type Output = MockU<U>;

    fn add(self, rhs: MockR<R>) -> Self::Output {
        MockU(self.0 + rhs.0)
    }
}

impl<L, R> AddAssign<MockR<R>> for MockL<L>
where
    L: AddAssign<R>,
{
    fn add_assign(&mut self, rhs: MockR<R>) {
        self.0 += rhs.0;
    }
}

impl<L, R, U> Sub<MockR<R>> for MockL<L>
where
    L: Sub<R, Output = U>,
{
    type Output = MockU<U>;

    fn sub(self, rhs: MockR<R>) -> Self::Output {
        MockU(self.0 - rhs.0)
    }
}

impl<L, R> SubAssign<MockR<R>> for MockL<L>
where
    L: SubAssign<R>,
{
    fn sub_assign(&mut self, rhs: MockR<R>) {
        self.0 -= rhs.0;
    }
}

impl<L, R, U> Mul<MockR<R>> for MockL<L>
where
    L: Mul<R, Output = U>,
{
    type Output = MockU<U>;

    fn mul(self, rhs: MockR<R>) -> Self::Output {
        MockU(self.0 * rhs.0)
    }
}

impl<T> Add for MockU<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<T, U> Neg for MockT<T>
where
    T: Neg<Output = U>,
{
    type Output = MockU<U>;

    fn neg(self) -> Self::Output {
        MockU(-self.0)
    }
}
