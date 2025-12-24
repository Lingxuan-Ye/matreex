use core::iter::FusedIterator;
use core::ptr::NonNull;

/// A struct representing a memory range.
///
/// The memory range may be partially or completely uninitialized. Reading
/// uninitialized parts of the memory range is *[undefined behavior]*.
///
/// # Invariants
///
/// - The memory range must be contained within a single allocated object.
/// - The memory range must be [valid] for both reads and writes.
/// - The memory range must be properly aligned, even if `T` has size 0.
///
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
/// [valid]: https://doc.rust-lang.org/core/ptr/index.html#safety
#[derive(Debug)]
pub(super) struct MemRange<T>(NonNull<[T]>);

impl<T> MemRange<T> {
    /// Creates a [`MemRange`].
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `start` must be non-null.
    /// - `start` must be properly aligned.
    /// - The memory range beginning at `start` with a size of `len * size_of::<T>()`
    ///   bytes must be contained within a single allocated object.
    /// - The memory range beginning at `start` with a size of `len * size_of::<T>()`
    ///   bytes must be [valid] for both reads and writes.
    ///
    /// [valid]: https://doc.rust-lang.org/core/ptr/index.html#safety
    pub(super) const unsafe fn new(start: *mut T, len: usize) -> Self {
        let start = unsafe { NonNull::new_unchecked(start) };
        Self(NonNull::slice_from_raw_parts(start, len))
    }

    /// Returns a pointer to the start of the memory range.
    ///
    /// Note that if the length of the memory range is `0`, the returned pointer is
    /// still inside the provenance of the allocated object, but may have `0` bytes
    /// it can read/write.
    fn start(&self) -> *mut T {
        self.0.as_ptr() as *mut T
    }

    /// Returns the length of the memory range.
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Initializes the memory range with the given value.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if the length of the memory range is `0`.
    ///
    /// If any part of the memory range has already been initialized, the original
    /// values will leak. However, this is considered safe.
    pub(super) unsafe fn init(&mut self, value: T)
    where
        T: Clone,
    {
        debug_assert_ne!(self.len(), 0);

        let mut to_init = self.start();

        for _ in 1..self.len() {
            unsafe {
                let value = value.clone();
                to_init.write(value);
                to_init = to_init.add(1);
            }
        }

        unsafe {
            to_init.write(value);
        }
    }

    /// Initializes the memory range with values generated based on their
    /// corresponding offsets.
    ///
    /// If any part of the memory range has already been initialized, the original
    /// values will leak. However, this is considered safe.
    pub(super) fn init_with<F>(&mut self, mut initializer: F)
    where
        F: FnMut(usize) -> T,
    {
        debug_assert_ne!(self.len(), 0);

        let mut to_init = self.start();

        for offset in 0..self.len() {
            unsafe {
                let value = initializer(offset);
                to_init.write(value);
                to_init = to_init.add(1);
            }
        }
    }

    /// Copies `self.len() * size_of::<T>()` bytes from `self` to `dst`. The source
    /// and destination may overlap.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `dst` must be [valid] for writes of `self.len() * size_of::<T>()` bytes,
    ///   and must remain valid even when `self` is read. (This means if the memory
    ///   ranges overlap, the `dst` pointer must not be invalidated by `src` reads.)
    /// - `dst` must be properly aligned.
    ///
    /// If `T` is not [`Copy`], the memory range of source that does not overlap
    /// with destination is considered uninitialized.
    ///
    /// [valid]: https://doc.rust-lang.org/core/ptr/index.html#safety
    pub(super) unsafe fn copy_to(self, dst: *mut T) {
        let src = self.start();
        let count = self.len();
        unsafe { src.copy_to(dst, count) }
    }

    /// Copies `self.len() * size_of::<T>()` bytes from `self` to `dst`. The source
    /// and destination may *not* overlap.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - `dst` must be [valid] for writes of `self.len() * size_of::<T>()` bytes.
    /// - `dst` must be properly aligned.
    /// - `self` must not overlap with the memory range beginning at `dst` with a
    ///   size of `self.len() * size_of::<T>()` bytes.
    ///
    /// If `T` is not [`Copy`], the memory range of source is consider uninitialized.
    pub(super) unsafe fn copy_to_nonoverlapping(self, dst: *mut T) {
        let src = self.start();
        let count = self.len();
        unsafe { src.copy_to_nonoverlapping(dst, count) }
    }

    /// Drops all values in the memory range in-place.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if the memory range is not fully initialized.
    ///
    /// See [`ptr::drop_in_place`] for more exhaustive safety concerns.
    ///
    /// [`ptr::drop_in_place`]: core::ptr::drop_in_place
    pub(super) unsafe fn drop_in_place(self) {
        unsafe { self.0.drop_in_place() }
    }
}

impl<T> IntoIterator for MemRange<T> {
    type Item = NonNull<T>;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self)
    }
}

#[derive(Debug)]
pub(super) struct IntoIter<T>(MemRange<T>);

impl<T> Iterator for IntoIter<T> {
    type Item = NonNull<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }

        let item = self.0.start();

        let start = unsafe { item.add(1) };
        let len = self.len() - 1;
        self.0 = unsafe { MemRange::new(start, len) };

        Some(unsafe { NonNull::new_unchecked(item) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }

        let start = self.0.start();
        let len = self.len() - 1;
        self.0 = unsafe { MemRange::new(start, len) };

        let item = unsafe { start.add(len) };
        Some(unsafe { NonNull::new_unchecked(item) })
    }
}

impl<T> FusedIterator for IntoIter<T> {}
