use super::Matrix;
use super::layout::Layout;
use super::order::Order;
use crate::error::Result;
use crate::index::Index;
use crate::shape::AsShape;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::mem::MaybeUninit;
use core::ptr;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Resizes the matrix to the specified shape, filling uninitialized parts with
    /// the given value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size of the shape exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// matrix.resize((2, 2), 0)?;
    /// assert_eq!(matrix, matrix![[1, 2], [4, 5]]);
    ///
    /// matrix.resize((3, 3), 0)?;
    /// assert_eq!(matrix, matrix![[1, 2, 0], [4, 5, 0], [0, 0, 0]]);
    /// #
    /// # Ok::<(), matreex::Error>(())
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn resize<S>(&mut self, shape: S, value: T) -> Result<&mut Self>
    where
        S: AsShape,
        T: Clone,
    {
        // # Exception Safety
        //
        // At any given point of execution, the invariant that `self.layout.size()`
        // is equal to `self.data.len()` must be upheld, and the memory range within
        // `self.layout` must be valid, or minimal exception safety will be violated.
        //
        // For this reason, `Vec::resize` is avoided, as it uses a drop guard that
        // increments the length of `self.data` after each successful write, which
        // could violate the invariant if `T::clone` or `T::drop` panic.
        //
        // # Terminology
        //
        // In the following code, some variables are prefixed with `tail`. A tail is
        // the trailing part of the underlying data that needs to be initialized or
        // dropped in the context of resizing. In particular:
        //
        // - If `new_layout.major() < old_layout.major()`, the tail refers to the range
        //   from `new_layout.major() * old_stride.major()` to `old_size` and needs to
        //   be dropped.
        // - If `new_layout.major() == old_layout.major()`, the tail does not exist.
        // - If `new_layout.major() > old_layout.major()`, the tail refers to the range
        //   from `old_layout.major() * new_stride.major()` to `new_size` and needs to
        //   be initialized.
        //
        // Note that the tail may overlap with other parts of memory, so the execution
        // order matters.

        let (old_layout, old_size) = (self.layout, self.size());
        let (new_layout, new_size) = Layout::from_shape(shape)?;
        let old_stride = old_layout.stride();
        let new_stride = new_layout.stride();
        let minor_stride = old_stride.minor();

        match (old_size, new_size) {
            (0, 0) => {
                self.layout = new_layout;
                return Ok(self);
            }

            (0, _) => unsafe {
                self.data.reserve(new_size);
                self.data
                    .spare_capacity_mut()
                    .get_unchecked_mut(..new_size)
                    .init(value);
                self.layout = new_layout;
                self.data.set_len(new_size);
                return Ok(self);
            },

            (_, 0) => unsafe {
                self.layout = new_layout;
                self.data.set_len(new_size);
                self.data
                    .spare_capacity_mut()
                    .get_unchecked_mut(..old_size)
                    .assume_init_drop();
                return Ok(self);
            },

            (_, _) => (),
        }

        // After early returns, it is guaranteed that:
        //
        // - All axis lengths and strides are non-zero.
        // - All loop peel iterations for major axis vectors are valid.

        let major_len_cmp = new_layout.major().cmp(&old_layout.major());
        let minor_len_cmp = new_layout.minor().cmp(&old_layout.minor());

        match minor_len_cmp {
            Ordering::Less => unsafe {
                self.layout = Layout::default();
                self.data.set_len(0);

                let to_copy_len = new_layout.minor() * minor_stride;
                let to_drop_len = (old_layout.minor() - new_layout.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let base = self.data.as_mut_ptr();
                        let mut src = 0;
                        let mut dst = 0;
                        let to_drop_start = to_copy_len;
                        let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(to_drop)
                            .assume_init_drop();
                        for _ in 1..new_layout.major() {
                            src += old_stride.major();
                            dst += new_stride.major();
                            ptr::copy(base.add(src), base.add(dst), to_copy_len);
                            let to_drop_start = src + to_copy_len;
                            let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                        }
                        let tail_start = new_layout.major() * old_stride.major();
                        let tail = tail_start..old_size;
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(tail)
                            .assume_init_drop();
                    }

                    Ordering::Equal => {
                        let base = self.data.as_mut_ptr();
                        let mut src = 0;
                        let mut dst = 0;
                        let to_drop_start = to_copy_len;
                        let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(to_drop)
                            .assume_init_drop();
                        for _ in 1..new_layout.major() {
                            src += old_stride.major();
                            dst += new_stride.major();
                            ptr::copy(base.add(src), base.add(dst), to_copy_len);
                            let to_drop_start = src + to_copy_len;
                            let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                        }
                    }

                    Ordering::Greater => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = 0;
                            let mut dst = 0;
                            let to_drop_start = to_copy_len;
                            let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                            for _ in 1..old_layout.major() {
                                src += old_stride.major();
                                dst += new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_drop_start = src + to_copy_len;
                                let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_drop)
                                    .assume_init_drop();
                            }
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = 0;
                            let mut dst = 0;
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_drop_start = to_copy_len;
                            let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                            for _ in 1..old_layout.major() {
                                src += old_stride.major();
                                dst += new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_drop_start = src + to_copy_len;
                                let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_drop)
                                    .assume_init_drop();
                            }
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init(value);
                            self.data = new_data;
                        }
                    }
                }

                self.layout = new_layout;
                self.data.set_len(new_size);
            },

            Ordering::Equal => unsafe {
                match major_len_cmp {
                    Ordering::Less => {
                        self.layout = new_layout;
                        self.data.set_len(new_size);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(0..(old_size - new_size))
                            .assume_init_drop();
                    }

                    Ordering::Equal => (),

                    Ordering::Greater => {
                        let additional = new_size - old_size;
                        self.data.reserve(additional);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(0..additional)
                            .init(value);
                        self.layout = new_layout;
                        self.data.set_len(new_size);
                    }
                }
            },

            Ordering::Greater => unsafe {
                self.layout = Layout::default();
                self.data.set_len(0);

                let to_copy_len = old_layout.minor() * minor_stride;
                let to_init_len = (new_layout.minor() - old_layout.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let tail_start = new_layout.major() * old_stride.major();
                        let tail = tail_start..old_size;
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(tail)
                            .assume_init_drop();
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init(value.clone());
                            }
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                new_data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init(value.clone());
                            }
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init(value);
                            self.data = new_data;
                        }
                    }

                    Ordering::Equal => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = old_size;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init(value.clone());
                            }
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_size;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                new_data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init(value.clone());
                            }
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init(value);
                            self.data = new_data;
                        }
                    }

                    Ordering::Greater => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init(value.clone());
                            let mut src = old_size;
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init(value.clone());
                            }
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init(value.clone());
                            let mut src = old_size;
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                new_data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init(value.clone());
                            }
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init(value);
                            self.data = new_data;
                        }
                    }
                }

                self.layout = new_layout;
                self.data.set_len(new_size);
            },
        }

        Ok(self)
    }

    /// Resizes the matrix to the specified shape, filling uninitialized parts with
    /// values generated based on their corresponding indices.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size of the shape exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Index, matrix};
    ///
    /// let mut matrix = matrix![
    ///     [Index::new(0, 0), Index::new(0, 1), Index::new(0, 2)],
    ///     [Index::new(1, 0), Index::new(1, 1), Index::new(1, 2)],
    /// ];
    ///
    /// matrix.resize_with((2, 2), |index| index)?;
    /// assert_eq!(
    ///     matrix,
    ///     matrix![
    ///         [Index::new(0, 0), Index::new(0, 1)],
    ///         [Index::new(1, 0), Index::new(1, 1)],
    ///     ]
    /// );
    ///
    /// matrix.resize_with((3, 3), |index| index)?;
    /// assert_eq!(
    ///     matrix,
    ///     matrix![
    ///         [Index::new(0, 0), Index::new(0, 1), Index::new(0, 2)],
    ///         [Index::new(1, 0), Index::new(1, 1), Index::new(1, 2)],
    ///         [Index::new(2, 0), Index::new(2, 1), Index::new(2, 2)],
    ///     ]
    /// );
    /// #
    /// # Ok::<(), matreex::Error>(())
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn resize_with<S, F>(&mut self, shape: S, mut f: F) -> Result<&mut Self>
    where
        S: AsShape,
        F: FnMut(Index) -> T,
    {
        // See `Matrix::resize` for details.

        let (old_layout, old_size) = (self.layout, self.size());
        let (new_layout, new_size) = Layout::from_shape(shape)?;
        let old_stride = old_layout.stride();
        let new_stride = new_layout.stride();
        let minor_stride = old_stride.minor();

        match (old_size, new_size) {
            (0, 0) => {
                self.layout = new_layout;
                return Ok(self);
            }

            (0, _) => unsafe {
                self.data.reserve(new_size);
                self.data
                    .spare_capacity_mut()
                    .get_unchecked_mut(..new_size)
                    .init_with(|index| {
                        let index = Index::from_linear::<O>(index, new_stride);
                        f(index)
                    });
                self.layout = new_layout;
                self.data.set_len(new_size);
                return Ok(self);
            },

            (_, 0) => unsafe {
                self.layout = new_layout;
                self.data.set_len(new_size);
                self.data
                    .spare_capacity_mut()
                    .get_unchecked_mut(..old_size)
                    .assume_init_drop();
                return Ok(self);
            },

            (_, _) => (),
        }

        let major_len_cmp = new_layout.major().cmp(&old_layout.major());
        let minor_len_cmp = new_layout.minor().cmp(&old_layout.minor());

        match minor_len_cmp {
            Ordering::Less => unsafe {
                self.layout = Layout::default();
                self.data.set_len(0);

                let to_copy_len = new_layout.minor() * minor_stride;
                let to_drop_len = (old_layout.minor() - new_layout.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let base = self.data.as_mut_ptr();
                        let mut src = 0;
                        let mut dst = 0;
                        let to_drop_start = to_copy_len;
                        let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(to_drop)
                            .assume_init_drop();
                        for _ in 1..new_layout.major() {
                            src += old_stride.major();
                            dst += new_stride.major();
                            ptr::copy(base.add(src), base.add(dst), to_copy_len);
                            let to_drop_start = src + to_copy_len;
                            let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                        }
                        let tail_start = new_layout.major() * old_stride.major();
                        let tail = tail_start..old_size;
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(tail)
                            .assume_init_drop();
                    }

                    Ordering::Equal => {
                        let base = self.data.as_mut_ptr();
                        let mut src = 0;
                        let mut dst = 0;
                        let to_drop_start = to_copy_len;
                        let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(to_drop)
                            .assume_init_drop();
                        for _ in 1..new_layout.major() {
                            src += old_stride.major();
                            dst += new_stride.major();
                            ptr::copy(base.add(src), base.add(dst), to_copy_len);
                            let to_drop_start = src + to_copy_len;
                            let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                        }
                    }

                    Ordering::Greater => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = 0;
                            let mut dst = 0;
                            let to_drop_start = to_copy_len;
                            let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                            for _ in 1..old_layout.major() {
                                src += old_stride.major();
                                dst += new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_drop_start = src + to_copy_len;
                                let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_drop)
                                    .assume_init_drop();
                            }
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init_with(|offset| {
                                    let index = tail_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = 0;
                            let mut dst = 0;
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_drop_start = to_copy_len;
                            let to_drop = to_drop_start..(old_layout.minor() * minor_stride);
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_drop)
                                .assume_init_drop();
                            for _ in 1..old_layout.major() {
                                src += old_stride.major();
                                dst += new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_drop_start = src + to_copy_len;
                                let to_drop = to_drop_start..(to_drop_start + to_drop_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_drop)
                                    .assume_init_drop();
                            }
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init_with(|offset| {
                                    let index = tail_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                            self.data = new_data;
                        }
                    }
                }

                self.layout = new_layout;
                self.data.set_len(new_size);
            },

            Ordering::Equal => unsafe {
                match major_len_cmp {
                    Ordering::Less => {
                        self.layout = new_layout;
                        self.data.set_len(new_size);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(0..(old_size - new_size))
                            .assume_init_drop();
                    }

                    Ordering::Equal => (),

                    Ordering::Greater => {
                        let additional = new_size - old_size;
                        self.data.reserve(additional);
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(0..additional)
                            .init_with(|offset| {
                                let index = old_size + offset;
                                let index = Index::from_linear::<O>(index, new_stride);
                                f(index)
                            });
                        self.layout = new_layout;
                        self.data.set_len(new_size);
                    }
                }
            },

            Ordering::Greater => unsafe {
                self.layout = Layout::default();
                self.data.set_len(0);

                let to_copy_len = old_layout.minor() * minor_stride;
                let to_init_len = (new_layout.minor() - old_layout.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let tail_start = new_layout.major() * old_stride.major();
                        let tail = tail_start..old_size;
                        self.data
                            .spare_capacity_mut()
                            .get_unchecked_mut(tail)
                            .assume_init_drop();
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init_with(|offset| {
                                        let index = to_init_start + offset;
                                        let index = Index::from_linear::<O>(index, new_stride);
                                        f(index)
                                    });
                            }
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init_with(|offset| {
                                    let index = to_init_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                new_data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init_with(|offset| {
                                        let index = to_init_start + offset;
                                        let index = Index::from_linear::<O>(index, new_stride);
                                        f(index)
                                    });
                            }
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init_with(|offset| {
                                    let index = to_init_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                            self.data = new_data;
                        }
                    }

                    Ordering::Equal => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = old_size;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init_with(|offset| {
                                        let index = to_init_start + offset;
                                        let index = Index::from_linear::<O>(index, new_stride);
                                        f(index)
                                    });
                            }
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init_with(|offset| {
                                    let index = to_init_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_size;
                            let mut dst = new_size;
                            for _ in 1..new_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                new_data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init_with(|offset| {
                                        let index = to_init_start + offset;
                                        let index = Index::from_linear::<O>(index, new_stride);
                                        f(index)
                                    });
                            }
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init_with(|offset| {
                                    let index = to_init_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                            self.data = new_data;
                        }
                    }

                    Ordering::Greater => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init_with(|offset| {
                                    let index = tail_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                            let mut src = old_size;
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy(base.add(src), base.add(dst), to_copy_len);
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                self.data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init_with(|offset| {
                                        let index = to_init_start + offset;
                                        let index = Index::from_linear::<O>(index, new_stride);
                                        f(index)
                                    });
                            }
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            self.data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init_with(|offset| {
                                    let index = to_init_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let tail_start = old_layout.major() * new_stride.major();
                            let tail = tail_start..new_size;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(tail)
                                .init_with(|offset| {
                                    let index = tail_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                            let mut src = old_size;
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src -= old_stride.major();
                                dst -= new_stride.major();
                                ptr::copy_nonoverlapping(
                                    old_base.add(src),
                                    new_base.add(dst),
                                    to_copy_len,
                                );
                                let to_init_start = dst + to_copy_len;
                                let to_init = to_init_start..(to_init_start + to_init_len);
                                new_data
                                    .spare_capacity_mut()
                                    .get_unchecked_mut(to_init)
                                    .init_with(|offset| {
                                        let index = to_init_start + offset;
                                        let index = Index::from_linear::<O>(index, new_stride);
                                        f(index)
                                    });
                            }
                            ptr::copy_nonoverlapping(old_base, new_base, to_copy_len);
                            let to_init_start = dst - to_init_len;
                            let to_init = to_init_start..dst;
                            new_data
                                .spare_capacity_mut()
                                .get_unchecked_mut(to_init)
                                .init_with(|offset| {
                                    let index = to_init_start + offset;
                                    let index = Index::from_linear::<O>(index, new_stride);
                                    f(index)
                                });
                            self.data = new_data;
                        }
                    }
                }

                self.layout = new_layout;
                self.data.set_len(new_size);
            },
        }

        Ok(self)
    }
}

trait MaybeUninitSliceExt {
    type Item;

    unsafe fn init(&mut self, value: Self::Item)
    where
        Self::Item: Clone;

    fn init_with<F>(&mut self, f: F)
    where
        F: FnMut(usize) -> Self::Item;
}

impl<T> MaybeUninitSliceExt for [MaybeUninit<T>] {
    type Item = T;

    /// Initializes the slice with the given value.
    ///
    /// If any part of the slice has already been initialized, the original values
    /// will leak. However, this is considered safe.
    ///
    /// # Safety
    ///
    /// This caller must ensure that `self.len() > 0`.
    unsafe fn init(&mut self, value: Self::Item)
    where
        Self::Item: Clone,
    {
        debug_assert_ne!(self.len(), 0);

        let last = self.len() - 1;

        for index in 0..last {
            let value = value.clone();
            unsafe {
                self.get_unchecked_mut(index).write(value);
            }
        }

        unsafe {
            self.get_unchecked_mut(last).write(value);
        }
    }

    /// Initializes the slice with values generated based on their corresponding
    /// offsets.
    ///
    /// If any part of the slice has already been initialized, the original values
    /// will leak. However, this is considered safe.
    fn init_with<F>(&mut self, mut f: F)
    where
        F: FnMut(usize) -> Self::Item,
    {
        for index in 0..self.len() {
            let value = f(index);
            unsafe {
                self.get_unchecked_mut(index).write(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_unary;
    use crate::error::Error;
    use crate::shape::Shape;
    use crate::testkit::{MockZeroSized, Scope};

    #[test]
    fn test_resize() -> Result<()> {
        #[cfg(miri)]
        let lens = [0, 1, 2, 3, 5];
        #[cfg(not(miri))]
        let lens = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19];

        let mut pairs = Vec::with_capacity(lens.len().pow(4));
        for old_nrows in lens {
            for old_ncols in lens {
                for new_nrows in lens {
                    for new_ncols in lens {
                        let old_shape = Shape::new(old_nrows, old_ncols);
                        let new_shape = Shape::new(new_nrows, new_ncols);
                        pairs.push((old_shape, new_shape));
                    }
                }
            }
        }

        dispatch_unary! {{
            for &(old_shape, new_shape) in &pairs {
                let mut matrix = Matrix::<_, O>::from_value(old_shape, MockZeroSized::new())?;
                Scope::with(|scope| {
                    matrix.resize(new_shape, MockZeroSized::new()).unwrap();
                    let expected_count = Count::expected(old_shape, new_shape);
                    if expected_count.init == 0 {
                        // Argument `value` passed to `Matrix::resize`
                        // results in an extra init and drop.
                        assert_eq!(scope.init_count(), 1);
                        assert_eq!(scope.drop_count(), expected_count.drop + 1);
                    } else {
                        assert_eq!(scope.init_count(), expected_count.init);
                        assert_eq!(scope.drop_count(), expected_count.drop);
                    }
                });
                let expected = Matrix::<_, RowMajor>::from_value(new_shape, MockZeroSized::new())?;
                assert_eq!(matrix, expected);

                let old_size = old_shape.size()?;
                let new_size = new_shape.size()?;
                if new_size <= old_size {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index)?;
                    matrix.resize(new_shape, Index::default())?;
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| {
                        if index.row < old_shape.nrows() && index.col < old_shape.ncols() {
                            index
                        } else {
                            Index::default()
                        }
                    })?;
                    assert_eq!(matrix, expected);
                } else {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index)?;
                    // Ensure the in-place path is taken.
                    matrix.data.reserve(new_size - old_size);
                    assert!(new_size <= matrix.capacity());
                    matrix.resize(new_shape, Index::default())?;
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| {
                        if index.row < old_shape.nrows() && index.col < old_shape.ncols() {
                            index
                        } else {
                            Index::default()
                        }
                    })?;
                    assert_eq!(matrix, expected);

                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index)?;
                    // Ensure the reallocation path is taken.
                    matrix.data.shrink_to_fit();
                    assert!(new_size > matrix.capacity());
                    matrix.resize(new_shape, Index::default())?;
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| {
                        if index.row < old_shape.nrows() && index.col < old_shape.ncols() {
                            index
                        } else {
                            Index::default()
                        }
                    })?;
                    assert_eq!(matrix, expected);
                }
            }

            let new_shape = Shape::new(usize::MAX, 2);
            let mut matrix = Matrix::<i32, O>::new();
            let unchanged = matrix.clone();
            let error = matrix.resize(new_shape, 0).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
            assert_eq!(matrix, unchanged);

            let new_shape = Shape::new(usize::MAX, 1);
            let mut matrix = Matrix::<i32, O>::new();
            let unchanged = matrix.clone();
            let error = matrix.resize(new_shape, 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
            assert_eq!(matrix, unchanged);

            // Unable to cover.
            // let new_shape = Shape::new(usize::MAX, 1);
            // let mut matrix = Matrix::<(), O>::new();
            // let unchanged = matrix.clone();
            // assert!(matrix.resize(new_shape, ()).is_ok());
            // assert_eq!(matrix, unchanged);
        }}

        Ok(())
    }

    #[test]
    fn test_resize_with() -> Result<()> {
        #[cfg(miri)]
        let lens = [0, 1, 2, 3, 5];
        #[cfg(not(miri))]
        let lens = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19];

        let mut pairs = Vec::with_capacity(lens.len().pow(4));
        for old_nrows in lens {
            for old_ncols in lens {
                for new_nrows in lens {
                    for new_ncols in lens {
                        let old_shape = Shape::new(old_nrows, old_ncols);
                        let new_shape = Shape::new(new_nrows, new_ncols);
                        pairs.push((old_shape, new_shape));
                    }
                }
            }
        }

        dispatch_unary! {{
            for &(old_shape, new_shape) in &pairs {
                let mut matrix = Matrix::<_, O>::from_value(old_shape, MockZeroSized::new())?;
                Scope::with(|scope| {
                    matrix
                        .resize_with(new_shape, |_| MockZeroSized::new())
                        .unwrap();
                    let expected_count = Count::expected(old_shape, new_shape);
                    assert_eq!(scope.init_count(), expected_count.init);
                    assert_eq!(scope.drop_count(), expected_count.drop);
                });
                let expected = Matrix::<_, RowMajor>::from_value(new_shape, MockZeroSized::new())?;
                assert_eq!(matrix, expected);

                let old_size = old_shape.size()?;
                let new_size = new_shape.size()?;
                if new_size <= old_size {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index)?;
                    matrix.resize_with(new_shape, |index| index)?;
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| index)?;
                    assert_eq!(matrix, expected);
                } else {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index)?;
                    // Ensure the in-place path is taken.
                    matrix.data.reserve(new_size - old_size);
                    assert!(new_size <= matrix.capacity());
                    matrix.resize_with(new_shape, |index| index)?;
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| index)?;
                    assert_eq!(matrix, expected);

                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index)?;
                    // Ensure the reallocation path is taken.
                    matrix.data.shrink_to_fit();
                    assert!(new_size > matrix.capacity());
                    matrix.resize_with(new_shape, |index| index)?;
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| index)?;
                    assert_eq!(matrix, expected);
                }
            }

            let new_shape = Shape::new(usize::MAX, 2);
            let mut matrix = Matrix::<i32, O>::new();
            let unchanged = matrix.clone();
            let error = matrix.resize_with(new_shape, |_| 0).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
            assert_eq!(matrix, unchanged);

            let new_shape = Shape::new(usize::MAX, 1);
            let mut matrix = Matrix::<i32, O>::new();
            let unchanged = matrix.clone();
            let error = matrix.resize_with(new_shape, |_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
            assert_eq!(matrix, unchanged);

            // Unable to cover.
            // let new_shape = Shape::new(usize::MAX, 1);
            // let mut matrix = Matrix::<(), O>::new();
            // let unchanged = matrix.clone();
            // assert!(matrix.resize_with(new_shape, |_| ()).is_ok());
            // assert_eq!(matrix, unchanged);
        }}

        Ok(())
    }

    #[derive(Debug)]
    struct Count {
        init: usize,
        drop: usize,
    }

    impl Count {
        fn expected(old_shape: Shape, new_shape: Shape) -> Self {
            let old_nrows = old_shape.nrows();
            let old_ncols = old_shape.ncols();
            let new_nrows = new_shape.nrows();
            let new_ncols = new_shape.ncols();
            let init;
            let drop;
            match (new_nrows.cmp(&old_nrows), new_ncols.cmp(&old_ncols)) {
                (Ordering::Less, Ordering::Less) => {
                    init = 0;
                    drop = old_nrows * old_ncols - new_nrows * new_ncols;
                }
                (Ordering::Less, Ordering::Equal) => {
                    init = 0;
                    drop = (old_nrows - new_nrows) * old_ncols;
                }
                (Ordering::Less, Ordering::Greater) => {
                    init = (new_ncols - old_ncols) * new_nrows;
                    drop = (old_nrows - new_nrows) * old_ncols;
                }
                (Ordering::Equal, Ordering::Less) => {
                    init = 0;
                    drop = (old_ncols - new_ncols) * old_nrows;
                }
                (Ordering::Equal, Ordering::Equal) => {
                    init = 0;
                    drop = 0;
                }
                (Ordering::Equal, Ordering::Greater) => {
                    init = (new_ncols - old_ncols) * old_nrows;
                    drop = 0;
                }
                (Ordering::Greater, Ordering::Less) => {
                    init = (new_nrows - old_nrows) * new_ncols;
                    drop = (old_ncols - new_ncols) * old_nrows;
                }
                (Ordering::Greater, Ordering::Equal) => {
                    init = (new_nrows - old_nrows) * new_ncols;
                    drop = 0;
                }
                (Ordering::Greater, Ordering::Greater) => {
                    init = new_nrows * new_ncols - old_nrows * old_ncols;
                    drop = 0;
                }
            }
            Self { init, drop }
        }
    }
}
