use super::Matrix;
use super::layout::{Layout, Order};
use crate::error::Result;
use crate::index::Index;
use crate::shape::AsShape;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ptr::NonNull;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Resizes the matrix to the specified shape, filling uninitialized parts with
    /// the given value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// matrix.resize((2, 2), 0)?;
    /// assert_eq!(matrix, matrix![[1, 2], [4, 5]]);
    ///
    /// matrix.resize((3, 3), 0)?;
    /// assert_eq!(matrix, matrix![[1, 2, 0], [4, 5, 0], [0, 0, 0]]);
    /// # Ok(())
    /// # }
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
        let (new_layout, new_size) = Layout::from_shape_with_size(shape)?;
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
                let tail_start = self.data.as_mut_ptr();
                let tail_len = new_size;
                MemRange::new(tail_start, tail_len).init(value);
                self.layout = new_layout;
                self.data.set_len(new_size);
                return Ok(self);
            },

            (_, 0) => unsafe {
                self.layout = new_layout;
                self.data.set_len(new_size);
                let tail_start = self.data.as_mut_ptr();
                let tail_len = old_size;
                MemRange::new(tail_start, tail_len).drop_in_place();
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
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_layout.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to(dst);
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        }
                        let tail_start_index = new_layout.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemRange::new(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => {
                        let base = self.data.as_mut_ptr();
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_layout.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to(dst);
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_layout.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base;
                            let mut dst = base;
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_layout.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                let to_drop_start = src.add(to_copy_len);
                                MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            }
                            let tail_start = base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init(value);
                        } else {
                            // Manually reallocate to avoid unnecessary memory copying.
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base;
                            let mut dst = new_base;
                            // `src` and `dst` will never overlap because
                            // they belong to different allocated objects.
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_layout.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                let to_drop_start = src.add(to_copy_len);
                                MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            }
                            let tail_start = new_base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init(value);
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
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemRange::new(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => (),

                    Ordering::Greater => {
                        let additional = new_size - old_size;
                        self.data.reserve(additional);
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = old_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = additional;
                        MemRange::new(tail_start, tail_len).init(value);
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
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_layout.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemRange::new(tail_start, tail_len).drop_in_place();
                        if new_size <= self.capacity() {
                            let mut src = tail_start;
                            let mut dst = base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                let to_init_start = dst.add(to_copy_len);
                                MemRange::new(to_init_start, to_init_len).init(value.clone());
                            }
                            let to_init_start = dst.sub(to_init_len);
                            MemRange::new(to_init_start, to_init_len).init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let new_base = new_data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                let to_init_start = dst.add(to_copy_len);
                                MemRange::new(to_init_start, to_init_len).init(value.clone());
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            let to_init_start = dst.add(to_copy_len);
                            MemRange::new(to_init_start, to_init_len).init(value);
                            self.data = new_data;
                        }
                    }

                    Ordering::Equal => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base.add(old_size);
                            let mut dst = base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                let to_init_start = dst.add(to_copy_len);
                                MemRange::new(to_init_start, to_init_len).init(value.clone());
                            }
                            let to_init_start = dst.sub(to_init_len);
                            MemRange::new(to_init_start, to_init_len).init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base.add(old_size);
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                let to_init_start = dst.add(to_copy_len);
                                MemRange::new(to_init_start, to_init_len).init(value.clone());
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            let to_init_start = dst.add(to_copy_len);
                            MemRange::new(to_init_start, to_init_len).init(value);
                            self.data = new_data;
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_layout.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let tail_start = base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init(value.clone());
                            let mut src = base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                let to_init_start = dst.add(to_copy_len);
                                MemRange::new(to_init_start, to_init_len).init(value.clone());
                            }
                            let to_init_start = dst.sub(to_init_len);
                            MemRange::new(to_init_start, to_init_len).init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let tail_start = new_base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init(value.clone());
                            let mut src = old_base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                let to_init_start = dst.add(to_copy_len);
                                MemRange::new(to_init_start, to_init_len).init(value.clone());
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            let to_init_start = dst.add(to_copy_len);
                            MemRange::new(to_init_start, to_init_len).init(value);
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
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::{Index, matrix};
    ///
    /// # fn main() -> Result<()> {
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
    /// # Ok(())
    /// # }
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
        let (new_layout, new_size) = Layout::from_shape_with_size(shape)?;
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
                let tail_start_index = 0;
                let tail_start = self.data.as_mut_ptr();
                let tail_len = new_size;
                MemRange::new(tail_start, tail_len).init_with(|offset| {
                    let index = tail_start_index + offset;
                    let index = Index::from_flattened::<O>(index, new_stride);
                    f(index)
                });
                self.layout = new_layout;
                self.data.set_len(new_size);
                return Ok(self);
            },

            (_, 0) => unsafe {
                self.layout = new_layout;
                self.data.set_len(new_size);
                let tail_start = self.data.as_mut_ptr();
                let tail_len = old_size;
                MemRange::new(tail_start, tail_len).drop_in_place();
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
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_layout.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to(dst);
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        }
                        let tail_start_index = new_layout.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemRange::new(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => {
                        let base = self.data.as_mut_ptr();
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_layout.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to(dst);
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_layout.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base;
                            let mut dst = base;
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_layout.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                let to_drop_start = src.add(to_copy_len);
                                MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            }
                            let tail_start = base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init_with(|offset| {
                                let index = tail_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base;
                            let mut dst = new_base;
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            let to_drop_start = src.add(to_copy_len);
                            MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_layout.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                let to_drop_start = src.add(to_copy_len);
                                MemRange::new(to_drop_start, to_drop_len).drop_in_place();
                            }
                            let tail_start = new_base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init_with(|offset| {
                                let index = tail_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
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
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemRange::new(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => (),

                    Ordering::Greater => {
                        let additional = new_size - old_size;
                        self.data.reserve(additional);
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = old_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = additional;
                        MemRange::new(tail_start, tail_len).init_with(|offset| {
                            let index = tail_start_index + offset;
                            let index = Index::from_flattened::<O>(index, new_stride);
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
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_layout.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemRange::new(tail_start, tail_len).drop_in_place();
                        let mut to_init_start_index = new_size + to_copy_len;
                        if new_size <= self.capacity() {
                            let mut src = tail_start;
                            let mut dst = base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = base.add(to_init_start_index);
                                MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                    let index = to_init_start_index + offset;
                                    let index = Index::from_flattened::<O>(index, new_stride);
                                    f(index)
                                });
                            }
                            to_init_start_index -= new_stride.major();
                            let to_init_start = base.add(to_init_start_index);
                            MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                let index = to_init_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let new_base = new_data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = new_base.add(to_init_start_index);
                                MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                    let index = to_init_start_index + offset;
                                    let index = Index::from_flattened::<O>(index, new_stride);
                                    f(index)
                                });
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            to_init_start_index -= new_stride.major();
                            let to_init_start = new_base.add(to_init_start_index);
                            MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                let index = to_init_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                            self.data = new_data;
                        }
                    }

                    Ordering::Equal => {
                        let mut to_init_start_index = new_size + to_copy_len;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base.add(old_size);
                            let mut dst = base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = base.add(to_init_start_index);
                                MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                    let index = to_init_start_index + offset;
                                    let index = Index::from_flattened::<O>(index, new_stride);
                                    f(index)
                                });
                            }
                            to_init_start_index -= new_stride.major();
                            let to_init_start = base.add(to_init_start_index);
                            MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                let index = to_init_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base.add(old_size);
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = new_base.add(to_init_start_index);
                                MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                    let index = to_init_start_index + offset;
                                    let index = Index::from_flattened::<O>(index, new_stride);
                                    f(index)
                                });
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            to_init_start_index -= new_stride.major();
                            let to_init_start = new_base.add(to_init_start_index);
                            MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                let index = to_init_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                            self.data = new_data;
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_layout.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        let mut to_init_start_index = tail_start_index + to_copy_len;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let tail_start = base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init_with(|offset| {
                                let index = tail_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                            let mut src = base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to(dst);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = base.add(to_init_start_index);
                                MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                    let index = to_init_start_index + offset;
                                    let index = Index::from_flattened::<O>(index, new_stride);
                                    f(index)
                                });
                            }
                            to_init_start_index -= new_stride.major();
                            let to_init_start = base.add(to_init_start_index);
                            MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                let index = to_init_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let tail_start = new_base.add(tail_start_index);
                            MemRange::new(tail_start, tail_len).init_with(|offset| {
                                let index = tail_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
                                f(index)
                            });
                            let mut src = old_base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_layout.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = new_base.add(to_init_start_index);
                                MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                    let index = to_init_start_index + offset;
                                    let index = Index::from_flattened::<O>(index, new_stride);
                                    f(index)
                                });
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            MemRange::new(src, to_copy_len).copy_to_nonoverlapping(dst);
                            to_init_start_index -= new_stride.major();
                            let to_init_start = new_base.add(to_init_start_index);
                            MemRange::new(to_init_start, to_init_len).init_with(|offset| {
                                let index = to_init_start_index + offset;
                                let index = Index::from_flattened::<O>(index, new_stride);
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
struct MemRange<T> {
    range: NonNull<[T]>,
    marker: PhantomData<*mut T>,
}

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
    const unsafe fn new(start: *mut T, len: usize) -> Self {
        let start = unsafe { NonNull::new_unchecked(start) };
        let range = NonNull::slice_from_raw_parts(start, len);
        let marker = PhantomData;
        Self { range, marker }
    }

    /// Returns a pointer to the start of the memory range.
    ///
    /// Note that if the length of the memory range is `0`, the returned pointer is
    /// still inside the provenance of the allocated object, but may have `0` bytes
    /// it can read/write.
    const fn start(&self) -> *mut T {
        self.range.as_ptr() as *mut T
    }

    /// Returns the length of the memory range.
    const fn len(&self) -> usize {
        self.range.len()
    }

    /// Initializes the memory range with the given value.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if the length of the memory range is `0`.
    ///
    /// If any part of the memory range has already been initialized, the original
    /// values will leak. However, this is considered safe.
    unsafe fn init(&mut self, value: T)
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
    fn init_with<F>(&mut self, mut f: F)
    where
        F: FnMut(usize) -> T,
    {
        let mut to_init = self.start();

        for offset in 0..self.len() {
            unsafe {
                let value = f(offset);
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
    const unsafe fn copy_to(self, dst: *mut T) {
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
    const unsafe fn copy_to_nonoverlapping(self, dst: *mut T) {
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
    unsafe fn drop_in_place(self) {
        unsafe { self.range.drop_in_place() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_unary;
    use crate::error::Error;
    use crate::mock::{MockZeroSized, Scope};
    use crate::shape::Shape;

    #[test]
    fn test_resize() {
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
                        pairs.push((old_shape, new_shape))
                    }
                }
            }
        }

        dispatch_unary! {{
            for &(old_shape, new_shape) in &pairs {
                let mut matrix =
                    Matrix::<_, O>::from_value(old_shape, MockZeroSized::new()).unwrap();
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
                let expected =
                    Matrix::<_, RowMajor>::from_value(new_shape, MockZeroSized::new()).unwrap();
                assert_eq!(matrix, expected);

                let old_size = old_shape.size().unwrap();
                let new_size = new_shape.size().unwrap();
                if new_size <= old_size {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index).unwrap();
                    matrix.resize(new_shape, Index::default()).unwrap();
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| {
                        if index.row < old_shape.nrows() && index.col < old_shape.ncols() {
                            index
                        } else {
                            Index::default()
                        }
                    })
                    .unwrap();
                    assert_eq!(matrix, expected);
                } else {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index).unwrap();
                    // Ensure the in-place path is taken.
                    matrix.data.reserve(new_size - old_size);
                    assert!(new_size <= matrix.capacity());
                    matrix.resize(new_shape, Index::default()).unwrap();
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| {
                        if index.row < old_shape.nrows() && index.col < old_shape.ncols() {
                            index
                        } else {
                            Index::default()
                        }
                    })
                    .unwrap();
                    assert_eq!(matrix, expected);

                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index).unwrap();
                    // Ensure the reallocation path is taken.
                    matrix.data.shrink_to_fit();
                    assert!(new_size > matrix.capacity());
                    matrix.resize(new_shape, Index::default()).unwrap();
                    let expected = Matrix::<_, RowMajor>::from_fn(new_shape, |index| {
                        if index.row < old_shape.nrows() && index.col < old_shape.ncols() {
                            index
                        } else {
                            Index::default()
                        }
                    })
                    .unwrap();
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
    }

    #[test]
    fn test_resize_with() {
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
                        pairs.push((old_shape, new_shape))
                    }
                }
            }
        }

        dispatch_unary! {{
            for &(old_shape, new_shape) in &pairs {
                let mut matrix =
                    Matrix::<_, O>::from_value(old_shape, MockZeroSized::new()).unwrap();
                Scope::with(|scope| {
                    matrix
                        .resize_with(new_shape, |_| MockZeroSized::new())
                        .unwrap();
                    let expected_count = Count::expected(old_shape, new_shape);
                    assert_eq!(scope.init_count(), expected_count.init);
                    assert_eq!(scope.drop_count(), expected_count.drop);
                });
                let expected =
                    Matrix::<_, RowMajor>::from_value(new_shape, MockZeroSized::new()).unwrap();
                assert_eq!(matrix, expected);

                let old_size = old_shape.size().unwrap();
                let new_size = new_shape.size().unwrap();
                if new_size <= old_size {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index).unwrap();
                    matrix.resize_with(new_shape, |index| index).unwrap();
                    let expected =
                        Matrix::<_, RowMajor>::from_fn(new_shape, |index| index).unwrap();
                    assert_eq!(matrix, expected);
                } else {
                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index).unwrap();
                    // Ensure the in-place path is taken.
                    matrix.data.reserve(new_size - old_size);
                    assert!(new_size <= matrix.capacity());
                    matrix.resize_with(new_shape, |index| index).unwrap();
                    let expected =
                        Matrix::<_, RowMajor>::from_fn(new_shape, |index| index).unwrap();
                    assert_eq!(matrix, expected);

                    let mut matrix = Matrix::<_, O>::from_fn(old_shape, |index| index).unwrap();
                    // Ensure the reallocation path is taken.
                    matrix.data.shrink_to_fit();
                    assert!(new_size > matrix.capacity());
                    matrix.resize_with(new_shape, |index| index).unwrap();
                    let expected =
                        Matrix::<_, RowMajor>::from_fn(new_shape, |index| index).unwrap();
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
