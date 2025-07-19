use crate::Matrix;
use crate::error::Result;
use crate::index::Index;
use crate::order::Order;
use crate::shape::{AsShape, AxisShape, Stride};
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZero;
use core::ptr;

impl<T> Matrix<T> {
    /// Resizes the matrix to the specified shape, filling uninitialized parts
    /// with the given value.
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
    /// use matreex::{Order, matrix};
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
        // At any given point of execution, the invariant that the size
        // of `self.shape` is equal to the length of `self.data` must be
        // upheld, and the memory within `self.shape` must be valid, or
        // minimal exception safety will be violated.
        //
        // For this reason, `Vec::resize` is avoided, as it uses a drop
        // guard that increments `self.data.len` after each successful
        // write, which could violate the invariant if `T::clone` or
        // `T::drop` panic.
        //
        // # Terminology
        //
        // In the following code, some variables are prefixed with `tail`.
        // A tail is the trailing part of the underlying data that needs
        // to be dropped or initialized in the context of resizing. In
        // particular:
        //
        // - If `new_shape.major() < old_shape.major()`, the tail starts
        //   from `new_shape.major() * old_stride.major()` to `old_size`,
        //   which needs to be dropped.
        // - If `new_shape.major() == old_shape.major()`, the tail does
        //   not exist.
        // - If `new_shape.major() > old_shape.major()`, the tail starts
        //   from `old_shape.major() * new_stride.major()` to `new_size`,
        //   which needs to be initialized.
        //
        // Note that the tail may overlap with other parts of memory, so
        // the execution order matters.

        let new_shape = AxisShape::from_shape(shape, self.order);
        let new_size = new_shape.size::<T>()?;
        let old_size = self.size();

        // After these early returns, it is guaranteed that:
        //
        // - All axis lengths and strides are non-zero.
        // - All loop peel iterations for major axis vectors are valid.
        match (old_size, new_size) {
            (0, 0) => {
                self.shape = new_shape;
                return Ok(self);
            }

            (0, _) => unsafe {
                self.data.reserve(new_size);
                let tail_start = self.data.as_mut_ptr();
                let tail_len = new_size;
                MemoryRange::new_unchecked(tail_start, tail_len).init(value);
                self.shape = new_shape;
                self.data.set_len(new_size);
                return Ok(self);
            },

            (_, 0) => unsafe {
                self.shape = new_shape;
                self.data.set_len(new_size);
                let tail_start = self.data.as_mut_ptr();
                let tail_len = old_size;
                MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                return Ok(self);
            },

            (_, _) => (),
        }

        let old_shape = self.shape;
        let old_stride = old_shape.stride();
        let new_stride = new_shape.stride();
        let minor_stride = old_stride.minor();

        let major_len_cmp = new_shape.major().cmp(&old_shape.major());
        let minor_len_cmp = new_shape.minor().cmp(&old_shape.minor());

        match minor_len_cmp {
            Ordering::Less => unsafe {
                self.shape = AxisShape::default();
                self.data.set_len(0);

                let to_copy_len = new_shape.minor() * minor_stride;
                let to_drop_len = (old_shape.minor() - new_shape.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let base = self.data.as_mut_ptr();
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_shape.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            ptr::copy(src, dst, to_copy_len);
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        }
                        let tail_start_index = new_shape.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => {
                        let base = self.data.as_mut_ptr();
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_shape.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            ptr::copy(src, dst, to_copy_len);
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_shape.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base;
                            let mut dst = base;
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_shape.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                let to_drop_start = src.add(to_copy_len);
                                MemoryRange::new_unchecked(to_drop_start, to_drop_len)
                                    .drop_in_place();
                            }
                            let tail_start = base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init(value);
                        } else {
                            // Manually reallocate to avoid unnecessary memory copying.
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base;
                            let mut dst = new_base;
                            // `src` and `dst` will never overlap because
                            // they belong to different allocated objects.
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_shape.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                let to_drop_start = src.add(to_copy_len);
                                MemoryRange::new_unchecked(to_drop_start, to_drop_len)
                                    .drop_in_place();
                            }
                            let tail_start = new_base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init(value);
                            self.data = new_data;
                        }
                    }
                }

                self.shape = new_shape;
                self.data.set_len(new_size);
            },

            Ordering::Equal => unsafe {
                match major_len_cmp {
                    Ordering::Less => {
                        self.shape = new_shape;
                        self.data.set_len(new_size);
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => (),

                    Ordering::Greater => {
                        let additional = new_size - old_size;
                        self.data.reserve(additional);
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = old_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = additional;
                        MemoryRange::new_unchecked(tail_start, tail_len).init(value);
                        self.shape = new_shape;
                        self.data.set_len(new_size);
                    }
                }
            },

            Ordering::Greater => unsafe {
                self.shape = AxisShape::default();
                self.data.set_len(0);

                let to_copy_len = old_shape.minor() * minor_stride;
                let to_init_len = (new_shape.minor() - old_shape.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_shape.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                        if new_size <= self.capacity() {
                            let mut src = tail_start;
                            let mut dst = base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                let to_init_start = dst.add(to_copy_len);
                                MemoryRange::new_unchecked(to_init_start, to_init_len)
                                    .init(value.clone());
                            }
                            let to_init_start = dst.sub(to_init_len);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let new_base = new_data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                let to_init_start = dst.add(to_copy_len);
                                MemoryRange::new_unchecked(to_init_start, to_init_len)
                                    .init(value.clone());
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            let to_init_start = dst.add(to_copy_len);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init(value);
                            self.data = new_data;
                        }
                    }

                    Ordering::Equal => {
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base.add(old_size);
                            let mut dst = base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                let to_init_start = dst.add(to_copy_len);
                                MemoryRange::new_unchecked(to_init_start, to_init_len)
                                    .init(value.clone());
                            }
                            let to_init_start = dst.sub(to_init_len);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base.add(old_size);
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                let to_init_start = dst.add(to_copy_len);
                                MemoryRange::new_unchecked(to_init_start, to_init_len)
                                    .init(value.clone());
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            let to_init_start = dst.add(to_copy_len);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init(value);
                            self.data = new_data;
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_shape.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let tail_start = base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init(value.clone());
                            let mut src = base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                let to_init_start = dst.add(to_copy_len);
                                MemoryRange::new_unchecked(to_init_start, to_init_len)
                                    .init(value.clone());
                            }
                            let to_init_start = dst.sub(to_init_len);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init(value);
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let tail_start = new_base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init(value.clone());
                            let mut src = old_base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                let to_init_start = dst.add(to_copy_len);
                                MemoryRange::new_unchecked(to_init_start, to_init_len)
                                    .init(value.clone());
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            let to_init_start = dst.add(to_copy_len);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init(value);
                            self.data = new_data;
                        }
                    }
                }

                self.shape = new_shape;
                self.data.set_len(new_size);
            },
        }

        Ok(self)
    }

    pub fn resize_with<S, F>(&mut self, shape: S, mut initializer: F) -> Result<&mut Self>
    where
        S: AsShape,
        F: FnMut(Index) -> T,
    {
        // Refer to `Matrix::resize` for details.

        let order = self.order;
        let new_shape = AxisShape::from_shape(shape, self.order);
        let new_size = new_shape.size::<T>()?;
        let old_size = self.size();
        let new_stride = new_shape.stride();

        match (old_size, new_size) {
            (0, 0) => {
                self.shape = new_shape;
                return Ok(self);
            }

            (0, _) => unsafe {
                self.data.reserve(new_size);
                let tail_start_index = 0;
                let tail_start = self.data.as_mut_ptr();
                let tail_len = new_size;
                MemoryRange::new_unchecked(tail_start, tail_len).init_with(
                    tail_start_index,
                    order,
                    new_stride,
                    &mut initializer,
                );
                self.shape = new_shape;
                self.data.set_len(new_size);
                return Ok(self);
            },

            (_, 0) => unsafe {
                self.shape = new_shape;
                self.data.set_len(new_size);
                let tail_start = self.data.as_mut_ptr();
                let tail_len = old_size;
                MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                return Ok(self);
            },

            (_, _) => (),
        }

        let old_shape = self.shape;
        let old_stride = old_shape.stride();
        let minor_stride = old_stride.minor();

        let major_len_cmp = new_shape.major().cmp(&old_shape.major());
        let minor_len_cmp = new_shape.minor().cmp(&old_shape.minor());

        match minor_len_cmp {
            Ordering::Less => unsafe {
                self.shape = AxisShape::default();
                self.data.set_len(0);

                let to_copy_len = new_shape.minor() * minor_stride;
                let to_drop_len = (old_shape.minor() - new_shape.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let base = self.data.as_mut_ptr();
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_shape.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            ptr::copy(src, dst, to_copy_len);
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        }
                        let tail_start_index = new_shape.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => {
                        let base = self.data.as_mut_ptr();
                        let mut src = base;
                        let mut dst = base;
                        let to_drop_start = src.add(to_copy_len);
                        MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        for _ in 1..new_shape.major() {
                            src = src.add(old_stride.major());
                            dst = dst.add(new_stride.major());
                            ptr::copy(src, dst, to_copy_len);
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_shape.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base;
                            let mut dst = base;
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_shape.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                let to_drop_start = src.add(to_copy_len);
                                MemoryRange::new_unchecked(to_drop_start, to_drop_len)
                                    .drop_in_place();
                            }
                            let tail_start = base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init_with(
                                tail_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_mut_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base;
                            let mut dst = new_base;
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            let to_drop_start = src.add(to_copy_len);
                            MemoryRange::new_unchecked(to_drop_start, to_drop_len).drop_in_place();
                            for _ in 1..old_shape.major() {
                                src = src.add(old_stride.major());
                                dst = dst.add(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                let to_drop_start = src.add(to_copy_len);
                                MemoryRange::new_unchecked(to_drop_start, to_drop_len)
                                    .drop_in_place();
                            }
                            let tail_start = new_base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init_with(
                                tail_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                            self.data = new_data;
                        }
                    }
                }

                self.shape = new_shape;
                self.data.set_len(new_size);
            },

            Ordering::Equal => unsafe {
                match major_len_cmp {
                    Ordering::Less => {
                        self.shape = new_shape;
                        self.data.set_len(new_size);
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                    }

                    Ordering::Equal => (),

                    Ordering::Greater => {
                        let additional = new_size - old_size;
                        self.data.reserve(additional);
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = old_size;
                        let tail_start = base.add(tail_start_index);
                        let tail_len = additional;
                        MemoryRange::new_unchecked(tail_start, tail_len).init_with(
                            tail_start_index,
                            order,
                            new_stride,
                            &mut initializer,
                        );
                        self.shape = new_shape;
                        self.data.set_len(new_size);
                    }
                }
            },

            Ordering::Greater => unsafe {
                self.shape = AxisShape::default();
                self.data.set_len(0);

                let to_copy_len = old_shape.minor() * minor_stride;
                let to_init_len = (new_shape.minor() - old_shape.minor()) * minor_stride;

                match major_len_cmp {
                    Ordering::Less => {
                        let base = self.data.as_mut_ptr();
                        let tail_start_index = new_shape.major() * old_stride.major();
                        let tail_start = base.add(tail_start_index);
                        let tail_len = old_size - tail_start_index;
                        MemoryRange::new_unchecked(tail_start, tail_len).drop_in_place();
                        let mut to_init_start_index = new_size + to_copy_len;
                        if new_size <= self.capacity() {
                            let mut src = tail_start;
                            let mut dst = base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = base.add(to_init_start_index);
                                MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                    to_init_start_index,
                                    order,
                                    new_stride,
                                    &mut initializer,
                                );
                            }
                            to_init_start_index -= new_stride.major();
                            let to_init_start = base.add(to_init_start_index);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                to_init_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let new_base = new_data.as_mut_ptr();
                            let mut src = tail_start;
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = new_base.add(to_init_start_index);
                                MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                    to_init_start_index,
                                    order,
                                    new_stride,
                                    &mut initializer,
                                );
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            to_init_start_index -= new_stride.major();
                            let to_init_start = new_base.add(to_init_start_index);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                to_init_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                            self.data = new_data;
                        }
                    }

                    Ordering::Equal => {
                        let mut to_init_start_index = new_size + to_copy_len;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let mut src = base.add(old_size);
                            let mut dst = base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = base.add(to_init_start_index);
                                MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                    to_init_start_index,
                                    order,
                                    new_stride,
                                    &mut initializer,
                                );
                            }
                            to_init_start_index -= new_stride.major();
                            let to_init_start = base.add(to_init_start_index);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                to_init_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let mut src = old_base.add(old_size);
                            let mut dst = new_base.add(new_size);
                            for _ in 1..new_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = new_base.add(to_init_start_index);
                                MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                    to_init_start_index,
                                    order,
                                    new_stride,
                                    &mut initializer,
                                );
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            to_init_start_index -= new_stride.major();
                            let to_init_start = new_base.add(to_init_start_index);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                to_init_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                            self.data = new_data;
                        }
                    }

                    Ordering::Greater => {
                        let tail_start_index = old_shape.major() * new_stride.major();
                        let tail_len = new_size - tail_start_index;
                        let mut to_init_start_index = tail_start_index + to_copy_len;
                        if new_size <= self.capacity() {
                            let base = self.data.as_mut_ptr();
                            let tail_start = base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init_with(
                                tail_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                            let mut src = base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy(src, dst, to_copy_len);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = base.add(to_init_start_index);
                                MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                    to_init_start_index,
                                    order,
                                    new_stride,
                                    &mut initializer,
                                );
                            }
                            to_init_start_index -= new_stride.major();
                            let to_init_start = base.add(to_init_start_index);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                to_init_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                        } else {
                            let mut new_data = Vec::<T>::with_capacity(new_size);
                            let old_base = self.data.as_ptr();
                            let new_base = new_data.as_mut_ptr();
                            let tail_start = new_base.add(tail_start_index);
                            MemoryRange::new_unchecked(tail_start, tail_len).init_with(
                                tail_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                            let mut src = old_base.add(old_size);
                            let mut dst = tail_start;
                            for _ in 1..old_shape.major() {
                                src = src.sub(old_stride.major());
                                dst = dst.sub(new_stride.major());
                                ptr::copy_nonoverlapping(src, dst, to_copy_len);
                                to_init_start_index -= new_stride.major();
                                let to_init_start = new_base.add(to_init_start_index);
                                MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                    to_init_start_index,
                                    order,
                                    new_stride,
                                    &mut initializer,
                                );
                            }
                            src = src.sub(old_stride.major());
                            dst = dst.sub(new_stride.major());
                            ptr::copy_nonoverlapping(src, dst, to_copy_len);
                            to_init_start_index -= new_stride.major();
                            let to_init_start = new_base.add(to_init_start_index);
                            MemoryRange::new_unchecked(to_init_start, to_init_len).init_with(
                                to_init_start_index,
                                order,
                                new_stride,
                                &mut initializer,
                            );
                            self.data = new_data;
                        }
                    }
                }

                self.shape = new_shape;
                self.data.set_len(new_size);
            },
        }

        Ok(self)
    }
}

/// A struct representing a non-empty memory range.
struct MemoryRange<T> {
    start: *mut T,
    len: NonZero<usize>,
}

impl<T> MemoryRange<T> {
    /// Creates a new [`MemoryRange`].
    ///
    /// # Safety
    ///
    /// `len` must not be zero.
    unsafe fn new_unchecked(start: *mut T, len: usize) -> Self {
        let len = unsafe { NonZero::new_unchecked(len) };
        Self { start, len }
    }

    /// Initializes the memory range with the given value.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are
    /// violated:
    ///
    /// - The entire memory range must be contained within a single
    ///   allocated object.
    /// - The entire memory range must be [valid] for writes.
    /// - `self.start` must be properly aligned, even if `T` has size 0.
    ///
    /// If any part of the memory range has already been initialized,
    /// the original values will leak. However, this is considered safe.
    ///
    /// [valid]: https://doc.rust-lang.org/core/ptr/index.html#safety
    unsafe fn init(self, value: T)
    where
        T: Clone,
    {
        let mut to_init = self.start;
        unsafe {
            for _ in 1..self.len.get() {
                ptr::write(to_init, value.clone());
                to_init = to_init.add(1);
            }
            ptr::write(to_init, value);
        }
    }

    /// Initializes the memory range with values initialized using
    /// their indices.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are
    /// violated:
    ///
    /// - The entire memory range must be contained within a single
    ///   allocated object.
    /// - The entire memory range must be [valid] for writes.
    /// - `self.start` must be properly aligned, even if `T` has size 0.
    ///
    /// If any part of the memory range has already been initialized,
    /// the original values will leak. However, this is considered safe.
    ///
    /// [valid]: https://doc.rust-lang.org/core/ptr/index.html#safety
    unsafe fn init_with<F>(
        self,
        start_index: usize,
        order: Order,
        new_stride: Stride,
        initializer: &mut F,
    ) where
        F: FnMut(Index) -> T,
    {
        let mut to_init_index = start_index;
        let mut to_init = self.start;
        unsafe {
            for _ in 1..self.len.get() {
                let index = Index::from_flattened(to_init_index, order, new_stride);
                let value = initializer(index);
                ptr::write(to_init, value);
                to_init_index += 1;
                to_init = to_init.add(1);
            }
            let index = Index::from_flattened(to_init_index, order, new_stride);
            let value = initializer(index);
            ptr::write(to_init, value);
        }
    }

    /// Drops the memory range.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are
    /// violated:
    ///
    /// - The entire memory range must be contained within a single
    ///   allocated object.
    /// - The entire memory range must be [valid] for both reads and
    ///   writes.
    /// - The entire memory range must be fully initialized.
    /// - `self.start` must be properly aligned, even if `T` has size 0.
    ///
    /// Refer to [`ptr::drop_in_place`] for more exhaustive safety
    /// concerns.
    ///
    /// [valid]: https://doc.rust-lang.org/core/ptr/index.html#safety
    unsafe fn drop_in_place(self) {
        let to_drop = ptr::slice_from_raw_parts_mut(self.start, self.len.get());
        unsafe {
            ptr::drop_in_place(to_drop);
        }
    }
}

#[cfg(test)]
mod tests {
    use self::mock::{Mock, Scope};
    use super::*;
    use crate::error::Error;
    use crate::shape::Shape;
    use crate::testkit;

    mod mock {
        extern crate std;

        use std::cell::Cell;
        use std::marker::PhantomData;
        use std::thread_local;

        thread_local! {
            static COUNT: Cell<Count> = const { Cell::new(Count::new()) };
        }

        #[derive(Clone, Copy)]
        struct Count {
            clone: usize,
            drop: usize,
        }

        impl Count {
            const fn new() -> Self {
                Self { clone: 0, drop: 0 }
            }
        }

        pub(super) struct Scope {
            outer_count: Count,
            marker: PhantomData<*const ()>,
        }

        impl Scope {
            pub(super) fn with<F>(f: F)
            where
                F: FnOnce(&Scope),
            {
                let outer_count = COUNT.get();
                COUNT.set(Count::new());
                let scope = Self {
                    outer_count,
                    marker: PhantomData,
                };
                f(&scope);
            }

            /// Returns the current clone count for the `Mock` instances in
            /// the innermost active `Scope` on this thread, regardless of
            /// which `Scope` instance this method is called on.
            ///
            /// # Notes
            ///
            /// All concrete types of `Mock<T>` are counted indiscriminately.
            pub(super) fn clone_count(&self) -> usize {
                COUNT.get().clone
            }

            /// Returns the current drop count for the `Mock` instances in
            /// the innermost active `Scope` on this thread, regardless of
            /// which `Scope` instance this method is called on.
            ///
            /// # Notes
            ///
            /// All concrete types of `Mock<T>` are counted indiscriminately.
            pub(super) fn drop_count(&self) -> usize {
                COUNT.get().drop
            }
        }

        impl Drop for Scope {
            fn drop(&mut self) {
                let _ = COUNT.try_with(|cell| {
                    cell.update(|count| {
                        let clone = self.outer_count.clone + count.clone;
                        let drop = self.outer_count.drop + count.drop;
                        Count { clone, drop }
                    })
                });
            }
        }

        #[derive(Debug, PartialEq)]
        pub(super) struct Mock<T>(pub(super) T);

        impl<T> Clone for Mock<T>
        where
            T: Clone,
        {
            fn clone(&self) -> Self {
                COUNT.with(|cell| {
                    cell.update(|mut count| {
                        count.clone += 1;
                        count
                    })
                });
                Mock(self.0.clone())
            }
        }

        impl<T> Drop for Mock<T> {
            fn drop(&mut self) {
                let _ = COUNT.try_with(|cell| {
                    cell.update(|mut count| {
                        count.drop += 1;
                        count
                    })
                });
            }
        }
    }

    fn assert_counts(scope: &Scope, old_shape: Shape, new_shape: Shape) {
        let old_nrows = old_shape.nrows();
        let old_ncols = old_shape.ncols();
        let new_nrows = new_shape.nrows();
        let new_ncols = new_shape.ncols();
        let expected_init_count;
        let expected_drop_count;
        match (new_nrows.cmp(&old_nrows), new_ncols.cmp(&old_ncols)) {
            (Ordering::Less, Ordering::Less) => {
                expected_init_count = 0;
                expected_drop_count = old_nrows * old_ncols - new_nrows * new_ncols;
            }
            (Ordering::Less, Ordering::Equal) => {
                expected_init_count = 0;
                expected_drop_count = (old_nrows - new_nrows) * old_ncols;
            }
            (Ordering::Less, Ordering::Greater) => {
                expected_init_count = (new_ncols - old_ncols) * new_nrows;
                expected_drop_count = (old_nrows - new_nrows) * old_ncols;
            }
            (Ordering::Equal, Ordering::Less) => {
                expected_init_count = 0;
                expected_drop_count = (old_ncols - new_ncols) * old_nrows;
            }
            (Ordering::Equal, Ordering::Equal) => {
                expected_init_count = 0;
                expected_drop_count = 0;
            }
            (Ordering::Equal, Ordering::Greater) => {
                expected_init_count = (new_ncols - old_ncols) * old_nrows;
                expected_drop_count = 0;
            }
            (Ordering::Greater, Ordering::Less) => {
                expected_init_count = (new_nrows - old_nrows) * new_ncols;
                expected_drop_count = (old_ncols - new_ncols) * old_nrows;
            }
            (Ordering::Greater, Ordering::Equal) => {
                expected_init_count = (new_nrows - old_nrows) * new_ncols;
                expected_drop_count = 0;
            }
            (Ordering::Greater, Ordering::Greater) => {
                expected_init_count = new_nrows * new_ncols - old_nrows * old_ncols;
                expected_drop_count = 0;
            }
        }
        if expected_init_count < 1 {
            assert_eq!(scope.clone_count(), 0);
            // Argument `value` passed to `Matrix::resize`
            // incurs an extra drop.
            assert_eq!(scope.drop_count(), expected_drop_count + 1);
        } else {
            // Argument `value` passed to `Matrix::resize`
            // saves an extra clone.
            assert_eq!(scope.clone_count(), expected_init_count - 1);
            assert_eq!(scope.drop_count(), expected_drop_count);
        }
    }

    #[test]
    fn test_resize() {
        #[cfg(miri)]
        let lens = [0, 1, 2, 3, 5];
        #[cfg(not(miri))]
        let lens = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19];

        for old_nrows in lens {
            for old_ncols in lens {
                let old_shape = Shape::new(old_nrows, old_ncols);

                for new_nrows in lens {
                    for new_ncols in lens {
                        let new_shape = Shape::new(new_nrows, new_ncols);

                        // For zero-sized type.
                        let matrix = Matrix::with_initializer(old_shape, |_| Mock(())).unwrap();
                        testkit::for_each_order_unary(matrix, |mut matrix| {
                            Scope::with(|scope| {
                                matrix.resize(new_shape, Mock(())).unwrap();
                                assert_counts(scope, old_shape, new_shape);
                                let expected =
                                    Matrix::with_initializer(new_shape, |_| Mock(())).unwrap();
                                testkit::assert_loose_eq(&matrix, &expected);
                            });
                        });

                        // For non-zero-sized type.
                        let old_size = old_shape.size().unwrap();
                        let new_size = new_shape.size().unwrap();
                        if new_size <= old_size {
                            let matrix = Matrix::with_initializer(old_shape, Mock).unwrap();
                            testkit::for_each_order_unary(matrix, |mut matrix| {
                                Scope::with(|scope| {
                                    matrix.resize(new_shape, Mock(Index::default())).unwrap();
                                    assert_counts(scope, old_shape, new_shape);
                                    let expected = Matrix::with_initializer(new_shape, |index| {
                                        if index.row >= old_nrows || index.col >= old_ncols {
                                            Mock(Index::default())
                                        } else {
                                            Mock(index)
                                        }
                                    })
                                    .unwrap();
                                    testkit::assert_loose_eq(&matrix, &expected);
                                });
                            });
                        } else {
                            let matrix = Matrix::with_initializer(old_shape, Mock).unwrap();
                            testkit::for_each_order_unary(matrix, |mut matrix| {
                                Scope::with(|scope| {
                                    // Ensure the in place path is taken.
                                    matrix.data.reserve(new_size - old_size);
                                    assert!(new_size <= matrix.capacity());
                                    matrix.resize(new_shape, Mock(Index::default())).unwrap();
                                    assert_counts(scope, old_shape, new_shape);
                                    let expected = Matrix::with_initializer(new_shape, |index| {
                                        if index.row >= old_nrows || index.col >= old_ncols {
                                            Mock(Index::default())
                                        } else {
                                            Mock(index)
                                        }
                                    })
                                    .unwrap();
                                    testkit::assert_loose_eq(&matrix, &expected);
                                });
                            });

                            let matrix = Matrix::with_initializer(old_shape, Mock).unwrap();
                            testkit::for_each_order_unary(matrix, |mut matrix| {
                                Scope::with(|scope| {
                                    // Ensure the reallocation path is taken.
                                    matrix.data.shrink_to_fit();
                                    assert!(new_size > matrix.capacity());
                                    matrix.resize(new_shape, Mock(Index::default())).unwrap();
                                    assert_counts(scope, old_shape, new_shape);
                                    let expected = Matrix::with_initializer(new_shape, |index| {
                                        if index.row >= old_nrows || index.col >= old_ncols {
                                            Mock(Index::default())
                                        } else {
                                            Mock(index)
                                        }
                                    })
                                    .unwrap();
                                    testkit::assert_loose_eq(&matrix, &expected);
                                });
                            });
                        }
                    }
                }

                let new_shape = Shape::new(usize::MAX, 2);
                let matrix = Matrix::with_initializer(old_shape, |_| Mock(())).unwrap();
                testkit::for_each_order_unary(matrix, |mut matrix| {
                    Scope::with(|scope| {
                        let error = matrix.resize(new_shape, Mock(())).unwrap_err();
                        assert_eq!(error, Error::SizeOverflow);
                        assert_eq!(scope.clone_count(), 0);
                        assert_eq!(scope.drop_count(), 1);
                    });
                });

                let new_shape = Shape::new(isize::MAX as usize / 4 + 1, 1);
                let matrix: Matrix<Mock<i32>> =
                    Matrix::with_initializer(old_shape, |_| Mock(0)).unwrap();
                testkit::for_each_order_unary(matrix, |mut matrix| {
                    Scope::with(|scope| {
                        let error = matrix.resize(new_shape, Mock(0)).unwrap_err();
                        assert_eq!(error, Error::CapacityOverflow);
                        assert_eq!(scope.clone_count(), 0);
                        assert_eq!(scope.drop_count(), 1);
                    });
                });

                let new_shape = Shape::new(isize::MAX as usize + 1, 1);
                let matrix: Matrix<Mock<u8>> =
                    Matrix::with_initializer(old_shape, |_| Mock(0)).unwrap();
                testkit::for_each_order_unary(matrix, |mut matrix| {
                    Scope::with(|scope| {
                        let error = matrix.resize(new_shape, Mock(0)).unwrap_err();
                        assert_eq!(error, Error::CapacityOverflow);
                        assert_eq!(scope.clone_count(), 0);
                        assert_eq!(scope.drop_count(), 1);
                    });
                });

                // Unable to cover.
                // let new_shape = Shape::new(isize::MAX as usize + 1, 1);
                // let matrix: Matrix<Mock<()>> =
                //     Matrix::with_initializer(old_shape, |_| Mock(())).unwrap();
                // testkit::for_each_order_unary(matrix, |mut matrix| {
                //     Scope::with(|scope| {
                //         assert!(matrix.resize(new_shape, Mock(())).is_ok());
                //         assert_eq!(scope.clone_count(), isize::MAX as usize);
                //         assert_eq!(scope.drop_count(), 0);
                //     });
                // });
            }
        }
    }
}
