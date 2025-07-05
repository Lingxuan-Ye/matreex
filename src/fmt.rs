use crate::Matrix;
use crate::index::Index;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

mod constant {
    pub(super) mod whitespace {
        pub(in super::super) const SPACE: &str = " ";
        pub(in super::super) const INDENT: &str = "    ";
        pub(in super::super) const NEWLINE: &str = "\n";
    }

    pub(super) mod matrix {
        pub(in super::super) const DELIMITER_LEFT: &str = "[";
        pub(in super::super) const DELIMITER_RIGHT: &str = "]";
    }

    pub(super) mod row {
        pub(in super::super) const INDEX_GAP: &str = "  ";
        pub(in super::super) const DELIMITER_LEFT: &str = "[  ";
        pub(in super::super) const DELIMITER_RIGHT: &str = "  ]";
        pub(in super::super) const DELIMITER_PADDING: &str = "   ";
        pub(in super::super) const SEPARATOR: &str = "";
        pub(in super::super) const SEPARATOR_PADDING: &str = "";
    }

    pub(super) mod element {
        pub(in super::super) const INDEX_GAP: &str = " ";
        pub(in super::super) const SEPARATOR: &str = "  ";
        pub(in super::super) const SEPARATOR_PADDING: &str = "  ";
    }
}

// Refactoring the `Debug` and `Display` implementations will NOT
// be considered a breaking change.

impl<T> fmt::Debug for Matrix<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str(constant::matrix::DELIMITER_LEFT)?;
            f.write_str(constant::matrix::DELIMITER_RIGHT)?;
            return Ok(());
        }

        let order = self.order;
        let stride = self.stride();
        let shape = self.shape();
        let nrows = shape.nrows();
        let ncols = shape.ncols();
        let size = self.size();
        let index_width = size.to_string().chars().count();
        let mut element_width = 0;
        let mut element_hight = 0;
        let mut cache = Vec::with_capacity(size);
        for element in self.data.iter() {
            let lines = Lines::from_debug(element);
            let width = lines.width();
            if width > element_width {
                element_width = width;
            }
            let height = lines.height();
            if height > element_hight {
                element_hight = height;
            }
            cache.push(lines);
        }
        let index_padding = constant::whitespace::SPACE.repeat(index_width);
        let element_padding = constant::whitespace::SPACE.repeat(element_width);

        f.write_str(constant::matrix::DELIMITER_LEFT)?;
        f.write_str(constant::whitespace::NEWLINE)?;

        f.write_str(constant::whitespace::INDENT)?;
        f.write_str(&index_padding)?;
        f.write_str(constant::row::INDEX_GAP)?;
        f.write_str(constant::row::DELIMITER_PADDING)?;
        for col in 0..ncols {
            if col != 0 {
                f.write_str(constant::element::SEPARATOR_PADDING)?;
            }
            f.write_index(col, index_width)?;
            f.write_str(constant::element::INDEX_GAP)?;
            f.write_str(&element_padding)?;
        }
        f.write_str(constant::row::DELIMITER_PADDING)?;
        f.write_str(constant::row::SEPARATOR_PADDING)?;
        f.write_str(constant::whitespace::NEWLINE)?;

        for row in 0..nrows {
            // The first line of the element representation
            f.write_str(constant::whitespace::INDENT)?;
            f.write_index(row, index_width)?;
            f.write_str(constant::row::INDEX_GAP)?;
            f.write_str(constant::row::DELIMITER_LEFT)?;
            for col in 0..ncols {
                if col != 0 {
                    f.write_str(constant::element::SEPARATOR)?;
                }
                let index = Index::new(row, col).to_flattened(order, stride);
                f.write_index(index, index_width)?;
                f.write_str(constant::element::INDEX_GAP)?;
                match cache[index].next() {
                    None if element_width > 0 => f.write_str(&element_padding)?,
                    Some(line) => f.write_element_line(line, element_width)?,
                    _ => (),
                }
            }
            f.write_str(constant::row::DELIMITER_RIGHT)?;
            f.write_str(constant::row::SEPARATOR)?;
            f.write_str(constant::whitespace::NEWLINE)?;

            // The remaining lines of the element representation.
            for _ in 1..element_hight {
                f.write_str(constant::whitespace::INDENT)?;
                f.write_str(&index_padding)?;
                f.write_str(constant::row::INDEX_GAP)?;
                f.write_str(constant::row::DELIMITER_PADDING)?;
                for col in 0..ncols {
                    if col != 0 {
                        f.write_str(constant::element::SEPARATOR_PADDING)?;
                    }
                    let index = Index::new(row, col).to_flattened(order, stride);
                    f.write_str(&index_padding)?;
                    f.write_str(constant::element::INDEX_GAP)?;
                    match cache[index].next() {
                        None if element_width > 0 => f.write_str(&element_padding)?,
                        Some(line) => f.write_element_line(line, element_width)?,
                        _ => (),
                    }
                }
                f.write_str(constant::row::DELIMITER_PADDING)?;
                f.write_str(constant::row::SEPARATOR_PADDING)?;
                f.write_str(constant::whitespace::NEWLINE)?;
            }
        }

        f.write_str(constant::matrix::DELIMITER_RIGHT)
    }
}

impl<T> fmt::Display for Matrix<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str(constant::matrix::DELIMITER_LEFT)?;
            f.write_str(constant::matrix::DELIMITER_RIGHT)?;
            return Ok(());
        }

        let order = self.order;
        let stride = self.stride();
        let shape = self.shape();
        let nrows = shape.nrows();
        let ncols = shape.ncols();
        let size = self.size();
        let mut element_width = 0;
        let mut element_hight = 0;
        let mut cache = Vec::with_capacity(size);
        for element in self.data.iter() {
            let lines = Lines::from_display(element);
            let width = lines.width();
            if width > element_width {
                element_width = width;
            }
            let height = lines.height();
            if height > element_hight {
                element_hight = height;
            }
            cache.push(lines);
        }
        let element_padding = constant::whitespace::SPACE.repeat(element_width);

        f.write_str(constant::matrix::DELIMITER_LEFT)?;
        f.write_str(constant::whitespace::NEWLINE)?;

        for row in 0..nrows {
            // The first line of the element representation.
            f.write_str(constant::whitespace::INDENT)?;
            f.write_str(constant::row::DELIMITER_LEFT)?;
            for col in 0..ncols {
                if col != 0 {
                    f.write_str(constant::element::SEPARATOR)?;
                }
                let index = Index::new(row, col).to_flattened(order, stride);
                match cache[index].next() {
                    None if element_width > 0 => f.write_str(&element_padding)?,
                    Some(line) => f.write_element_line(line, element_width)?,
                    _ => (),
                }
            }
            f.write_str(constant::row::DELIMITER_RIGHT)?;
            f.write_str(constant::row::SEPARATOR)?;
            f.write_str(constant::whitespace::NEWLINE)?;

            // The remaining lines of the element representation.
            for _ in 1..element_hight {
                f.write_str(constant::whitespace::INDENT)?;
                f.write_str(constant::row::DELIMITER_PADDING)?;
                for col in 0..ncols {
                    if col != 0 {
                        f.write_str(constant::element::SEPARATOR_PADDING)?;
                    }
                    let index = Index::new(row, col).to_flattened(order, stride);
                    match cache[index].next() {
                        None if element_width > 0 => f.write_str(&element_padding)?,
                        Some(line) => f.write_element_line(line, element_width)?,
                        _ => (),
                    }
                }
                f.write_str(constant::row::DELIMITER_PADDING)?;
                f.write_str(constant::row::SEPARATOR_PADDING)?;
                f.write_str(constant::whitespace::NEWLINE)?;
            }
        }

        f.write_str(constant::matrix::DELIMITER_RIGHT)
    }
}

#[derive(Debug)]
struct Lines(VecDeque<String>);

impl Lines {
    fn from_debug<T>(element: T) -> Self
    where
        T: fmt::Debug,
    {
        Self(format!("{element:?}").lines().map(String::from).collect())
    }

    fn from_display<T>(element: T) -> Self
    where
        T: fmt::Display,
    {
        Self(format!("{element}").lines().map(String::from).collect())
    }

    fn width(&self) -> usize {
        self.0
            .iter()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0)
    }

    fn height(&self) -> usize {
        self.0.len()
    }
}

impl Iterator for Lines {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for Lines {
    fn len(&self) -> usize {
        self.0.len()
    }
}

trait Write {
    fn write_index(&mut self, index: usize, width: usize) -> fmt::Result;

    fn write_element_line(&mut self, line: String, width: usize) -> fmt::Result;
}

impl<T> Write for T
where
    T: fmt::Write,
{
    #[cfg(not(feature = "pretty-debug"))]
    fn write_index(&mut self, index: usize, width: usize) -> fmt::Result {
        write!(self, "{index:>width$}")
    }

    #[cfg(feature = "pretty-debug")]
    fn write_index(&mut self, index: usize, width: usize) -> fmt::Result {
        let plain = format!("{index:>width$}");
        let style = owo_colors::Style::new().green().dimmed();
        let styled = style.style(plain);
        write!(self, "{styled}")
    }

    fn write_element_line(&mut self, line: String, width: usize) -> fmt::Result {
        write!(self, "{line:<width$}")
    }
}
