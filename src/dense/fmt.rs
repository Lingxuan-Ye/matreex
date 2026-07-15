use super::Matrix;
use super::order::Order;
use crate::index::Index;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use core::fmt;

mod whitespace {
    pub(super) const SPACE: &str = " ";
    pub(super) const INDENT: &str = "    ";
    pub(super) const NEWLINE: &str = "\n";
}

mod matrix {
    pub(super) const DELIMITER_LEFT: &str = "[";
    pub(super) const DELIMITER_RIGHT: &str = "]";
}

mod row {
    pub(super) const INDEX_GAP: &str = "  ";
    pub(super) const DELIMITER_LEFT: &str = "[  ";
    pub(super) const DELIMITER_RIGHT: &str = "  ]";
    pub(super) const DELIMITER_PADDING: &str = "   ";
    pub(super) const SEPARATOR: &str = "";
    pub(super) const SEPARATOR_PADDING: &str = "";
}

mod element {
    pub(super) const INDEX_GAP: &str = " ";
    pub(super) const SEPARATOR: &str = "  ";
    pub(super) const SEPARATOR_PADDING: &str = "  ";
}

impl<T, O> fmt::Debug for Matrix<T, O>
where
    T: fmt::Debug,
    O: Order,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str(matrix::DELIMITER_LEFT)?;
            f.write_str(matrix::DELIMITER_RIGHT)?;
            return Ok(());
        }

        let stride = self.stride();
        let shape = self.shape();
        let index_width = self.size().to_string().chars().count();
        let mut element_width = 0;
        let mut element_hight = 0;
        let mut cache: Box<[Lines]> = self
            .data
            .iter()
            .map(|element| {
                let string = if f.alternate() {
                    format!("{element:#?}")
                } else {
                    format!("{element:?}")
                };
                let (lines, info) = Lines::new(string);
                element_width = usize::max(element_width, info.width);
                element_hight = usize::max(element_hight, info.height);
                lines
            })
            .collect();
        let index_padding = whitespace::SPACE.repeat(index_width);
        let element_padding = whitespace::SPACE.repeat(element_width);

        f.write_str(matrix::DELIMITER_LEFT)?;
        f.write_str(whitespace::NEWLINE)?;

        f.write_str(whitespace::INDENT)?;
        f.write_str(&index_padding)?;
        f.write_str(row::INDEX_GAP)?;
        f.write_str(row::DELIMITER_PADDING)?;
        for col in 0..shape.ncols {
            if col != 0 {
                f.write_str(element::SEPARATOR_PADDING)?;
            }
            f.write_index(col, index_width)?;
            f.write_str(element::INDEX_GAP)?;
            f.write_str(&element_padding)?;
        }
        f.write_str(row::DELIMITER_PADDING)?;
        f.write_str(row::SEPARATOR_PADDING)?;
        f.write_str(whitespace::NEWLINE)?;

        for row in 0..shape.nrows {
            // The first line of the element representation.
            f.write_str(whitespace::INDENT)?;
            f.write_index(row, index_width)?;
            f.write_str(row::INDEX_GAP)?;
            f.write_str(row::DELIMITER_LEFT)?;
            for col in 0..shape.ncols {
                if col != 0 {
                    f.write_str(element::SEPARATOR)?;
                }
                let index = Index::new(row, col).to_linear::<O>(stride);
                f.write_index(index, index_width)?;
                f.write_str(element::INDEX_GAP)?;
                match cache[index].next_line() {
                    None if element_width > 0 => f.write_str(&element_padding)?,
                    Some(line) => f.write_element_line(line, element_width)?,
                    _ => (),
                }
            }
            f.write_str(row::DELIMITER_RIGHT)?;
            f.write_str(row::SEPARATOR)?;
            f.write_str(whitespace::NEWLINE)?;

            // The remaining lines of the element representation.
            for _ in 1..element_hight {
                f.write_str(whitespace::INDENT)?;
                f.write_str(&index_padding)?;
                f.write_str(row::INDEX_GAP)?;
                f.write_str(row::DELIMITER_PADDING)?;
                for col in 0..shape.ncols {
                    if col != 0 {
                        f.write_str(element::SEPARATOR_PADDING)?;
                    }
                    let index = Index::new(row, col).to_linear::<O>(stride);
                    f.write_str(&index_padding)?;
                    f.write_str(element::INDEX_GAP)?;
                    match cache[index].next_line() {
                        None if element_width > 0 => f.write_str(&element_padding)?,
                        Some(line) => f.write_element_line(line, element_width)?,
                        _ => (),
                    }
                }
                f.write_str(row::DELIMITER_PADDING)?;
                f.write_str(row::SEPARATOR_PADDING)?;
                f.write_str(whitespace::NEWLINE)?;
            }
        }

        f.write_str(matrix::DELIMITER_RIGHT)
    }
}

impl<T, O> fmt::Display for Matrix<T, O>
where
    T: fmt::Display,
    O: Order,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str(matrix::DELIMITER_LEFT)?;
            f.write_str(matrix::DELIMITER_RIGHT)?;
            return Ok(());
        }

        let stride = self.stride();
        let shape = self.shape();
        let mut element_width = 0;
        let mut element_hight = 0;
        let mut cache: Box<[Lines]> = self
            .data
            .iter()
            .map(|element| {
                let string = format!("{element}");
                let (lines, info) = Lines::new(string);
                element_width = usize::max(element_width, info.width);
                element_hight = usize::max(element_hight, info.height);
                lines
            })
            .collect();
        let element_padding = whitespace::SPACE.repeat(element_width);

        f.write_str(matrix::DELIMITER_LEFT)?;
        f.write_str(whitespace::NEWLINE)?;

        for row in 0..shape.nrows {
            // The first line of the element representation.
            f.write_str(whitespace::INDENT)?;
            f.write_str(row::DELIMITER_LEFT)?;
            for col in 0..shape.ncols {
                if col != 0 {
                    f.write_str(element::SEPARATOR)?;
                }
                let index = Index::new(row, col).to_linear::<O>(stride);
                match cache[index].next_line() {
                    None if element_width > 0 => f.write_str(&element_padding)?,
                    Some(line) => f.write_element_line(line, element_width)?,
                    _ => (),
                }
            }
            f.write_str(row::DELIMITER_RIGHT)?;
            f.write_str(row::SEPARATOR)?;
            f.write_str(whitespace::NEWLINE)?;

            // The remaining lines of the element representation.
            for _ in 1..element_hight {
                f.write_str(whitespace::INDENT)?;
                f.write_str(row::DELIMITER_PADDING)?;
                for col in 0..shape.ncols {
                    if col != 0 {
                        f.write_str(element::SEPARATOR_PADDING)?;
                    }
                    let index = Index::new(row, col).to_linear::<O>(stride);
                    match cache[index].next_line() {
                        None if element_width > 0 => f.write_str(&element_padding)?,
                        Some(line) => f.write_element_line(line, element_width)?,
                        _ => (),
                    }
                }
                f.write_str(row::DELIMITER_PADDING)?;
                f.write_str(row::SEPARATOR_PADDING)?;
                f.write_str(whitespace::NEWLINE)?;
            }
        }

        f.write_str(matrix::DELIMITER_RIGHT)
    }
}

#[derive(Debug)]
struct Lines {
    index: usize,
    string: Box<str>,
}

#[derive(Debug)]
struct LinesInfo {
    width: usize,
    height: usize,
}

impl Lines {
    fn new(string: String) -> (Self, LinesInfo) {
        let index = 0;
        let string = string.into_boxed_str();
        let mut width = 0;
        let mut height = 0;
        for line in string.lines() {
            let line_width = line.chars().count();
            width = usize::max(width, line_width);
            height += 1;
        }
        let lines = Self { index, string };
        let info = LinesInfo { width, height };
        (lines, info)
    }

    fn next_line(&mut self) -> Option<&str> {
        let len = self.string.len();
        if self.index == len {
            return None;
        }
        let slice = &self.string[self.index..];
        match slice.find('\n') {
            None => {
                self.index = len;
                Some(slice)
            }
            Some(index) => {
                self.index += index + 1;
                let slice = &slice[..index];
                slice.strip_suffix('\r').or(Some(slice))
            }
        }
    }
}

trait Write {
    fn write_index(&mut self, index: usize, width: usize) -> fmt::Result;

    fn write_element_line(&mut self, line: &str, width: usize) -> fmt::Result;
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
        let style = owo_colors::Style::new().green();
        let styled = style.style(plain);
        write!(self, "{styled}")
    }

    fn write_element_line(&mut self, line: &str, width: usize) -> fmt::Result {
        write!(self, "{line:<width$}")
    }
}
