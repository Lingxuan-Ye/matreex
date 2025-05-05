use crate::Matrix;
use crate::index::Index;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

const LEFT_DELIMITER: &str = "[";
const RIGHT_DELIMITER: &str = "]";
const SPACE: &str = " ";
const INDENT: &str = "    ";
const ROW_INNER: &str = "  ";
const ELEMENT_INNER: &str = " ";
const ELEMENT_OUTER: &str = "  ";

#[cfg(not(feature = "pretty-debug"))]
macro_rules! write_index {
    ($dst:expr, $($arg:tt)*) => {
        write!($dst, $($arg)*)
    };
}

#[cfg(feature = "pretty-debug")]
macro_rules! write_index {
    ($dst:expr, $($arg:tt)*) => {{
        let plain = format!($($arg)*);
        let styled = owo_colors::Style::new().green().dimmed().style(plain);
        write!($dst, "{styled}")
    }};
}

impl<T> fmt::Debug for Matrix<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str(LEFT_DELIMITER)?;
            f.write_str(RIGHT_DELIMITER)?;
            return Ok(());
        }

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

        f.write_str(LEFT_DELIMITER)?;
        writeln!(f)?;

        f.write_str(INDENT)?;
        write!(f, "{SPACE:>index_width$}")?;
        f.write_str(ROW_INNER)?;
        f.write_str(SPACE)?;
        for col in 0..ncols {
            if col != 0 {
                f.write_str(ELEMENT_OUTER)?;
            }
            write_index!(f, "{col:>index_width$}")?;
            f.write_str(ELEMENT_INNER)?;
            write!(f, "{SPACE:<element_width$}")?;
        }
        writeln!(f)?;

        for row in 0..nrows {
            // first line of the element representation
            f.write_str(INDENT)?;
            write_index!(f, "{row:>index_width$}")?;
            f.write_str(ROW_INNER)?;
            f.write_str(LEFT_DELIMITER)?;
            for col in 0..ncols {
                if col != 0 {
                    f.write_str(ELEMENT_OUTER)?;
                }
                // hope loop-invariant code motion applies here,
                // as well as to similar code
                let index = Index::new(row, col).to_flattened(self.order, self.shape);
                write_index!(f, "{index:>index_width$}")?;
                f.write_str(ELEMENT_INNER)?;
                match cache[index].next() {
                    None if element_width > 0 => write!(f, "{SPACE:<element_width$}")?,
                    Some(line) => write!(f, "{line:<element_width$}")?,
                    _ => (),
                }
            }
            f.write_str(RIGHT_DELIMITER)?;
            writeln!(f)?;

            // remaining lines of the element representation
            for _ in 1..element_hight {
                f.write_str(INDENT)?;
                write!(f, "{SPACE:>index_width$}")?;
                f.write_str(ROW_INNER)?;
                f.write_str(SPACE)?;
                for col in 0..ncols {
                    if col != 0 {
                        f.write_str(ELEMENT_OUTER)?;
                    }
                    let index = Index::new(row, col).to_flattened(self.order, self.shape);
                    write!(f, "{SPACE:>index_width$}")?;
                    f.write_str(ELEMENT_INNER)?;
                    match cache[index].next() {
                        None if element_width > 0 => write!(f, "{SPACE:<element_width$}")?,
                        Some(line) => write!(f, "{line:<element_width$}")?,
                        _ => (),
                    }
                }
                writeln!(f)?;
            }
        }

        f.write_str(RIGHT_DELIMITER)
    }
}

impl<T> fmt::Display for Matrix<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            f.write_str(LEFT_DELIMITER)?;
            f.write_str(RIGHT_DELIMITER)?;
            return Ok(());
        }

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

        f.write_str(LEFT_DELIMITER)?;
        writeln!(f)?;

        for row in 0..nrows {
            // first line of the element representation
            f.write_str(INDENT)?;
            f.write_str(LEFT_DELIMITER)?;
            for col in 0..ncols {
                if col != 0 {
                    f.write_str(ELEMENT_OUTER)?;
                }
                let index = Index::new(row, col).to_flattened(self.order, self.shape);
                match cache[index].next() {
                    None if element_width > 0 => write!(f, "{SPACE:<element_width$}")?,
                    Some(line) => write!(f, "{line:<element_width$}")?,
                    _ => (),
                }
            }
            f.write_str(RIGHT_DELIMITER)?;
            writeln!(f)?;

            // remaining lines of the element representation
            for _ in 1..element_hight {
                f.write_str(INDENT)?;
                f.write_str(SPACE)?;
                for col in 0..ncols {
                    if col != 0 {
                        f.write_str(ELEMENT_OUTER)?;
                    }
                    let index = Index::new(row, col).to_flattened(self.order, self.shape);
                    match cache[index].next() {
                        None if element_width > 0 => write!(f, "{SPACE:<element_width$}")?,
                        Some(line) => write!(f, "{line:<element_width$}")?,
                        _ => (),
                    }
                }
                writeln!(f)?;
            }
        }

        f.write_str(RIGHT_DELIMITER)
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
