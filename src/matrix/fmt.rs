use super::index::Index;
use super::Matrix;

const LEFT_DELIMITER: &str = "[";
const RIGHT_DELIMITER: &str = "]";
const SPACE: &str = " ";
const TAB_SIZE: usize = 4;
const OUTER_GAP: usize = 2;
const INTER_GAP: usize = 2;
const INNER_GAP: usize = 1;

macro_rules! write_dim {
    ($dst:expr, $($arg:tt)*) => {
        std::write!($dst, "\u{001b}[2m{}\u{001b}[22m", std::format_args!($($arg)*))
    };
}

impl<T: std::fmt::Debug> std::fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let nrows = shape.nrows;
        let ncols = shape.ncols;
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

        writeln!(f, "Matrix{SPACE}{{")?;
        writeln!(f, "{SPACE:TAB_SIZE$}order:{SPACE}{:?}", self.order)?;
        writeln!(f, "{SPACE:TAB_SIZE$}shape:{SPACE}{:?}", self.shape)?;
        writeln!(f, "{SPACE:TAB_SIZE$}data:")?;

        write!(f, "{SPACE:TAB_SIZE$}{SPACE:TAB_SIZE$}")?;
        write!(f, "{LEFT_DELIMITER:<TAB_SIZE$}")?;
        write!(f, "{SPACE:>index_width$}")?;
        write!(f, "{SPACE:OUTER_GAP$}")?;
        write!(f, "{SPACE}")?;
        for col in 0..ncols {
            if col != 0 {
                write!(f, "{SPACE:INTER_GAP$}")?;
            }
            write_dim!(f, "{col:>index_width$}")?;
            write!(f, "{SPACE:INNER_GAP$}")?;
            write!(f, "{SPACE:<element_width$}")?;
        }
        writeln!(f)?;

        for row in 0..nrows {
            // first line of the element representation
            write!(f, "{SPACE:TAB_SIZE$}{SPACE:TAB_SIZE$}")?;
            write!(f, "{SPACE:TAB_SIZE$}")?;
            write_dim!(f, "{row:>index_width$}")?;
            write!(f, "{SPACE:OUTER_GAP$}")?;
            write!(f, "{LEFT_DELIMITER}")?;
            for col in 0..ncols {
                if col != 0 {
                    write!(f, "{SPACE:<INTER_GAP$}")?;
                }
                let index =
                    Self::flatten_index_unchecked(Index::new(row, col), self.order, self.shape);
                write_dim!(f, "{index:>index_width$}")?;
                write!(f, "{SPACE:INNER_GAP$}")?;
                match cache[index].next() {
                    None => write!(f, "{SPACE:<element_width$}")?,
                    Some(line) => write!(f, "{line:<element_width$}")?,
                }
            }
            writeln!(f, "{RIGHT_DELIMITER}")?;

            // remaining lines of the element representation
            for _ in 1..element_hight {
                write!(f, "{SPACE:TAB_SIZE$}{SPACE:TAB_SIZE$}")?;
                write!(f, "{SPACE:TAB_SIZE$}")?;
                write!(f, "{SPACE:>index_width$}")?;
                write!(f, "{SPACE:OUTER_GAP$}")?;
                write!(f, "{SPACE}")?;
                for col in 0..ncols {
                    if col != 0 {
                        write!(f, "{SPACE:<INTER_GAP$}")?;
                    }
                    let index =
                        Self::flatten_index_unchecked(Index::new(row, col), self.order, self.shape);
                    write!(f, "{SPACE:>index_width$}")?;
                    write!(f, "{SPACE:INNER_GAP$}")?;
                    match cache[index].next() {
                        None => write!(f, "{SPACE:<element_width$}")?,
                        Some(line) => write!(f, "{line:<element_width$}")?,
                    }
                }
                writeln!(f)?;
            }
        }

        writeln!(f, "{SPACE:TAB_SIZE$}{SPACE:TAB_SIZE$}{RIGHT_DELIMITER}")?;

        writeln!(f, "}}")
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let nrows = shape.nrows;
        let ncols = shape.ncols;
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

        writeln!(f, "{LEFT_DELIMITER}")?;

        for row in 0..nrows {
            // first line of the element representation
            write!(f, "{SPACE:TAB_SIZE$}")?;
            write!(f, "{LEFT_DELIMITER}")?;
            for col in 0..ncols {
                if col != 0 {
                    write!(f, "{SPACE:INTER_GAP$}")?;
                }
                let index =
                    Self::flatten_index_unchecked(Index::new(row, col), self.order, self.shape);
                match cache[index].next() {
                    None => write!(f, "{SPACE:<element_width$}")?,
                    Some(line) => write!(f, "{line:<element_width$}")?,
                }
            }
            writeln!(f, "{RIGHT_DELIMITER}")?;

            // remaining lines of the element representation
            for _ in 1..element_hight {
                write!(f, "{SPACE:TAB_SIZE$}")?;
                write!(f, "{SPACE}")?;
                for col in 0..ncols {
                    if col != 0 {
                        write!(f, "{SPACE:INTER_GAP$}")?;
                    }
                    let index =
                        Self::flatten_index_unchecked(Index::new(row, col), self.order, self.shape);
                    match cache[index].next() {
                        None => write!(f, "{SPACE:<element_width$}")?,
                        Some(line) => write!(f, "{line:<element_width$}")?,
                    }
                }
                writeln!(f)?;
            }
        }

        writeln!(f, "{RIGHT_DELIMITER}")
    }
}

struct Lines(std::collections::VecDeque<String>);

impl Lines {
    fn from_debug<T: std::fmt::Debug>(element: T) -> Self {
        Self(format!("{:?}", element).lines().map(String::from).collect())
    }

    fn from_display<T: std::fmt::Display>(element: T) -> Self {
        Self(format!("{}", element).lines().map(String::from).collect())
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
}

#[cfg(test)]
mod tests {
    use crate::matrix;

    struct Mock(usize);

    impl std::fmt::Debug for Mock {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            for i in 0..self.0 {
                writeln!(f, "{}", "+".repeat(i + 1))?;
            }
            Ok(())
        }
    }

    impl std::fmt::Display for Mock {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            for i in 0..self.0 {
                writeln!(f, "{}", "=".repeat(i + 1))?;
            }
            Ok(())
        }
    }

    // The expected strings are not formatted to multiple lines to prevent
    // the trailing whitespaces from being removed by the editor.

    #[test]
    fn test_debug() {
        let matrix = matrix![[Mock(0), Mock(1), Mock(2)], [Mock(3), Mock(4), Mock(5)]];
        let result = format!("{:?}", matrix);
        let expected = "Matrix {\n    order: RowMajor\n    shape: AxisShape { major: 2, minor: 3 }\n    data:\n        [       \u{1b}[2m0\u{1b}[22m        \u{1b}[2m1\u{1b}[22m        \u{1b}[2m2\u{1b}[22m      \n            \u{1b}[2m0\u{1b}[22m  [\u{1b}[2m0\u{1b}[22m        \u{1b}[2m1\u{1b}[22m +      \u{1b}[2m2\u{1b}[22m +    ]\n                                    ++   \n                                         \n                                         \n                                         \n            \u{1b}[2m1\u{1b}[22m  [\u{1b}[2m3\u{1b}[22m +      \u{1b}[2m4\u{1b}[22m +      \u{1b}[2m5\u{1b}[22m +    ]\n                  ++       ++       ++   \n                  +++      +++      +++  \n                           ++++     ++++ \n                                    +++++\n        ]\n}\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_display() {
        let matrix = matrix![[Mock(0), Mock(1), Mock(2)], [Mock(3), Mock(4), Mock(5)]];
        let result = format!("{}", matrix);
        let expected = "[\n    [       =      =    ]\n                   ==   \n                        \n                        \n                        \n    [=      =      =    ]\n     ==     ==     ==   \n     ===    ===    ===  \n            ====   ==== \n                   =====\n]\n";
        assert_eq!(result, expected);
    }
}
