//! Describes the memory layout of a matrix.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An enum representing the memory layout of a [`Matrix<T>`].
///
/// [`Matrix<T>`]: crate::Matrix
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum Order {
    /// Elements are stored row by row, with consecutive elements of
    /// a row being stored contiguously in memory.
    #[default]
    RowMajor,

    /// Elements are stored column by column, with consecutive elements of
    /// a column being stored contiguously in memory.
    ColMajor,
}

impl Order {
    /// Switches the order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Order;
    ///
    /// let mut order = Order::RowMajor;
    /// order.switch();
    /// assert_eq!(order, Order::ColMajor);
    /// ```
    pub fn switch(&mut self) -> &mut Self {
        *self = match self {
            Self::RowMajor => Self::ColMajor,
            Self::ColMajor => Self::RowMajor,
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switch() {
        let mut order = Order::RowMajor;

        order.switch();
        assert_eq!(order, Order::ColMajor);

        order.switch();
        assert_eq!(order, Order::RowMajor);
    }
}
