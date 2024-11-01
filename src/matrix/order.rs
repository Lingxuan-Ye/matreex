/// Represents the memory order of a matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Order {
    #[default]
    RowMajor,
    ColMajor,
}

impl Order {
    /// Switches to the other memory order.
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
