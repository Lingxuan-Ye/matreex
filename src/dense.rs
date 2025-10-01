pub use self::layout::{ColMajor, RowMajor};

use self::layout::{Layout, Order};
use alloc::vec::Vec;

mod layout;

pub struct Matrix<T, O = RowMajor>
where
    O: Order,
{
    layout: Layout<T, O>,
    data: Vec<T>,
}
