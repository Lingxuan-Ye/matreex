use crate::Matrix;
use crate::error::Error::SizeMismatch;
use crate::order::Order;
use crate::shape::{MemoryShape, Shape};
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use serde::de::{Deserialize, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

const FIELDS: &[&str] = &["order", "shape", "data"];

impl<T> Serialize for Matrix<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut matrix = serializer.serialize_struct("Matrix", 3)?;
        matrix.serialize_field("order", &self.order)?;
        matrix.serialize_field("shape", &self.shape())?;
        matrix.serialize_field("data", &self.data)?;
        matrix.end()
    }
}

impl<'de, T> Deserialize<'de> for Matrix<T>
where
    T: Deserialize<'de>,
{
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("Matrix", FIELDS, MatrixVisitor(PhantomData))
    }
}

#[derive(Debug)]
struct MatrixVisitor<T>(PhantomData<T>);

impl<'de, T> Visitor<'de> for MatrixVisitor<T>
where
    T: Deserialize<'de>,
{
    type Value = Matrix<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("struct Matrix")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let order: Order;
        let shape: Shape;
        let data: Vec<T>;

        if seq.size_hint() == Some(2) {
            order = Order::default();
            shape = seq
                .next_element()?
                .ok_or_else(|| Error::invalid_length(0, &self))?;
            data = seq
                .next_element()?
                .ok_or_else(|| Error::invalid_length(1, &self))?;
        } else {
            order = seq
                .next_element()?
                .ok_or_else(|| Error::invalid_length(0, &self))?;
            shape = seq
                .next_element()?
                .ok_or_else(|| Error::invalid_length(1, &self))?;
            data = seq
                .next_element()?
                .ok_or_else(|| Error::invalid_length(2, &self))?;
        }

        let shape = MemoryShape::from_shape(shape, order);
        match shape.size::<T>() {
            Ok(size) if data.len() == size => Ok(Self::Value { order, shape, data }),
            _ => Err(A::Error::custom(SizeMismatch)),
        }
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut order: Option<Order> = None;
        let mut shape: Option<Shape> = None;
        let mut data: Option<Vec<T>> = None;

        while let Some(key) = map.next_key()? {
            match key {
                Field::Order => {
                    if order.is_some() {
                        return Err(Error::duplicate_field("order"));
                    }
                    order = Some(map.next_value()?);
                }
                Field::Shape => {
                    if shape.is_some() {
                        return Err(Error::duplicate_field("shape"));
                    }
                    shape = Some(map.next_value()?);
                }
                Field::Data => {
                    if data.is_some() {
                        return Err(Error::duplicate_field("data"));
                    }
                    data = Some(map.next_value()?);
                }
            }
        }

        let order = order.unwrap_or_default();
        let shape = shape.ok_or_else(|| Error::missing_field("shape"))?;
        let data = data.ok_or_else(|| Error::missing_field("data"))?;

        let shape = MemoryShape::from_shape(shape, order);
        match shape.size::<T>() {
            Ok(size) if data.len() == size => Ok(Self::Value { order, shape, data }),
            _ => Err(A::Error::custom(SizeMismatch)),
        }
    }
}

#[derive(Debug)]
enum Field {
    Order,
    Shape,
    Data,
}

impl<'de> Deserialize<'de> for Field {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_identifier(FieldVisitor)
    }
}

#[derive(Debug)]
struct FieldVisitor;

impl Visitor<'_> for FieldVisitor {
    type Value = Field;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("`order`, `shape` or `data`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        match value {
            "order" => Ok(Self::Value::Order),
            "shape" => Ok(Self::Value::Shape),
            "data" => Ok(Self::Value::Data),
            _ => Err(E::unknown_field(value, FIELDS)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;
    use alloc::string::ToString;
    use alloc::vec;
    use serde_test::{Token, assert_de_tokens, assert_de_tokens_error};

    #[test]
    fn test_deserialize_seq() {
        {
            let tokens = vec![
                Token::Seq { len: None },
                Token::UnitVariant {
                    name: "Order",
                    variant: "RowMajor",
                },
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::SeqEnd,
            ];

            let mut expected = matrix![[1, 2, 3], [4, 5, 6]];
            expected.set_order(Order::RowMajor);
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Seq { len: None },
                Token::UnitVariant {
                    name: "Order",
                    variant: "ColMajor",
                },
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(4),
                Token::I32(2),
                Token::I32(5),
                Token::I32(3),
                Token::I32(6),
                Token::SeqEnd,
                Token::SeqEnd,
            ];

            let mut expected = matrix![[1, 2, 3], [4, 5, 6]];
            expected.set_order(Order::ColMajor);
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Seq { len: Some(2) },
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::SeqEnd,
            ];

            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Seq { len: Some(2) },
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::SeqEnd,
                Token::SeqEnd,
            ];

            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }
    }

    #[test]
    fn test_deserialize_map() {
        {
            let tokens = vec![
                Token::Map { len: None },
                Token::Str("order"),
                Token::UnitVariant {
                    name: "Order",
                    variant: "RowMajor",
                },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::MapEnd,
            ];

            let mut expected = matrix![[1, 2, 3], [4, 5, 6]];
            expected.set_order(Order::RowMajor);
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Map { len: None },
                Token::Str("order"),
                Token::UnitVariant {
                    name: "Order",
                    variant: "ColMajor",
                },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(4),
                Token::I32(2),
                Token::I32(5),
                Token::I32(3),
                Token::I32(6),
                Token::SeqEnd,
                Token::MapEnd,
            ];

            let mut expected = matrix![[1, 2, 3], [4, 5, 6]];
            expected.set_order(Order::ColMajor);
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Map { len: None },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::MapEnd,
            ];

            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Map { len: None },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::SeqEnd,
                Token::MapEnd,
            ];

            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }
    }

    #[test]
    fn test_deserialize_struct() {
        {
            let tokens = vec![
                Token::Struct {
                    name: "Matrix",
                    len: 3,
                },
                Token::Str("order"),
                Token::UnitVariant {
                    name: "Order",
                    variant: "RowMajor",
                },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::StructEnd,
            ];

            let mut expected = matrix![[1, 2, 3], [4, 5, 6]];
            expected.set_order(Order::RowMajor);
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Struct {
                    name: "Matrix",
                    len: 3,
                },
                Token::Str("order"),
                Token::UnitVariant {
                    name: "Order",
                    variant: "ColMajor",
                },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(4),
                Token::I32(2),
                Token::I32(5),
                Token::I32(3),
                Token::I32(6),
                Token::SeqEnd,
                Token::StructEnd,
            ];

            let mut expected = matrix![[1, 2, 3], [4, 5, 6]];
            expected.set_order(Order::ColMajor);
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Struct {
                    name: "Matrix",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::StructEnd,
            ];

            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_de_tokens(&expected, &tokens);
        }

        {
            let tokens = vec![
                Token::Struct {
                    name: "Matrix",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Struct {
                    name: "Shape",
                    len: 2,
                },
                Token::Str("nrows"),
                Token::U64(2),
                Token::Str("ncols"),
                Token::U64(3),
                Token::StructEnd,
                Token::Str("data"),
                Token::Seq { len: None },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::SeqEnd,
                Token::StructEnd,
            ];

            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }
    }
}
