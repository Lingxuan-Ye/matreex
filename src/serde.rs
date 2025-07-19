use crate::Matrix;
use crate::error::Error::SizeMismatch;
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
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
        deserializer.deserialize_struct("Matrix", FIELDS, MatrixVisitor::new())
    }
}

#[derive(Debug)]
struct MatrixVisitor<T> {
    marker: PhantomData<T>,
}

impl<T> MatrixVisitor<T> {
    fn new() -> Self {
        let marker = PhantomData;
        Self { marker }
    }
}

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
        let order: Order = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(0, &self))?;
        let shape: Shape = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(1, &self))?;
        let data: Vec<T> = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(2, &self))?;

        let shape = AxisShape::from_shape(shape, order);
        match shape.size::<T>() {
            Ok(size) if data.len() == size => Ok(Matrix { order, shape, data }),
            _ => Err(Error::custom(SizeMismatch)),
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

        let order = order.ok_or_else(|| Error::missing_field("order"))?;
        let shape = shape.ok_or_else(|| Error::missing_field("shape"))?;
        let data = data.ok_or_else(|| Error::missing_field("data"))?;

        let shape = AxisShape::from_shape(shape, order);
        match shape.size::<T>() {
            Ok(size) if data.len() == size => Ok(Matrix { order, shape, data }),
            _ => Err(Error::custom(SizeMismatch)),
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

    fn visit_str<E>(self, value: &str) -> Result<Field, E>
    where
        E: Error,
    {
        match value {
            "order" => Ok(Field::Order),
            "shape" => Ok(Field::Shape),
            "data" => Ok(Field::Data),
            _ => Err(Error::unknown_field(value, FIELDS)),
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
    fn test_deserialize_struct() {
        {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
            matrix.set_order(Order::RowMajor);

            let mut tokens = vec![
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
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::StructEnd,
            ];
            assert_de_tokens(&matrix, &tokens);

            let index = tokens.iter().position(|&x| x == Token::I32(6)).unwrap();
            tokens.remove(index);
            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }

        {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
            matrix.set_order(Order::ColMajor);

            let mut tokens = vec![
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
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(4),
                Token::I32(2),
                Token::I32(5),
                Token::I32(3),
                Token::I32(6),
                Token::SeqEnd,
                Token::StructEnd,
            ];
            assert_de_tokens(&matrix, &tokens);

            let index = tokens.iter().position(|&x| x == Token::I32(6)).unwrap();
            tokens.remove(index);
            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }
    }

    #[test]
    fn test_deserialize_seq() {
        {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
            matrix.set_order(Order::RowMajor);

            let mut tokens = vec![
                Token::Seq { len: Some(3) },
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
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::SeqEnd,
            ];
            assert_de_tokens(&matrix, &tokens);

            let index = tokens.iter().position(|&x| x == Token::I32(6)).unwrap();
            tokens.remove(index);
            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }

        {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
            matrix.set_order(Order::ColMajor);

            let mut tokens = vec![
                Token::Seq { len: Some(3) },
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
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(4),
                Token::I32(2),
                Token::I32(5),
                Token::I32(3),
                Token::I32(6),
                Token::SeqEnd,
                Token::SeqEnd,
            ];
            assert_de_tokens(&matrix, &tokens);

            let index = tokens.iter().position(|&x| x == Token::I32(6)).unwrap();
            tokens.remove(index);
            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }
    }

    #[test]
    fn test_deserialize_map() {
        {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
            matrix.set_order(Order::RowMajor);

            let mut tokens = vec![
                Token::Map { len: Some(3) },
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
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::I32(4),
                Token::I32(5),
                Token::I32(6),
                Token::SeqEnd,
                Token::MapEnd,
            ];
            assert_de_tokens(&matrix, &tokens);

            let index = tokens.iter().position(|&x| x == Token::I32(6)).unwrap();
            tokens.remove(index);
            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }

        {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
            matrix.set_order(Order::ColMajor);

            let mut tokens = vec![
                Token::Map { len: Some(3) },
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
                Token::Seq { len: Some(6) },
                Token::I32(1),
                Token::I32(4),
                Token::I32(2),
                Token::I32(5),
                Token::I32(3),
                Token::I32(6),
                Token::SeqEnd,
                Token::MapEnd,
            ];
            assert_de_tokens(&matrix, &tokens);

            let index = tokens.iter().position(|&x| x == Token::I32(6)).unwrap();
            tokens.remove(index);
            assert_de_tokens_error::<Matrix<i32>>(&tokens, &SizeMismatch.to_string());
        }
    }
}
