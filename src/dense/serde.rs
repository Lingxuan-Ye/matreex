use super::Matrix;
use super::layout::{Layout, Order};
use crate::error::Error::SizeMismatch;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use serde::de::{Deserialize, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

const FIELDS: &[&str] = &["layout", "data"];

impl<T, O> Serialize for Matrix<T, O>
where
    T: Serialize,
    O: Order,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut matrix = serializer.serialize_struct("Matrix", 2)?;
        matrix.serialize_field("layout", &self.layout)?;
        matrix.serialize_field("data", &self.data)?;
        matrix.end()
    }
}

impl<'de, T, O> Deserialize<'de> for Matrix<T, O>
where
    T: Deserialize<'de>,
    O: Order,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("Matrix", FIELDS, MatrixVisitor(PhantomData))
    }
}

#[derive(Debug)]
struct MatrixVisitor<T, O>(PhantomData<(T, O)>);

impl<'de, T, O> Visitor<'de> for MatrixVisitor<T, O>
where
    T: Deserialize<'de>,
    O: Order,
{
    type Value = Matrix<T, O>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("struct Matrix")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let layout: Layout<T, O> = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(0, &self))?;
        let data: Vec<T> = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(1, &self))?;

        if layout.size() != data.len() {
            Err(Error::custom(SizeMismatch))
        } else {
            Ok(Matrix { layout, data })
        }
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut layout: Option<Layout<T, O>> = None;
        let mut data: Option<Vec<T>> = None;

        while let Some(key) = map.next_key()? {
            match key {
                Field::Layout => {
                    if layout.is_some() {
                        return Err(Error::duplicate_field("layout"));
                    }
                    layout = Some(map.next_value()?);
                }
                Field::Data => {
                    if data.is_some() {
                        return Err(Error::duplicate_field("data"));
                    }
                    data = Some(map.next_value()?);
                }
            }
        }

        let layout = layout.ok_or_else(|| Error::missing_field("layout"))?;
        let data = data.ok_or_else(|| Error::missing_field("data"))?;

        if layout.size() != data.len() {
            Err(Error::custom(SizeMismatch))
        } else {
            Ok(Matrix { layout, data })
        }
    }
}

#[derive(Debug)]
enum Field {
    Layout,
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
        formatter.write_str("`layout` or `data`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        match value {
            "layout" => Ok(Field::Layout),
            "data" => Ok(Field::Data),
            _ => Err(Error::unknown_field(value, FIELDS)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::layout::{ColMajor, RowMajor};
    use crate::matrix;
    use alloc::string::ToString;
    use serde_test::{Token, assert_de_tokens, assert_de_tokens_error, assert_ser_tokens};

    #[test]
    fn test_serialize() {
        let rmatrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let cmatrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<ColMajor>();
        let tokens = [
            Token::Struct {
                name: "Matrix",
                len: 2,
            },
            Token::Str("layout"),
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
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
        assert_ser_tokens(&rmatrix, &tokens);
        assert_ser_tokens(&cmatrix, &tokens);
    }

    #[test]
    fn test_deserialize_seq() {
        let tokens = [
            Token::Seq { len: None },
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
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
        let rexpected = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let cexpected = matrix![[1, 4], [2, 5], [3, 6]].with_order::<ColMajor>();
        assert_de_tokens(&rexpected, &tokens);
        assert_de_tokens(&cexpected, &tokens);

        let tokens = [
            Token::Seq { len: None },
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
            Token::U64(3),
            Token::StructEnd,
            Token::Seq { len: None },
            Token::SeqEnd,
            Token::SeqEnd,
        ];
        assert_de_tokens_error::<Matrix<i32, RowMajor>>(&tokens, &SizeMismatch.to_string());
        assert_de_tokens_error::<Matrix<i32, ColMajor>>(&tokens, &SizeMismatch.to_string());
    }

    #[test]
    fn test_deserialize_map() {
        let tokens = [
            Token::Map { len: None },
            Token::Str("layout"),
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
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
        let rexpected = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let cexpected = matrix![[1, 4], [2, 5], [3, 6]].with_order::<ColMajor>();
        assert_de_tokens(&rexpected, &tokens);
        assert_de_tokens(&cexpected, &tokens);

        let tokens = [
            Token::Map { len: None },
            Token::Str("layout"),
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
            Token::U64(3),
            Token::StructEnd,
            Token::Str("data"),
            Token::Seq { len: None },
            Token::SeqEnd,
            Token::MapEnd,
        ];
        assert_de_tokens_error::<Matrix<i32, RowMajor>>(&tokens, &SizeMismatch.to_string());
        assert_de_tokens_error::<Matrix<i32, ColMajor>>(&tokens, &SizeMismatch.to_string());
    }

    #[test]
    fn test_deserialize_struct() {
        let tokens = [
            Token::Struct {
                name: "Matrix",
                len: 2,
            },
            Token::Str("layout"),
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
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
        let rexpected = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let cexpected = matrix![[1, 4], [2, 5], [3, 6]].with_order::<ColMajor>();
        assert_de_tokens(&rexpected, &tokens);
        assert_de_tokens(&cexpected, &tokens);

        let tokens = [
            Token::Struct {
                name: "Matrix",
                len: 2,
            },
            Token::Str("layout"),
            Token::Struct {
                name: "Layout",
                len: 2,
            },
            Token::Str("major"),
            Token::U64(2),
            Token::Str("minor"),
            Token::U64(3),
            Token::StructEnd,
            Token::Str("data"),
            Token::Seq { len: None },
            Token::SeqEnd,
            Token::StructEnd,
        ];
        assert_de_tokens_error::<Matrix<i32, RowMajor>>(&tokens, &SizeMismatch.to_string());
        assert_de_tokens_error::<Matrix<i32, ColMajor>>(&tokens, &SizeMismatch.to_string());
    }
}
