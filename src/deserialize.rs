use crate::Matrix;
use crate::error::Error::SizeMismatch;
use crate::order::Order;
use crate::shape::AxisShape;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use serde::de::Error;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

const FIELDS: &[&str] = &["order", "shape", "data"];

impl<'de, T> Deserialize<'de> for Matrix<T>
where
    T: Deserialize<'de>,
{
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

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("struct Matrix")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let order: Order = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(0, &self))?;
        let shape: AxisShape = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(1, &self))?;
        let data: Vec<T> = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(2, &self))?;
        if shape.size() != data.len() {
            Err(Error::custom(SizeMismatch))
        } else {
            Ok(Matrix { order, shape, data })
        }
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut order: Option<Order> = None;
        let mut shape: Option<AxisShape> = None;
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

        if shape.size() != data.len() {
            Err(Error::custom(SizeMismatch))
        } else {
            Ok(Matrix { order, shape, data })
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

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
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
