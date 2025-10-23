use super::{Layout, Order};
use core::fmt;
use core::marker::PhantomData;
use serde::de::{Deserialize, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

const FIELDS: &[&str] = &["major", "minor"];

impl<T, O> Serialize for Layout<T, O>
where
    O: Order,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut layout = serializer.serialize_struct("Layout", 2)?;
        layout.serialize_field("major", &self.major)?;
        layout.serialize_field("minor", &self.minor)?;
        layout.end()
    }
}

impl<'de, T, O> Deserialize<'de> for Layout<T, O>
where
    O: Order,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("Layout", FIELDS, LayoutVisitor(PhantomData))
    }
}

#[derive(Debug)]
struct LayoutVisitor<T, O>(PhantomData<(T, O)>);

impl<'de, T, O> Visitor<'de> for LayoutVisitor<T, O>
where
    O: Order,
{
    type Value = Layout<T, O>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("struct Layout")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let major: usize = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(0, &self))?;
        let minor: usize = seq
            .next_element()?
            .ok_or_else(|| Error::invalid_length(1, &self))?;

        Layout::new_with_size(major, minor)
            .map(|(layout, _)| layout)
            .map_err(A::Error::custom)
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut major: Option<usize> = None;
        let mut minor: Option<usize> = None;

        while let Some(key) = map.next_key()? {
            match key {
                Field::Major => {
                    if major.is_some() {
                        return Err(Error::duplicate_field("major"));
                    }
                    major = Some(map.next_value()?);
                }
                Field::Minor => {
                    if minor.is_some() {
                        return Err(Error::duplicate_field("minor"));
                    }
                    minor = Some(map.next_value()?);
                }
            }
        }

        let major = major.ok_or_else(|| Error::missing_field("major"))?;
        let minor = minor.ok_or_else(|| Error::missing_field("minor"))?;

        Layout::new_with_size(major, minor)
            .map(|(layout, _)| layout)
            .map_err(A::Error::custom)
    }
}

#[derive(Debug)]
enum Field {
    Major,
    Minor,
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
        formatter.write_str("`major` or `minor`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        match value {
            "major" => Ok(Self::Value::Major),
            "minor" => Ok(Self::Value::Minor),
            _ => Err(E::unknown_field(value, FIELDS)),
        }
    }
}
