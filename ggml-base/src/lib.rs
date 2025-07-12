use anyhow::{Error, anyhow, bail, ensure};
use byteorder::{ByteOrder, LE, ReadBytesExt};
use std::collections::HashMap;
use std::ffi::CStr;
use std::io::Read;

pub mod sys {
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

fn read_gguf_string<O: ByteOrder>(read: &mut impl Read) -> Result<String, Error> {
    let len = read.read_u64::<O>()?;
    let mut string = String::with_capacity(len as usize);
    read.take(len).read_to_string(&mut string)?;
    Ok(string)
}

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgmlTensorInfo>,
    pub data_start: u64,
}

struct Position<'a, R> {
    read: &'a mut R,
    pos: u64,
}

impl<R: Read> Read for Position<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = R::read(self.read, buf)?;
        self.pos += count as u64;
        Ok(count)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        R::read_exact(self.read, buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }
}

impl GgufFile {
    pub fn read(read: &mut impl Read) -> Result<GgufFile, Error> {
        Self::read_ordered::<LE>(read)
    }

    pub fn read_ordered<O: ByteOrder>(read: &mut impl Read) -> Result<GgufFile, Error> {
        let mut read = Position { read, pos: 0 };
        let mut header = [0u8; 4];
        read.read_exact(&mut header)?;
        ensure!(header == [b'G', b'G', b'U', b'F'], "not a gguf file");
        let version = read.read_u32::<O>()?;
        ensure!(version == 3, "not a version 3 gguf file");

        let tensor_count = read.read_u64::<O>()?;
        let kv_count = read.read_u64::<O>()?;
        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let k = read_gguf_string::<O>(&mut read)?;
            let v = GgufValue::read::<O>(&mut read)?;
            metadata.insert(k, v);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            tensors.push(GgmlTensorInfo::read::<O>(&mut read)?);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a as u64,
            _ => 32,
        };
        let padding = (alignment - read.pos % alignment) % alignment;

        Ok(GgufFile {
            metadata,
            tensors,
            data_start: read.pos + padding,
        })
    }
}

pub const I8: GgmlTypeId = sys::ggml_type_GGML_TYPE_I8;
pub const I16: GgmlTypeId = sys::ggml_type_GGML_TYPE_I16;
pub const I32: GgmlTypeId = sys::ggml_type_GGML_TYPE_I32;
pub const I64: GgmlTypeId = sys::ggml_type_GGML_TYPE_I64;
pub const F16: GgmlTypeId = sys::ggml_type_GGML_TYPE_F16;
pub const BF16: GgmlTypeId = sys::ggml_type_GGML_TYPE_BF16;
pub const F32: GgmlTypeId = sys::ggml_type_GGML_TYPE_F32;
pub const F64: GgmlTypeId = sys::ggml_type_GGML_TYPE_F64;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    fn read_ty<O: ByteOrder>(ty: u32, read: &mut impl Read) -> Result<GgufValue, Error> {
        use GgufValue::*;
        Ok(match ty {
            0 => Uint8(read.read_u8()?),
            1 => Int8(read.read_i8()?),
            2 => Uint16(read.read_u16::<O>()?),
            3 => Int16(read.read_i16::<O>()?),
            4 => Uint32(read.read_u32::<O>()?),
            5 => Int32(read.read_i32::<O>()?),
            6 => Float32(read.read_f32::<O>()?),
            7 => Bool(read.read_u8()? != 0),
            8 => String(read_gguf_string::<O>(read)?),
            9 => Array({
                let el_ty = read.read_u32::<O>()?;
                let len = read.read_u64::<O>()?;
                let mut vec = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    vec.push(Self::read_ty::<O>(el_ty, read)?);
                }
                vec
            }),
            10 => Uint64(read.read_u64::<O>()?),
            11 => Int64(read.read_i64::<O>()?),
            12 => Float64(read.read_f64::<O>()?),
            _ => bail!("unknown metadata type {ty}"),
        })
    }

    pub fn read<O: ByteOrder>(read: &mut impl Read) -> Result<GgufValue, Error> {
        Self::read_ty::<O>(read.read_u32::<O>()?, read)
    }
}

pub type GgmlTypeId = sys::ggml_type;

pub struct GgmlTensorInfo {
    pub name: String,
    pub ty: GgmlTypeId,
    pub ty_name: &'static str,
    pub shape: Vec<u64>,
    pub nbytes: usize,
    pub offset: u64,
}

impl GgmlTensorInfo {
    pub fn read<O: ByteOrder>(read: &mut impl Read) -> Result<Self, Error> {
        let name = read_gguf_string::<O>(read)?;
        let ndimensions = read.read_u32::<O>()?;
        let mut shape = Vec::with_capacity(4);
        for _ in 0..ndimensions {
            shape.push(read.read_u64::<O>()?);
        }
        shape.reverse();
        let ty = read.read_u32::<O>()?;
        let offset = read.read_u64::<O>()?;
        let mut this = GgmlTensorInfo {
            name,
            ty,
            ty_name: "",
            shape,
            nbytes: 0,
            offset,
        };
        this.update_from_ggml()?;
        Ok(this)
    }

    fn update_from_ggml(&mut self) -> Result<(), Error> {
        let (_traits, ty_name, nbytes) = get_type_and_size(self.ty, &self.shape)?;
        self.ty_name = ty_name;
        self.nbytes = nbytes;
        Ok(())
    }

    pub fn nelements(&self) -> usize {
        self.shape.iter().copied().product::<u64>() as usize
    }
}

fn get_type_traits(ty: GgmlTypeId) -> Option<&'static sys::ggml_type_traits> {
    if ty >= sys::ggml_type_GGML_TYPE_COUNT {
        return None;
    }
    let traits = unsafe { sys::ggml_get_type_traits(ty) };
    if traits.is_null() {
        return None;
    }
    let traits: &'static _ = unsafe { &*traits };
    Some(traits)
}

fn get_type_and_size(
    ty: GgmlTypeId,
    shape: &[u64],
) -> Result<(&'static sys::ggml_type_traits, &'static str, usize), Error> {
    let traits = get_type_traits(ty).ok_or_else(|| anyhow!("{} is not a a valid ggml type", ty))?;
    let ty_name: &'static _ = unsafe { CStr::from_ptr(traits.type_name) };
    let ty_name = ty_name.to_str()?;

    let blck_size: u64 = traits.blck_size as u64;
    let type_size: u64 = traits.type_size as u64;
    ensure!(blck_size > 0, ty_name);
    ensure!(type_size > 0, ty_name);

    let mut stride = unsafe { sys::ggml_type_size(ty) } as u64;
    let mut ne = shape.iter().rev().copied();
    stride = stride * ne.next().ok_or_else(|| anyhow!("empty shape"))? / blck_size;
    for ne in ne {
        stride = stride
            .checked_mul(ne)
            .ok_or_else(|| anyhow!("tensor size overflowed"))?;
    }

    Ok((traits, ty_name, stride.try_into()?))
}

pub fn get_type_name(ty: GgmlTypeId) -> Option<&'static str> {
    let traits = get_type_traits(ty)?;
    let ty_name: &'static _ = unsafe { CStr::from_ptr(traits.type_name) };
    ty_name.to_str().ok()
}

pub fn dequantize(ty: GgmlTypeId, shape: &[u64], bytes: &[u8]) -> Result<Vec<f32>, Error> {
    let (traits, ty_name, nbytes) = get_type_and_size(ty, shape)?;
    let nelements = shape.iter().copied().product::<u64>();
    if nelements == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        bytes.len() >= nbytes,
        "buffer has length {} (expected {})",
        bytes.len(),
        nbytes
    );
    let to_float = traits
        .to_float
        .ok_or_else(|| anyhow!("{ty_name} has no dequantization method"))?;
    let mut floats = vec![0f32; nelements as usize];
    unsafe { to_float(bytes.as_ptr() as _, floats.as_mut_ptr(), nelements as i64) };
    Ok(floats)
}

#[cfg(feature = "serde_json")]
impl From<GgufValue> for serde_json::Value {
    fn from(value: GgufValue) -> Self {
        use GgufValue::*;
        match value {
            Uint8(x) => x.into(),
            Int8(x) => x.into(),
            Uint16(x) => x.into(),
            Int16(x) => x.into(),
            Uint32(x) => x.into(),
            Int32(x) => x.into(),
            Float32(x) => x.into(),
            Uint64(x) => x.into(),
            Int64(x) => x.into(),
            Float64(x) => serde_json::Number::from_f64(x).unwrap().into(),
            Bool(x) => x.into(),
            String(x) => x.into(),
            GgufValue::Array(x) => x.into_iter().map(serde_json::Value::from).collect(),
        }
    }
}

#[cfg(feature = "serde_json")]
impl From<&'_ GgufValue> for serde_json::Value {
    fn from(value: &GgufValue) -> Self {
        use GgufValue::*;
        match *value {
            Uint8(x) => x.into(),
            Int8(x) => x.into(),
            Uint16(x) => x.into(),
            Int16(x) => x.into(),
            Uint32(x) => x.into(),
            Int32(x) => x.into(),
            Float32(x) => x.into(),
            Uint64(x) => x.into(),
            Int64(x) => x.into(),
            Float64(x) => serde_json::Number::from_f64(x).unwrap().into(),
            Bool(x) => x.into(),
            String(ref x) => x.clone().into(),
            GgufValue::Array(ref x) => x.iter().map(serde_json::Value::from).collect(),
        }
    }
}
