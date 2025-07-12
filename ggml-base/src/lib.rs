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
    pub data_start: usize,
}

struct Position<'a, R> {
    read: &'a mut R,
    pos: usize,
}

impl<R: Read> Read for Position<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = R::read(self.read, buf)?;
        self.pos += count;
        Ok(count)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        R::read_exact(self.read, buf)?;
        self.pos += buf.len();
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
            Some(GgufValue::Uint32(a)) => *a as usize,
            _ => bail!("no general.alignment metadata"),
        };
        let padding = (alignment - read.pos % alignment) % alignment;

        Ok(GgufFile {
            metadata,
            tensors,
            data_start: read.pos + padding,
        })
    }
}

pub enum GgmlUnquantizedType {
    I8,
    I16,
    I32,
    I64,
    F64,
    F32,
}

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

pub struct GgmlTensorInfo {
    pub name: String,
    ty: u32,
    ty_name: &'static str,
    shape: Vec<u64>,
    nbytes: u64,
    offset: u64,
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

    fn traits_from_ggml(&self) -> Result<&'static sys::ggml_type_traits, Error> {
        ensure!(
            self.ty < sys::ggml_type_GGML_TYPE_COUNT,
            "ggml type={} is too large",
            self.ty
        );
        let traits = unsafe { sys::ggml_get_type_traits(self.ty) };
        ensure!(
            !traits.is_null(),
            "ggml has no information for type={}",
            self.ty
        );
        let traits: &'static _ = unsafe { &*traits };
        Ok(traits)
    }

    fn update_from_ggml(&mut self) -> Result<(), Error> {
        let traits = self.traits_from_ggml()?;
        let ty_name: &'static _ = unsafe { CStr::from_ptr(traits.type_name) };
        self.ty_name = ty_name.to_str()?;

        let blck_size: u64 = traits.blck_size as u64;
        let type_size: u64 = traits.type_size as u64;
        ensure!(blck_size > 0, self.ty_name);
        ensure!(type_size > 0, self.ty_name);

        let mut stride = unsafe { sys::ggml_type_size(self.ty) } as u64;
        let mut ne = self.shape.iter().rev().copied();
        stride = stride * ne.next().ok_or_else(|| anyhow!("empty shape"))? / blck_size;
        for ne in ne {
            stride = stride
                .checked_mul(ne)
                .ok_or_else(|| anyhow!("tensor size overflowed"))?;
        }
        self.nbytes = stride;
        Ok(())
    }

    pub fn unquantized(&self) -> Option<GgmlUnquantizedType> {
        use GgmlUnquantizedType::*;
        match self.ty {
            sys::ggml_type_GGML_TYPE_I8 => Some(I8),
            sys::ggml_type_GGML_TYPE_I16 => Some(I16),
            sys::ggml_type_GGML_TYPE_I32 => Some(I32),
            sys::ggml_type_GGML_TYPE_I64 => Some(I64),
            sys::ggml_type_GGML_TYPE_F32 => Some(F32),
            sys::ggml_type_GGML_TYPE_F64 => Some(F64),
            _ => None,
        }
    }

    pub fn start(&self) -> usize {
        self.offset as usize
    }

    pub fn end(&self) -> usize {
        (self.offset + self.nbytes) as usize
    }

    pub fn nbytes(&self) -> usize {
        self.nbytes as usize
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn nelements(&self) -> usize {
        self.shape.iter().copied().product::<u64>() as usize
    }

    pub fn type_name(&self) -> &'static str {
        self.ty_name
    }

    pub fn dequantize(&mut self, data: &[u16]) -> Option<Vec<f32>> {
        assert!(data.len() == self.nbytes() / 2);
        let k = self.nelements();
        let mut floats = vec![0f32; k];
        let dequantize = self.traits_from_ggml().unwrap().to_float?;
        unsafe { dequantize(data.as_ptr() as _, floats.as_mut_ptr() as _, k as i64) };
        Some(floats)
    }
}
