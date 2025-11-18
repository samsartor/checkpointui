use crate::model::{LE, ModuleInfo, ModuleSource, PathSplit, TensorInfo, TensorTy};
use crate::storage::Storage;
use anyhow::{Error, Result, bail};
use safetensors::{SafeTensorError, tensor::Metadata};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::mem::size_of;
use std::path::Path;
use weakref::Ref;

pub struct Safetensors<S> {
    storage: S,
    data_offset: u64,
    metadata: Metadata,
}

impl<S: Storage> Safetensors<S> {
    pub fn open(mut storage: S) -> Result<Self> {
        let path = storage.display();
        let (metadata, data_offset) = read_metadata(storage.reader()?, &path)?;
        let data_offset = data_offset as u64;
        Ok(Safetensors {
            storage,
            data_offset,
            metadata,
        })
    }

    fn tensor_bytes(&mut self, start: u64, nbytes: usize) -> Result<Vec<u8>> {
        let mut r = self.storage.reader()?;
        r.seek(std::io::SeekFrom::Start(start + self.data_offset))?;
        let mut data = vec![0; nbytes];
        r.read_exact(&mut data)?;
        Ok(data)
    }
}

unsafe impl<I: Storage> Send for Safetensors<I> where I: Send {}

fn flatten_value(path: String, value: &Value, map: &mut HashMap<String, String>) {
    match value {
        Value::Null => {
            map.insert(path, "null".into());
        }
        Value::Bool(true) => {
            map.insert(path, "true".into());
        }
        Value::Bool(false) => {
            map.insert(path, "false".into());
        }
        Value::Number(number) => {
            map.insert(path, number.to_string());
        }
        Value::String(string) => {
            map.insert(path, string.clone());
        }
        Value::Array(array) => {
            for (k, v) in array.iter().enumerate() {
                flatten_value(format!("{path}.{k}"), v, map);
            }
        }
        Value::Object(object) => {
            for (k, v) in object.iter() {
                flatten_value(format!("{path}.{k}"), v, map);
            }
        }
    }
}

impl<S: Storage> ModuleSource for Safetensors<S> {
    fn module(&mut self, split: &PathSplit) -> Result<ModuleInfo> {
        let tensors = self.metadata.tensors();
        Ok(ModuleInfo::build_from_tensors(
            tensors
                .iter()
                .map(|(name, &info)| (name.clone(), info.into())),
            split,
        ))
    }

    fn metadata(&mut self) -> Result<Value> {
        let mut map = serde_json::value::Map::new();
        if let Some(meta) = self.metadata.metadata() {
            for (k, v) in meta {
                map.insert(k.clone(), v.as_str().into());
            }
        }
        Ok(map.into())
    }

    fn write_metadata(&mut self, metadata: Value) -> std::result::Result<(), Error> {
        let mut new_metadata = HashMap::new();
        flatten_value("".into(), &metadata, &mut new_metadata);
        let mut tensors: Vec<_> = self
            .metadata
            .tensors()
            .into_iter()
            .map(|(k, v)| (k, v.clone()))
            .collect();
        // the safetensors crate needlessly scrambles the order
        tensors.sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));
        let new_metadata = Metadata::new(Some(new_metadata), tensors)?;
        let mut new_header = serde_json::ser::to_vec(&new_metadata)?;
        let n = new_header.len() as u64;
        new_header.splice(0..0, u64::to_le_bytes(n));
        self.storage
            .splice(0..self.data_offset as usize, &new_header)?;
        self.data_offset = n + 8;
        self.metadata = new_metadata;
        Ok(())
    }

    fn tensor_f32(
        &mut self,
        tensor: TensorInfo,
        _cancel: Ref<()>,
    ) -> std::result::Result<Vec<f32>, Error> {
        tensor.read_f32::<LE>(&self.tensor_bytes(tensor.offset, tensor.size as usize)?)
    }

    fn tensor_f64(
        &mut self,
        tensor: TensorInfo,
        _cancel: Ref<()>,
    ) -> std::result::Result<Vec<f64>, Error> {
        tensor.read_f64::<LE>(&self.tensor_bytes(tensor.offset, tensor.size as usize)?)
    }
}

impl From<safetensors::Dtype> for TensorTy {
    fn from(value: safetensors::Dtype) -> Self {
        use TensorTy::*;
        use safetensors::Dtype;
        match value {
            Dtype::BOOL => BOOL,
            Dtype::U8 => U8,
            Dtype::I8 => I8,
            Dtype::F8_E5M2 => F8_E5M2,
            Dtype::F8_E4M3 => F8_E4M3,
            Dtype::I16 => I16,
            Dtype::U16 => U16,
            Dtype::F16 => F16,
            Dtype::BF16 => BF16,
            Dtype::F32 => F32,
            Dtype::F64 => F64,
            Dtype::I64 => I64,
            Dtype::U64 => U64,
            _ => Unknown(format!("{:?}", value)),
        }
    }
}

impl From<&'_ safetensors::tensor::TensorInfo> for TensorInfo {
    fn from(value: &'_ safetensors::tensor::TensorInfo) -> TensorInfo {
        TensorInfo {
            ty: value.dtype.into(),
            shape: value.shape.iter().map(|&x| x as u64).collect(),
            size: value.data_offsets.1.saturating_sub(value.data_offsets.0),
            offset: value.data_offsets.0 as u64,
        }
    }
}

const HEADER_MIB_LIMIT: usize = 100;

fn read_metadata<I: Read>(io: &mut I, path: &str) -> Result<(Metadata, usize), Error> {
    let mut header_size_bytes = [0u8; 8];
    io.read_exact(&mut header_size_bytes)?;
    let n = u64::from_le_bytes(header_size_bytes) as usize;

    if n > HEADER_MIB_LIMIT * 1024 * 1024 {
        bail!("Header is larger than {HEADER_MIB_LIMIT}MiB. Is {path} a safetensors file?",);
    }

    let mut metadata_bytes = vec![0u8; n];
    io.read_exact(&mut metadata_bytes)?;

    let metadata_str =
        std::str::from_utf8(&metadata_bytes).map_err(|err| SafeTensorError::InvalidHeader(err))?;

    let metadata: Metadata = serde_json::from_str(metadata_str)
        .map_err(|err| SafeTensorError::InvalidHeaderDeserialization(err))?;

    Ok((metadata, n + 8))
}
