use crate::model::{LE, ModuleInfo, ModuleSource, PathSplit, TensorInfo, TensorSeek, TensorTy};
use anyhow::{Error, Result, anyhow, bail};
use safetensors::{SafeTensorError, tensor::Metadata};
use serde_json::Value;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

pub struct Safetensors<I> {
    io: I,
    metadata: Metadata,
}

impl Safetensors<File> {
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut io = File::open(&path)?;
        let metadata = read_metadata(&mut io, &path)?;
        Ok(Safetensors { io, metadata })
    }
}

impl<I: Read + Seek> Safetensors<I> {
    fn tensor_bytes(&mut self, seek: &TensorSeek) -> Result<Vec<u8>> {
        let &TensorSeek::InFile { start, end } = seek;
        self.io.seek(std::io::SeekFrom::Start(start))?;
        let mut data = vec![
            0;
            end.checked_sub(start)
                .ok_or(anyhow!("tensor ends before start"))? as usize
        ];
        self.io.read_exact(&mut data)?;
        Ok(data)
    }
}

impl<I: Read + Seek> ModuleSource for Safetensors<I> {
    fn module(&mut self, split: &PathSplit) -> Result<ModuleInfo> {
        let tensors = self.metadata.tensors();
        Ok(ModuleInfo::build_from_tensors(
            tensors
                .iter()
                .map(|(name, &info)| (name.clone(), TensorInfo::from(info))),
            split,
        ))
    }

    fn metadata(&mut self) -> Result<Value> {
        let mut map = serde_json::value::Map::new();
        if let Some(meta) = self.metadata.metadata() {
            for (k, v) in meta {
                map.insert(
                    k.clone(),
                    match json5::from_str(v) {
                        Ok(v) => v,
                        Err(_) => Value::String(v.clone()),
                    },
                );
            }
        }
        Ok(map.into())
    }

    fn tensor_f32(&mut self, tensor: TensorInfo) -> std::result::Result<Vec<f32>, Error> {
        tensor.read_f32::<LE>(&self.tensor_bytes(&tensor.seek)?)
    }

    fn tensor_f64(&mut self, tensor: TensorInfo) -> std::result::Result<Vec<f64>, Error> {
        tensor.read_f64::<LE>(&self.tensor_bytes(&tensor.seek)?)
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
    fn from(value: &safetensors::tensor::TensorInfo) -> Self {
        TensorInfo {
            ty: value.dtype.into(),
            shape: value.shape.iter().map(|&x| x as u64).collect(),
            size: value.data_offsets.1.saturating_sub(value.data_offsets.0) as u64,
            seek: TensorSeek::InFile {
                start: value.data_offsets.0 as u64,
                end: value.data_offsets.1 as u64,
            },
        }
    }
}

const HEADER_MIB_LIMIT: usize = 100;

fn read_metadata<I: Read>(io: &mut I, path: &Path) -> Result<Metadata, Error> {
    let mut header_size_bytes = [0u8; 8];
    io.read_exact(&mut header_size_bytes)?;
    let n = u64::from_le_bytes(header_size_bytes) as usize;

    if n > HEADER_MIB_LIMIT * 1024 * 1024 {
        bail!(
            "Header is larger than {HEADER_MIB_LIMIT}MiB. Is {} a safetensors file?",
            path.display()
        );
    }

    let mut metadata_bytes = vec![0u8; n];
    io.read_exact(&mut metadata_bytes)?;

    let metadata_str =
        std::str::from_utf8(&metadata_bytes).map_err(|_| SafeTensorError::InvalidHeader)?;

    let metadata: Metadata = serde_json::from_str(metadata_str)
        .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;

    Ok(metadata)
}
