use crate::model::{LE, ModuleInfo, ModuleSource, PathSplit, TensorInfo, TensorTy};
use anyhow::{Error, Result};
use ggml_base::{GgmlTensorInfo, GgufFile, GgufValue};
use serde_json::Value;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use weakref::Ref;

pub struct Gguf<I> {
    io: I,
    inner: GgufFile,
}

impl Gguf<File> {
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut reader = BufReader::new(File::open(&path)?);
        let inner = GgufFile::read(&mut reader)?;
        let io = reader.into_inner();
        Ok(Gguf { io, inner })
    }
}

impl<I: Read + Seek> Gguf<I> {
    fn tensor_bytes(&mut self, offset: u64, nbytes: usize) -> Result<Vec<u8>> {
        self.io
            .seek(std::io::SeekFrom::Start(offset + self.inner.data_start))?;
        let mut data = vec![0; nbytes];
        self.io.read_exact(&mut data)?;
        Ok(data)
    }
}

unsafe impl<I: Read + Seek> Send for Gguf<I> where I: Send {}

impl<I: Read + Seek> ModuleSource for Gguf<I> {
    fn module(&mut self, split: &PathSplit) -> Result<ModuleInfo> {
        let tensors = &self.inner.tensors;
        Ok(ModuleInfo::build_from_tensors(
            tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), TensorInfo::from(tensor))),
            split,
        ))
    }

    fn metadata(&mut self) -> Result<Value> {
        let mut map = serde_json::value::Map::new();
        for (k, v) in &self.inner.metadata {
            match v {
                // TODO: find a way to show that we truncated
                GgufValue::Array(arr) if arr.len() > 100 => continue,
                _ => (),
            }
            map.insert(k.clone(), v.into());
        }
        Ok(map.into())
    }

    fn tensor_f32(
        &mut self,
        tensor: TensorInfo,
        _cancel: Ref<()>,
    ) -> std::result::Result<Vec<f32>, Error> {
        tensor.read_f32::<LE>(&self.tensor_bytes(tensor.offset, tensor.size)?)
    }

    fn tensor_f64(
        &mut self,
        tensor: TensorInfo,
        _cancel: Ref<()>,
    ) -> std::result::Result<Vec<f64>, Error> {
        tensor.read_f64::<LE>(&self.tensor_bytes(tensor.offset, tensor.size)?)
    }
}

impl From<&'_ GgmlTensorInfo> for TensorInfo {
    fn from(value: &GgmlTensorInfo) -> Self {
        TensorInfo {
            ty: match value.ty {
                ggml_base::I8 => TensorTy::I8,
                ggml_base::I16 => TensorTy::I16,
                ggml_base::I32 => TensorTy::I32,
                ggml_base::I64 => TensorTy::I64,
                ggml_base::F16 => TensorTy::F16,
                ggml_base::BF16 => TensorTy::BF16,
                ggml_base::F32 => TensorTy::F32,
                ggml_base::F64 => TensorTy::F64,
                _ => TensorTy::Ggml(value.ty),
            },
            shape: value.shape.clone(),
            size: value.nbytes,
            offset: value.offset,
        }
    }
}
