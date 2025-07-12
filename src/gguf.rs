use crate::model::{LE, ModuleInfo, ModuleSource, PathSplit, TensorInfo, TensorSeek};
use anyhow::{Error, Result, anyhow};
use ggml_base::{GgmlTensorInfo, GgufFile};
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
    fn tensor_bytes(&mut self, seek: &TensorSeek) -> Result<Vec<u8>> {
        let &TensorSeek::InFile { start, end } = seek;
        self.io
            .seek(std::io::SeekFrom::Start(start + self.inner.data_start))?;
        let mut data = vec![
            0;
            end.checked_sub(start)
                .ok_or(anyhow!("tensor ends before start"))? as usize
        ];
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
                .map(|tensor| (tensor.name.clone(), TensorInfo::from(info))),
            split,
        ))
    }

    fn metadata(&mut self) -> Result<Value> {
        let mut map = serde_json::value::Map::new();
        for (k, v) in &self.inner.metadata {
            map.insert(k.clone(), v.into());
        }
        Ok(map.into())
    }

    fn tensor_f32(
        &mut self,
        tensor: TensorInfo,
        _cancel: Ref<()>,
    ) -> std::result::Result<Vec<f32>, Error> {
        tensor.read_f32::<LE>(&self.tensor_bytes(&tensor.seek)?)
    }

    fn tensor_f64(
        &mut self,
        tensor: TensorInfo,
        _cancel: Ref<()>,
    ) -> std::result::Result<Vec<f64>, Error> {
        tensor.read_f64::<LE>(&self.tensor_bytes(&tensor.seek)?)
    }
}

impl From<GgmlTensorInfo> for TensorInfo {
    fn from(value: GgmlTensorInfo) -> Self {
        TensorInfo {
            ty: (),
            shape: (),
            size: (),
            seek: (),
        }
    }
}
