use anyhow::{Result, bail};
use safetensors::tensor::{Metadata, SafeTensorError, TensorInfo};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::{fmt, mem};

const HEADER_MIB_LIMIT: usize = 100;

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone, Hash)]
pub enum Key {
    Name(String),
    Index(u64),
    Cons(Box<Key>, Box<Key>),
}

impl Key {}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Key::Name(n) => fmt::Display::fmt(n, f),
            Key::Index(i) => fmt::Display::fmt(i, f),
            Key::Cons(a, b) => write!(f, "{}.{}", a, b),
        }
    }
}

#[derive(Default)]
pub struct ModuleInfo {
    pub full_name: String,
    pub tensor_info: Option<TensorInfo>,
    pub children: BTreeMap<Key, ModuleInfo>,
    pub total_tensors: usize,
    pub total_params: usize,
}

impl ModuleInfo {
    pub fn new(full_name: String) -> Self {
        Self {
            full_name,
            tensor_info: None,
            children: BTreeMap::new(),
            total_tensors: 0,
            total_params: 0,
        }
    }

    pub fn build(tensors: HashMap<String, &TensorInfo>) -> Result<Self> {
        let mut root = ModuleInfo::new("".to_string());

        for (name, info) in tensors {
            let params = info.shape.iter().copied().product::<usize>();
            let parts: Vec<&str> = name.split('.').collect();
            let mut current = &mut root;
            current.total_params += params;
            current.total_tensors += 1;

            for (i, &part) in parts.iter().enumerate() {
                let key = match part.parse() {
                    Ok(i) => Key::Index(i),
                    Err(_) => Key::Name(part.to_string()),
                };
                current = current
                    .children
                    .entry(key)
                    .or_insert_with(|| ModuleInfo::new(name.clone()));
                current.total_params += params;
                current.total_tensors += 1;

                if i == parts.len() - 1 {
                    current.tensor_info = Some(info.clone());
                }
            }
        }

        Ok(root)
    }

    pub fn flatten_single_children(&mut self) {
        self.children = mem::take(&mut self.children)
            .into_iter()
            .map(|(k, mut v)| {
                v.flatten_single_children();
                if v.children.len() != 1 {
                    return (k, v);
                }
                let (ck, cv) = v.children.into_iter().next().unwrap();
                (Key::Cons(Box::new(k), Box::new(ck)), cv)
            })
            .collect();
    }
}

pub struct SafeTensorsData {
    pub metadata: Metadata,
    pub header_size: usize,
    pub tree: ModuleInfo,
}

impl SafeTensorsData {
    pub fn from_file(file_path: &Path) -> Result<Self> {
        let (header_size, metadata) = read_metadata_from_file(file_path)?;
        let mut tree = ModuleInfo::build(metadata.tensors())?;
        tree.flatten_single_children();

        Ok(Self {
            metadata,
            header_size,
            tree,
        })
    }

    pub fn parse_metadata(&self) -> Value {
        let mut map = serde_json::value::Map::new();
        if let Some(meta) = self.metadata.metadata() {
            for (k, v) in meta {
                map.insert(
                    k.clone(),
                    match serde_json::from_str(v) {
                        Ok(v) => v,
                        Err(_) => Value::String(v.clone()),
                    },
                );
            }
        }
        map.into()
    }
}

pub fn shorten_value(value: &mut Value) {
    use Value::*;
    match value {
        String(text) if text.len() > 10_000 || text.starts_with("data:image/") => {
            *text = "...".to_string();
        }
        Array(values) => {
            for value in values {
                shorten_value(value);
            }
        }
        Object(map) => {
            for value in map.values_mut() {
                shorten_value(value);
            }
        }
        _ => (),
    }
}

fn read_metadata_from_file(file_path: &Path) -> Result<(usize, Metadata)> {
    let mut file = File::open(file_path)?;

    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)?;
    let n = u64::from_le_bytes(header_size_bytes) as usize;

    if n > HEADER_MIB_LIMIT * 1024 * 1024 {
        bail!(
            "Header is larger than {HEADER_MIB_LIMIT}MiB. Is {} a safetensors file?",
            file_path.display()
        );
    }

    let mut metadata_bytes = vec![0u8; n];
    file.read_exact(&mut metadata_bytes)?;

    let metadata_str =
        std::str::from_utf8(&metadata_bytes).map_err(|_| SafeTensorError::InvalidHeader)?;

    let metadata: Metadata = serde_json::from_str(metadata_str)
        .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;

    Ok((n, metadata))
}
