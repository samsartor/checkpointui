use anyhow::{Result, bail};
use safetensors::tensor::{Metadata, SafeTensorError, TensorInfo};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const HEADER_MIB_LIMIT: usize = 100;

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone)]
pub enum Key {
    Name(String),
    Index(u64),
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Key::Name(n) => fmt::Display::fmt(n, f),
            Key::Index(i) => fmt::Display::fmt(i, f),
        }
    }
}

#[derive(Default)]
pub struct ModuleInfo {
    pub full_name: String,
    pub tensor_info: Option<TensorInfo>,
    pub children: BTreeMap<Key, ModuleInfo>,
    pub params: usize,
}

impl ModuleInfo {
    pub fn new(full_name: String) -> Self {
        Self {
            full_name,
            tensor_info: None,
            children: BTreeMap::new(),
            params: 0,
        }
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
        let tree = build_tree(metadata.tensors())?;
        let flattened_tree = flatten_single_child_chains(tree);

        Ok(Self {
            metadata,
            header_size,
            tree: flattened_tree,
        })
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

fn build_tree(tensors: HashMap<String, &TensorInfo>) -> Result<ModuleInfo> {
    let mut root = ModuleInfo::new("".to_string());

    for (name, info) in tensors {
        let params = info.shape.iter().copied().product::<usize>();
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = &mut root;
        current.params += params;

        for (i, &part) in parts.iter().enumerate() {
            let key = match part.parse() {
                Ok(i) => Key::Index(i),
                Err(_) => Key::Name(part.to_string()),
            };
            current = current
                .children
                .entry(key)
                .or_insert_with(|| ModuleInfo::new(name.clone()));
            current.params += params;

            if i == parts.len() - 1 {
                current.tensor_info = Some(info.clone());
            }
        }
    }

    Ok(root)
}

fn flatten_single_child_chains(mut module: ModuleInfo) -> ModuleInfo {
    for (_, child) in module.children.iter_mut() {
        *child = flatten_single_child_chains(std::mem::take(child));
    }

    while module.children.len() == 1 && module.tensor_info.is_none() {
        let children = std::mem::take(&mut module.children);
        let (key, child) = children.into_iter().next().unwrap();
        if child.children.is_empty() {
            module.children.insert(key, child);
            break;
        }
        module = child;
    }

    module
}
