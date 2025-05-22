use safetensors::tensor::TensorInfo;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
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

pub struct ModuleInfo {
    pub full_name: String,
    pub tensor_info: Option<TensorInfo>,
    pub children: BTreeMap<Key, ModuleInfo>,
    pub params: usize,
}
