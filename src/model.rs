use anyhow::Error;
use owning_ref::ArcRef;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::{cmp, fmt, hash, mem, ops};

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum TensorTy {
    BOOL,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    I16,
    U16,
    F16,
    BF16,
    F32,
    F64,
    I64,
    U64,
    Unknown(String),
}

impl fmt::Display for TensorTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TensorTy::*;
        let text = match self {
            BOOL => "BOOL",
            U8 => "U8",
            I8 => "I8",
            F8_E5M2 => "F8_E5M2",
            F8_E4M3 => "F8_E4M3",
            I16 => "I16",
            U16 => "U16",
            F16 => "F16",
            BF16 => "BF16",
            F32 => "F32",
            F64 => "F64",
            I64 => "I64",
            U64 => "U64",
            Unknown(text) => text,
        };
        write!(f, "{}", text,)
    }
}

#[derive(Debug)]
pub enum TensorSeek {
    InFile { start: u64, end: u64 },
}

#[derive(Debug)]
pub struct TensorInfo {
    pub ty: TensorTy,
    pub shape: Vec<u64>,
    pub size: u64,
    pub seek: TensorSeek,
}

pub enum TensorReadError {
    Unsupported(TensorTy),
}

fn convertbytes<T, O>(bytes: &[u8], map: impl Fn(T) -> O) -> Vec<O>
where
    T: zerocopy::AsBytes + zerocopy::FromBytes,
{
    let stride = std::mem::size_of::<T>();
    let len = bytes.len() / stride;
    let mut out = Vec::with_capacity(len);
    let mut i = 0;
    while i < bytes.len() {
        let mut this: T = T::new_zeroed();
        this.as_bytes_mut().copy_from_slice(&bytes[i..][..stride]);
        out.push(map(this));
        i += stride;
    }
    out
}

impl TensorInfo {
    pub fn read_f32(&self, bytes: &[u8]) -> Result<Vec<f32>, TensorReadError> {
        use TensorTy::*;
        Ok(match &self.ty {
            F32 => convertbytes::<f32, _>(bytes, |x| x),
            F64 => convertbytes::<f64, _>(bytes, |x| x as f32),
            F16 => convertbytes::<half::f16, _>(bytes, |x| x.into()),
            BF16 => convertbytes::<half::bf16, _>(bytes, |x| x.into()),
            F8_E4M3 => convertbytes::<float8::F8E4M3, _>(bytes, |x| x.into()),
            F8_E5M2 => convertbytes::<float8::F8E5M2, _>(bytes, |x| x.into()),
            other => return Err(TensorReadError::Unsupported(other.clone())),
        })
    }

    pub fn read_f64(&self, bytes: &[u8]) -> Result<Vec<f64>, TensorReadError> {
        use TensorTy::*;
        Ok(match &self.ty {
            F32 => convertbytes::<f32, _>(bytes, |x| x as f64),
            F64 => convertbytes::<f64, _>(bytes, |x| x),
            F16 => convertbytes::<half::f16, _>(bytes, |x| x.into()),
            BF16 => convertbytes::<half::bf16, _>(bytes, |x| x.into()),
            F8_E4M3 => convertbytes::<float8::F8E4M3, _>(bytes, |x| x.into()),
            F8_E5M2 => convertbytes::<float8::F8E5M2, _>(bytes, |x| x.into()),
            other => return Err(TensorReadError::Unsupported(other.clone())),
        })
    }
}

pub enum PathSplit {
    Delim(char),
}

impl Default for PathSplit {
    fn default() -> Self {
        PathSplit::Delim('.')
    }
}

impl PathSplit {
    pub fn split(&self, fullname: Arc<str>) -> Vec<Key> {
        let mut parts = Vec::new();
        let mut at = 0;
        match self {
            &PathSplit::Delim(d) => {
                while let Some(off) = fullname[at..].find(d) {
                    parts.push(Key {
                        full: fullname.clone(),
                        start: at,
                        end: at + off,
                    });
                    at += off;
                    at += 1;
                }
            }
        }
        parts.push(Key {
            full: fullname.clone(),
            start: at,
            end: fullname.len(),
        });
        parts
    }
}

#[derive(Default, Debug)]
pub struct ModuleInfo {
    pub full_name: Key,
    pub tensor_info: Option<TensorInfo>,
    pub children: BTreeMap<Key, ModuleInfo>,
    pub total_tensors: u64,
    pub total_params: u64,
}

impl ModuleInfo {
    pub fn new(full_name: Key) -> Self {
        Self {
            full_name,
            tensor_info: None,
            children: BTreeMap::new(),
            total_tensors: 0,
            total_params: 0,
        }
    }

    pub fn build_from_tensors(
        tensors: impl IntoIterator<Item = (String, TensorInfo)>,
        split: &PathSplit,
    ) -> Self {
        let mut root = ModuleInfo::default();

        for (name, info) in tensors.into_iter() {
            let params = info.shape.iter().copied().product::<u64>();

            let parts = split.split(name.into());
            let mut current = &mut root;
            current.total_params += params;
            current.total_tensors += 1;

            for key in parts {
                current = current
                    .children
                    .entry(key.clone())
                    .or_insert_with(|| ModuleInfo::new(key.absolute()));
                current.total_params += params;
                current.total_tensors += 1;
            }
            current.tensor_info = Some(info);
        }

        root
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
                (k.join(ck), cv)
            })
            .collect();
    }
}

pub trait ModuleSource {
    fn module(&mut self, split: &PathSplit) -> Result<ModuleInfo, Error>;
    fn metadata(&mut self) -> Result<Value, Error>;
    fn tensor(&mut self, seek: TensorSeek) -> Result<Vec<u8>, Error>;
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

#[derive(Clone, Debug)]
pub struct Key {
    full: Arc<str>,
    start: usize,
    end: usize,
}

impl Key {
    pub fn absolute(mut self) -> Self {
        self.start = 0;
        self
    }

    pub fn join(&self, mut child: Key) -> Self {
        assert_eq!(self.full[..self.end], child.full[..self.end]);
        child.start = self.start;
        child
    }
}

impl ops::Deref for Key {
    type Target = str;

    fn deref(&self) -> &str {
        &self.full[self.start..self.end]
    }
}

impl From<Key> for ArcRef<str> {
    fn from(Key { full, start, end }: Key) -> Self {
        ArcRef::from(full).map(|s| &s[start..end])
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <str as fmt::Display>::fmt(self, f)
    }
}

impl cmp::PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(<Self as cmp::Ord>::cmp(self, other))
    }
}

impl cmp::Ord for Key {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        <str as cmp::Ord>::cmp(self, other)
        /*match (self.parse::<u64>(), other.parse::<u64>()) {
            (Some(a), Some(b)) => <str as cmp::Ord>::cmp(self, other),
            (None, Some(_)) => cmp::Ordering::Less,
            (Some(_), None) => cmp::Ordering::Greater,
            (None, None) => <str as cmp::Ord>::cmp(self, other),
        }*/
    }
}

impl cmp::PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.full, &other.full) && self.start == other.start && self.end == other.end)
            || <str as cmp::PartialEq>::eq(self, other)
    }
}

impl cmp::Eq for Key {}

impl hash::Hash for Key {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        <str as hash::Hash>::hash(self, state)
    }
}

impl Default for Key {
    fn default() -> Self {
        Key {
            full: Arc::from(""),
            start: 0,
            end: 0,
        }
    }
}
