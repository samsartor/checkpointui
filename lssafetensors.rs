#!/usr/bin/env -S cargo +nightly -Zscript
---cargo
[package]
edition="2021"

[dependencies]
clap = { version = "4.2", features = ["derive"] }
safetensors = "0.4.2"
colored = "2.0"
regex = "1.5"
human_format = "1.1"
serde_json = "1"
colored_json = "5.0"
---

// Written with input from:
// https://www.perplexity.ai/search/write-a-small-rust-cli-to-prin-3iWqYfykQHSrGwHeo8u9vQ
// https://www.perplexity.ai/search/i-have-this-script-for-printin-c52vyPKHTICBv8M6vftiDA

use clap::Parser;
use colored::*;
use regex::Regex;
use safetensors::SafeTensors;
use safetensors::tensor::{TensorInfo, Metadata, SafeTensorError};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::fmt;
use std::io::Read;
use std::path::{PathBuf, Path};
use human_format::Formatter;
use colored_json::prelude::*;

#[derive(Parser)]
#[command(name = "safetensors_metadata")]
#[command(about = "Print safetensors metadata to stdout")]
struct Cli {
    #[arg(help = "Path to the safetensors file")]
    file_path: PathBuf,
    #[arg(short, long, help = "Regex pattern to filter tensor names")]
    regex: Option<String>,
    #[arg(short = 'j', long = "json", help = "Pretty-print metadata as JSON")]
    json: bool,
}

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord)]
enum Key {
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

struct ModuleInfo {
    full_name: String,
    tensor_info: Option<TensorInfo>,
    children: BTreeMap<Key, ModuleInfo>,
    params: usize,
}

fn read_metadata_from_file(file_path: &Path) -> Result<(usize, Metadata), SafeTensorError> {
    let mut file = File::open(file_path)?;
    
    // Read first 8 bytes
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)?;
    let n = u64::from_le_bytes(header_size_bytes) as usize;
    
    // Read n bytes for metadata
    let mut metadata_bytes = vec![0u8; n];
    file.read_exact(&mut metadata_bytes)?;
    
    let metadata_str = std::str::from_utf8(&metadata_bytes)
        .map_err(|_| SafeTensorError::InvalidHeader)?;
    
    let metadata: Metadata = serde_json::from_str(metadata_str)
        .map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
    
    Ok((n, metadata))
}

fn build_tree(tensors: HashMap<String, &TensorInfo>, regex: Option<&Regex>) -> ModuleInfo {
    let mut root = ModuleInfo {
        full_name: "".to_string(),
        tensor_info: None,
        children: BTreeMap::new(),
        params: 0,
    };

    for (name, info) in tensors {
        match regex {
            Some(r) if !r.is_match(&name) => continue,
            _ => (),
        }

        let params = info.shape.iter().copied().product::<usize>();
                
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = &mut root;
        current.params += params;

        for (i, &part) in parts.iter().enumerate() {
            let key = match part.parse() {
                Ok(i) => Key::Index(i),
                Err(_) => Key::Name(part.to_string()),
            };
            current = current.children.entry(key).or_insert(ModuleInfo {
                full_name: name.clone(),
                tensor_info: None,
                children: BTreeMap::new(),
                params: 0,
            });
            current.params += params;

            if i == parts.len() - 1 {
                current.tensor_info = Some(info.clone());
            }
        }
    }

    root
}

fn print_tree(module: &ModuleInfo, name: &str, depth: usize, count_form: &Formatter, size_form: &Formatter) {
    let indent = "  ".repeat(depth);

    if module.tensor_info.is_some() {
        let info = module.tensor_info.as_ref().unwrap();
        println!(
            "{indent}{}: {:?} ({} params) {} {}", 
            module.full_name.cyan(),
            info.shape,
            count_form.format(module.params as f64),
            format!("{:?}", info.dtype).yellow(),
            size_form.format((info.data_offsets.1 - info.data_offsets.0) as f64),
        );
    } else if !module.children.is_empty() {
        println!("{indent}{} ({} params)", name.blue().bold(), count_form.format(module.params as f64));
    }

    for (child_name, child_module) in &module.children {
        print_tree(child_module, &format!("{child_name}"), depth + 1, count_form, size_form);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let regex = cli.regex.as_ref().map(|r| Regex::new(r).expect("Invalid regex pattern"));

    let (header_size, metadata) = read_metadata_from_file(&cli.file_path)?;
    if cli.json {
        // Pretty-print JSON with colors
        let json_output = serde_json::to_string_pretty(&metadata)?;
        println!("{}", json_output.to_colored_json_auto()?);
        return Ok(())
    }

    let mut size_form = Formatter::new();
    let size_form = size_form.with_decimals(2).with_separator("").with_units("B");
    let mut count_form = Formatter::new();
    let count_form = count_form.with_decimals(2).with_separator("");
    
    let tree = build_tree(metadata.tensors(), regex.as_ref());

    println!("{}", "Safetensors Metadata".green().bold());
    println!("{}: {}", "File".cyan(), cli.file_path.display());    
    println!("{}: {}", "Header size".cyan(), size_form.format(header_size as f64));
    println!("{}: {}", "Number of tensors".cyan(), metadata.tensors().len());
    println!("{}: {}", "Number of parameters".cyan(), count_form.format(tree.params as f64));

    println!("\n{}", "Tensor Tree".yellow().bold());
    for (child_name, child_module) in tree.children {
        print_tree(&child_module, &format!("{child_name}"), 0, count_form, size_form);
    }

    if let Some(extra_metadata) = metadata.metadata() {
        println!("\n{}", "Extra Metadata".yellow().bold());
        for (key, value) in extra_metadata {
            println!("{}: {}", key.cyan(), value);
        }
    }

    Ok(())
}

