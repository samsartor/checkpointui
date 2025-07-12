# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust TUI application for inspecting safetensors/gguf files.

## Architecture

The codebase consists of:
- The entrypoint (`src/main.rs`)
- Main TUI application (`src/app.rs`)
- Utils for understanding checkpoint files (`src/model.rs`)
- Generates statistics from huge arrays of f32s  (`src/analysis.rs`)
- Safetensors-specific logic (`src/safetensors.rs`)
- GGUF-specific logic (`src/gguf.rs`)
- Unsafe wrapper around the ggml library, mainly for dequantization (`ggml-base`)
- The ggml library dependency - don't look here unless instructed (`ggml-base/ggml`)

## Key Design Requirements

The TUI should implement a two-panel layout:
- Left-most panel: Interactive module tree with arrow key navigation
- Right panels: Info display showing metadata and tensor analysis

Navigation behavior:
- Arrow keys for tree traversal (up/down for selection, right to enter modules, left to go up levels)
- Space to expand/collapse modules in-place
- Automatic flattening of single-child module chains
- Top bar showing current file path and module location
- Bottom bar with parameter statistics

## Commands

Build and run:
```bash
cargo run
```

Build release:
```bash
cargo build --release
```

## Dependencies

The project uses ratatui for the TUI interface
