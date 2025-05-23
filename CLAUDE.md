# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust TUI application for inspecting safetensors files.

## Architecture

The codebase consists of:
- Main TUI application (`src/main.rs`)
- Reference implementation (`lssafetensors.rs`) - standalone script showing safetensors parsing and tree building logic

The reference script demonstrates the core data structures and algorithms needed:
- `ModuleInfo` struct for tree nodes with parameter counts
- Tree building from tensor names using dot notation parsing
- Hierarchical parameter counting and display formatting

## Key Design Requirements

The TUI should implement a two-panel layout:
- Left panel: Interactive module tree with arrow key navigation
- Right panel: Info display showing metadata

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

Run the reference script:
```bash
./lssafetensors.rs <safetensors_file>
```

## Dependencies

The project uses ratatui for the TUI interface and safetensors parsing dependencies from the reference script.
