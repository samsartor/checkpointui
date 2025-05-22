# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust TUI application for inspecting safetensors files. The project is in early development with only a basic "Hello, world!" implementation in the main binary.

The goal is to build an interactive terminal UI that displays safetensors files as a hierarchical tree structure, similar to the existing `lssafetensors.rs` script but with full navigation capabilities.

## Architecture

The codebase consists of:
- Main TUI application (`src/main.rs`) - currently minimal, needs full implementation
- Reference implementation (`lssafetensors.rs`) - standalone script showing safetensors parsing and tree building logic

The reference script demonstrates the core data structures and algorithms needed:
- `ModuleInfo` struct for tree nodes with parameter counts
- Tree building from tensor names using dot notation parsing
- Hierarchical parameter counting and display formatting

## Key Design Requirements

The TUI should implement a two-panel layout:
- Left panel: Interactive module tree with arrow key navigation
- Right panel: Info display showing safetensors metadata

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

## Implementation Plan

### Phase 1: Setup & Core Data Structures
1. Add ratatui dependencies to `Cargo.toml` ✓
2. Extract and adapt the `ModuleInfo` struct and tree-building logic from `lssafetensors.rs`
3. Create a basic ratatui application skeleton with event handling

### Phase 2: File Loading & Data Processing
4. Implement safetensors file loading and parsing
5. Build the hierarchical tree structure from tensor names
6. Apply automatic flattening for single-child module chains

### Phase 3: UI Layout & Navigation
7. Create two-panel layout (tree on left, info on right)
8. Implement tree widget with expandable/collapsible nodes
9. Add arrow key navigation (up/down selection, left/right for tree traversal)
10. Add space key for expand/collapse functionality

### Phase 4: Information Display
11. Implement top bar showing file path and current module location
12. Add bottom bar with parameter statistics
13. Create right panel info display for safetensors metadata

### Phase 5: Polish & Testing
14. Add error handling for invalid files
15. Optimize rendering performance for large models
16. Test with various safetensors files

## Dependencies

The project uses ratatui for the TUI interface and safetensors parsing dependencies from the reference script.