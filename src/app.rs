use anyhow::{Error, bail};
use human_format::{Formatter, Scales};
use lexical_sort::natural_lexical_cmp;
use owning_ref::ArcRef;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Text};
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, ListState, Paragraph, StatefulWidget, Wrap,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::Hash;
use std::io::{Stdout, stdout};
use std::mem;
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use weakref::Own;

use crate::analysis::{Analysis, AnalysisCell, start_analysis_thread};
use crate::gguf::Gguf;
use crate::model::{Key, ModuleInfo, ModuleSource, PathSplit, shorten_value};
use crate::safetensors::Safetensors;
use crate::storage::FileStorage;

pub trait TreeData: Send + Sync {
    type Id: Ord + Hash + Clone;

    fn has_children(&self) -> bool;
    fn children(this: ArcRef<Self>) -> Box<dyn Iterator<Item = (String, ArcRef<Self>)>>;
    fn unique_id(&self) -> Self::Id;
}

impl TreeData for ModuleInfo {
    type Id = Key;

    fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    fn children(this: ArcRef<Self>) -> Box<dyn Iterator<Item = (String, ArcRef<Self>)>> {
        let keys: Vec<_> = this.children.keys().cloned().collect();
        Box::new(keys.into_iter().map(move |key| {
            let child = this.clone().map(|m| &m.children[&key]);
            (key.to_string(), child)
        }))
    }

    fn unique_id(&self) -> Self::Id {
        self.full_name.clone()
    }
}

impl ModuleInfo {
    pub fn is_tensor(&self) -> bool {
        self.tensor_info.is_some()
    }
}

impl TreeData for Value {
    type Id = *const Value;

    fn has_children(&self) -> bool {
        match self {
            Value::Object(map) => !map.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            _ => false,
        }
    }

    fn children(this: ArcRef<Self>) -> Box<dyn Iterator<Item = (String, ArcRef<Self>)>> {
        match &*this {
            Value::Object(map) => {
                let keys: Vec<_> = map.keys().cloned().collect();
                let iter: Box<dyn Iterator<Item = (String, ArcRef<Self>)>> =
                    Box::new(keys.into_iter().map(move |key| {
                        let child = this.clone().map(|v| match v {
                            Value::Object(map) => &map[&key],
                            _ => unreachable!(),
                        });
                        (key, child)
                    }));
                iter
            }
            Value::Array(arr) => {
                let len = arr.len();
                let iter: Box<dyn Iterator<Item = (String, ArcRef<Self>)>> =
                    Box::new((0..len).map(move |i| {
                        let child = this.clone().map(|v| match v {
                            Value::Array(arr) => &arr[i],
                            _ => unreachable!(),
                        });
                        (format!("[{i}]"), child)
                    }));
                iter
            }
            _ => {
                let iter: Box<dyn Iterator<Item = (String, ArcRef<Self>)>> =
                    Box::new(std::iter::empty());
                iter
            }
        }
    }

    fn unique_id(&self) -> Self::Id {
        // Use the pointer address as a unique identifier for Value items
        self as *const Value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
enum Panel {
    #[default]
    Tree,
    SelectedInfo,
    FileInfo,
    Analysis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DialogType {
    Edit,
    Delete,
    Error(String),
}

impl Panel {
    fn next(self, analysis: bool) -> Self {
        match self {
            Panel::Tree => Panel::FileInfo,
            Panel::SelectedInfo => Panel::FileInfo,
            Panel::FileInfo if analysis => Panel::Analysis,
            Panel::FileInfo => Panel::Tree,
            Panel::Analysis => Panel::Tree,
        }
    }

    fn prev(self, analysis: bool) -> Self {
        match self {
            Panel::Tree if analysis => Panel::Analysis,
            Panel::Tree => Panel::FileInfo,
            Panel::SelectedInfo => Panel::Tree,
            Panel::FileInfo => Panel::Tree,
            Panel::Analysis => Panel::FileInfo,
        }
    }
}

pub type Backend = CrosstermBackend<Stdout>;

pub const PANEL_BORDER: Color = Color::White;
pub const PANEL_BORDER_SECONDARY: Color = Color::White;
pub const PANEL_BORDER_SELECTED: Color = Color::Yellow;
pub const MODULE_FG: Color = Color::Blue;
pub const TENSOR_FG: Color = Color::Cyan;
pub const SHAPE_FG: Color = Color::White;
pub const DTYPE_FG: Color = Color::Yellow;
pub const COUNT_FG: Color = Color::White;
pub const BYTESIZE_FG: Color = Color::Magenta;

#[derive(Default)]
pub struct App {
    should_quit: bool,
    file_path: Option<PathBuf>,
    tree_state: Option<TreeState<ModuleInfo>>,
    meta_tree_state: Option<TreeState<Value>>,
    source: Option<Arc<Mutex<dyn ModuleSource + Send>>>,
    count_formatter: Formatter,
    bytes_formatter: Formatter,
    selected_panel: Panel,
    pub helptext: String,
    pub path_split: PathSplit,
    analysis_sender: Option<Own<Box<AnalysisCell>>>,
    current_analysis: Option<Own<Box<Analysis>>>,
    histogram_size_limit: u64,
    spectrum_size_limit: u64,
    dialog_type: Option<DialogType>,
    edit_draft: String,
}

struct TreeState<T: TreeData> {
    data: ArcRef<T>,
    data_history: Vec<ArcRef<T>>,
    expanded: HashSet<T::Id>,
    visible_items: Vec<TreeItem<T>>,
    list_state: RefCell<ListState>,
}

#[derive(Clone)]
struct TreeItem<T: TreeData> {
    name: String,
    depth: i32,
    is_expanded: bool,
    info: ArcRef<T>,
}

impl<T: TreeData> TreeItem<T> {
    pub fn has_children(&self) -> bool {
        self.info.has_children()
    }
}

impl<T: TreeData> TreeState<T> {
    fn new(root: ArcRef<T>) -> Self {
        Self {
            data: root,
            data_history: Vec::new(),
            expanded: HashSet::new(),
            visible_items: Vec::new(),
            list_state: RefCell::new(ListState::default()),
        }
    }

    fn rebuild_visible_items(&mut self) {
        self.visible_items.clear();
        let mut stack = vec![(self.data.clone(), "".to_string(), -1)];
        while let Some((info, name, depth)) = stack.pop() {
            // Use the unique_id method to get a proper identifier for each item
            let is_expanded = depth < 0 || self.expanded.contains(&info.unique_id());
            if is_expanded {
                let stack_at = stack.len();
                for (key, child) in T::children(info.clone()) {
                    stack.push((child, key, depth + 1));
                }
                // Sort by name for now - we'll make this more sophisticated later
                stack[stack_at..].sort_by(|(_, a_name, ..), (_, b_name, ..)| {
                    natural_lexical_cmp(b_name, a_name)
                });
            }
            if depth >= 0 {
                self.visible_items.push(TreeItem {
                    name,
                    depth,
                    is_expanded,
                    info,
                });
            }
        }
    }

    fn toggle_expanded(&mut self) {
        let Some(index) = self.list_state.borrow().selected() else {
            return;
        };
        let Some(item) = self.visible_items.get(index) else {
            return;
        };
        if !item.has_children() {
            return;
        }
        let id = item.info.unique_id();
        if self.expanded.contains(&id) {
            self.expanded.remove(&id);
        } else {
            self.expanded.insert(id);
        }
    }

    fn move_up(&mut self) {
        self.list_state.get_mut().select_previous();
    }

    fn move_down(&mut self) {
        self.list_state.get_mut().select_next();
    }

    fn move_right(&mut self) {
        let Some(index) = self.list_state.get_mut().selected() else {
            return;
        };
        let Some(item) = self.visible_items.get(index) else {
            return;
        };
        if !item.has_children() {
            return;
        }
        let prev_data = mem::replace(&mut self.data, item.info.clone());
        self.data_history.push(prev_data);
        self.rebuild_visible_items();
        self.list_state.get_mut().select(Some(0));
    }

    fn move_left(&mut self) {
        let Some(goto_data) = self.data_history.pop() else {
            return;
        };
        let prev_data = mem::replace(&mut self.data, goto_data);
        self.rebuild_visible_items();
        let index = self
            .visible_items
            .iter()
            .position(|i| std::ptr::eq(&*i.info, &*prev_data));
        self.list_state.get_mut().select(index);
    }
}

impl App {
    pub fn new() -> Self {
        let mut this = App::default();
        let mut count_scales = Scales::new();
        count_scales
            .with_base(1000)
            .with_suffixes(vec!["", "K", "M", "B", "T"]);
        this.count_formatter
            .with_separator("")
            .with_scales(count_scales);
        this.bytes_formatter
            .with_scales(Scales::Binary())
            .with_units("B");
        // Set configurable size limits for analysis
        // Lower limit for histogram as it's cheaper to compute
        this.histogram_size_limit = 100 * 1024 * 1024; // 100Mi elements
        this.spectrum_size_limit = 2 * 1024 * 1024; // 2Mi elements (SVD is more expensive)
        this
    }

    pub fn load_file(&mut self, file_path: PathBuf) -> Result<(), Error> {
        let ext = file_path.extension().and_then(|ext| ext.to_str());
        let storage = FileStorage::new(file_path.clone());
        if ext == Some("safetensors") {
            self.source = Some(Arc::new(Mutex::new(Safetensors::open(storage)?)));
        } else if ext == Some("gguf") {
            self.source = Some(Arc::new(Mutex::new(Gguf::open(storage)?)));
        } else {
            bail!("could not infer file type");
        }
        self.file_path = Some(file_path);
        self.rebuild_module()
    }

    pub fn rebuild_module(&mut self) -> Result<(), Error> {
        let Some(source) = &self.source else {
            return Ok(());
        };

        {
            // Create module tree state
            let mut data = source.lock().unwrap();
            let mut module = data.module(&self.path_split)?;
            module.flatten_single_children();
            let mut state = TreeState::new(Arc::new(module).into());
            state.rebuild_visible_items();
            self.tree_state = Some(state);

            // Create metadata tree state
            let extra_metadata = data.metadata()?;
            let mut meta_state = TreeState::new(Arc::new(extra_metadata).into());
            meta_state.rebuild_visible_items();
            self.meta_tree_state = Some(meta_state);
        }

        // Now that we have the tree, move the source to the analysis thread
        let sender = self
            .analysis_sender
            .insert(Own::new_box(AnalysisCell::new()))
            .refer();
        start_analysis_thread(source.clone(), sender);

        // Start analysis for the initially selected tensor
        self.update_analysis_for_selected_tensor();
        Ok(())
    }

    pub fn handle_events(&mut self) -> Result<(), Error> {
        if let Event::Key(key) = event::read()? {
            // Handle dialog events first
            if let Some(dialog_type) = &self.dialog_type {
                match key.code {
                    KeyCode::Esc => {
                        // Cancel dialog
                        self.dialog_type = None;
                        self.edit_draft.clear();
                    }
                    KeyCode::Enter => {
                        // Confirm action
                        match dialog_type {
                            DialogType::Edit => {
                                // Parse the edit_draft and update metadata
                                self.dialog_type = None;
                                let new_value = self.parse_edit_draft();
                                self.edit_draft.clear();
                                self.update_selected_metadata(Some(new_value));
                            }
                            DialogType::Delete => {
                                // Delete the metadata
                                self.dialog_type = None;
                                self.edit_draft.clear();
                                self.update_selected_metadata(None);
                            }
                            DialogType::Error(_) => {
                                // Close error dialog
                                self.dialog_type = None;
                            }
                        }
                    }
                    KeyCode::Char(c) if matches!(dialog_type, DialogType::Edit) => {
                        // Add character to edit draft
                        self.edit_draft.push(c);
                    }
                    KeyCode::Backspace if matches!(dialog_type, DialogType::Edit) => {
                        // Remove last character from edit draft
                        self.edit_draft.pop();
                    }
                    _ => {}
                }
                return Ok(());
            }

            match (key.code, self.selected_panel, &mut self.tree_state) {
                (KeyCode::Char('q') | KeyCode::Esc, _, _) => self.should_quit = true,
                (KeyCode::Tab, _, _) => {
                    self.selected_panel =
                        self.selected_panel.next(self.should_show_analysis_panel())
                }
                (KeyCode::BackTab, _, _) => {
                    self.selected_panel =
                        self.selected_panel.prev(self.should_show_analysis_panel())
                }
                // Tree panel controls
                (KeyCode::Up, Panel::Tree, Some(s)) => {
                    s.move_up();
                    self.update_analysis_for_selected_tensor();
                }
                (KeyCode::Down, Panel::Tree, Some(s)) => {
                    s.move_down();
                    self.update_analysis_for_selected_tensor();
                }
                (KeyCode::Left, Panel::Tree, Some(s)) => {
                    s.move_left();
                    self.update_analysis_for_selected_tensor();
                }
                (KeyCode::Right, Panel::Tree, Some(s)) => {
                    s.move_right();
                    self.update_analysis_for_selected_tensor();
                }
                (KeyCode::Char(' ') | KeyCode::Enter, Panel::Tree, Some(s)) => {
                    s.toggle_expanded();
                    s.rebuild_visible_items();
                    self.update_analysis_for_selected_tensor();
                }
                (KeyCode::Char('y'), _, _) => {
                    self.handle_y_key();
                }

                // FileInfo panel controls (metadata tree)
                (KeyCode::Up, Panel::FileInfo, _) => {
                    if let Some(s) = &mut self.meta_tree_state {
                        s.move_up();
                    }
                }
                (KeyCode::Down, Panel::FileInfo, _) => {
                    if let Some(s) = &mut self.meta_tree_state {
                        s.move_down();
                    }
                }
                (KeyCode::Left, Panel::FileInfo, _) => {
                    if let Some(s) = &mut self.meta_tree_state {
                        s.move_left();
                    }
                }
                (KeyCode::Right, Panel::FileInfo, _) => {
                    if let Some(s) = &mut self.meta_tree_state {
                        s.move_right();
                    }
                }
                (KeyCode::Char(' ') | KeyCode::Enter, Panel::FileInfo, _) => {
                    if let Some(s) = &mut self.meta_tree_state {
                        s.toggle_expanded();
                        s.rebuild_visible_items();
                    }
                }
                (KeyCode::Char('e'), Panel::FileInfo, _) => {
                    // Open edit dialog for selected metadata item
                    if let Some(value_str) = self.get_selected_metadata_value_string() {
                        self.edit_draft = value_str;
                        self.dialog_type = Some(DialogType::Edit);
                    }
                }
                (KeyCode::Char('d'), Panel::FileInfo, _) => {
                    // Open delete dialog for selected metadata item
                    if self.is_metadata_item_selected() {
                        self.dialog_type = Some(DialogType::Delete);
                    }
                }

                // Analysis panel controls (currently read-only)
                (_, Panel::Analysis, _) => {}
                _ => {}
            }
        }
        Ok(())
    }

    pub fn run(&mut self, terminal: &mut Terminal<Backend>) -> Result<(), Error> {
        while !self.should_quit {
            terminal.draw(|f| self.render_ui(f))?;
            if event::poll(Duration::from_millis(100))? {
                self.handle_events()?;
            }
        }
        Ok(())
    }

    fn render_ui(&mut self, f: &mut ratatui::Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Top bar
                Constraint::Min(1),    // Main content
                Constraint::Length(3), // Bottom bar
            ])
            .split(f.area());

        // Top bar
        let title = if let Some(path) = &self.file_path {
            format!("CheckpoinTUI - {}", path.display())
        } else {
            "CheckpoinTUI - No file loaded".to_string()
        };

        let top_bar = Paragraph::new(title)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(PANEL_BORDER_SECONDARY));
        f.render_widget(top_bar, chunks[0]);

        // Main content area
        if self.tree_state.is_some() {
            let should_show_analysis = self.should_show_analysis_panel();

            if should_show_analysis {
                // Three-panel layout when tensor is selected
                let main_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(33), // Tree panel
                        Constraint::Percentage(33), // Info panel
                        Constraint::Percentage(34), // Analysis panel
                    ])
                    .split(chunks[1]);

                self.render_tree_panel(f, main_chunks[0]);

                // Split info panel into two vertical sections
                let info_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(25), // Selected item info
                        Constraint::Percentage(75), // File info
                    ])
                    .split(main_chunks[1]);

                self.render_selected_info_panel(f, info_chunks[0]);
                self.render_file_meta_tree_panel(f, info_chunks[1]);
                self.render_analysis_panel(f, main_chunks[2]);
            } else {
                // Two-panel layout when module is selected
                let main_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(50), // Tree panel
                        Constraint::Percentage(50), // Info panel
                    ])
                    .split(chunks[1]);

                self.render_tree_panel(f, main_chunks[0]);

                // Split info panel into two vertical sections
                let info_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(25), // Selected item info
                        Constraint::Percentage(75), // File info
                    ])
                    .split(main_chunks[1]);

                self.render_selected_info_panel(f, info_chunks[0]);
                self.render_file_meta_tree_panel(f, info_chunks[1]);
            }
        } else {
            let help = Paragraph::new(self.helptext.as_str())
                .block(Block::default().borders(Borders::ALL).title("Help"))
                .style(Style::default().fg(Color::White));
            f.render_widget(help, chunks[1]);
        }

        // Bottom bar
        let help_text = if self.tree_state.is_some() {
            if self.selected_panel == Panel::FileInfo && self.is_metadata_item_selected() {
                "↑/↓: Navigate | ←/→: Enter/Exit | Space: Expand/Collapse | e: Edit | d: Delete | Tab: Switch Panel | q: Quit"
            } else {
                "↑/↓: Navigate | ←/→: Enter/Exit Module | Space/Enter: Expand/Collapse | Tab/Shift+Tab: Switch Panel | q/Esc: Quit"
            }
        } else {
            "q/Esc: Quit"
        };

        let bottom_bar = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Gray));
        f.render_widget(bottom_bar, chunks[2]);

        // Render dialog overlay if open
        if self.dialog_type.is_some() {
            self.render_dialog(f, f.area());
        }
    }

    fn render_tree_panel(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let Some(tree) = &self.tree_state else {
            return;
        };

        let lines: Vec<Line> = tree
            .visible_items
            .iter()
            .map(|item| {
                let mut spans = Vec::new();

                // Indentation
                if item.depth > 0 {
                    spans.push("  ".repeat(item.depth as usize).into());
                }

                // Icon
                let icon_span = if item.has_children() {
                    if item.is_expanded { "▼ " } else { "▶ " }
                } else if item.info.is_tensor() {
                    "📄 "
                } else {
                    "  "
                }
                .into();
                spans.push(icon_span);

                // Name
                let name_span = if item.info.is_tensor() {
                    item.name.as_str().fg(TENSOR_FG)
                } else if item.has_children() {
                    item.name.as_str().fg(MODULE_FG).bold()
                } else {
                    item.name.as_str().white()
                };
                spans.push(name_span);

                // Parameter count
                let param_text = format!(" ({})", self.format_count(item.info.total_params));
                spans.push(param_text.fg(COUNT_FG));

                // Tensor details
                if let Some(tensor_info) = &item.info.tensor_info {
                    spans.push(format!(" {:?}", tensor_info.shape).fg(SHAPE_FG));
                    spans.push(format!(" {}", tensor_info.ty).fg(DTYPE_FG));
                    let size = self.format_bytes(tensor_info.size as u64);
                    spans.push(format!(" {size}").fg(BYTESIZE_FG));
                }

                Line::from(spans)
            })
            .collect();

        let mut title: Line = "Module Tree".into();
        if !tree.data.full_name.is_empty() {
            title += " - ".into();
            title += tree.data.full_name.fg(MODULE_FG);
        }

        let items: Vec<ListItem> = lines.into_iter().map(ListItem::new).collect();

        let list = List::new(items)
            .block(self.format_block(title, Panel::Tree))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().bg(Color::Blue).fg(Color::White));
        list.render(area, f.buffer_mut(), &mut *tree.list_state.borrow_mut());
    }

    fn render_selected_info_panel(&self, f: &mut ratatui::Frame, area: Rect) {
        let Some(tree) = &self.tree_state else { return };
        let selected_item = tree
            .list_state
            .borrow()
            .selected()
            .and_then(|i| tree.visible_items.get(i));

        let mut text = Text::default();
        let title = if let Some(item) = selected_item {
            if let Some(tensor_info) = &item.info.tensor_info {
                text.push_line(vec!["Path: ".bold(), item.info.full_name.fg(TENSOR_FG)]);
                text.push_line(vec![
                    "Shape: ".bold(),
                    format!("{:?}", tensor_info.shape).fg(SHAPE_FG),
                ]);
                text.push_line(vec![
                    "Data Type: ".bold(),
                    format!("{}", tensor_info.ty).fg(DTYPE_FG),
                ]);
                text.push_line(vec![
                    "Parameters: ".bold(),
                    self.format_count(item.info.total_params).fg(COUNT_FG),
                ]);
                text.push_line(vec![
                    "Size: ".bold(),
                    self.format_bytes(tensor_info.size as u64).fg(BYTESIZE_FG),
                ]);
                "Tensor Info"
            } else {
                text.push_line(vec!["Path: ".bold(), item.info.full_name.fg(MODULE_FG)]);
                text.push_line(vec![
                    "Tensors: ".bold(),
                    item.info.total_tensors.to_string().fg(COUNT_FG),
                ]);
                text.push_line(vec![
                    "Parameters: ".bold(),
                    self.format_count(item.info.total_params).fg(COUNT_FG),
                ]);
                "Module Info"
            }
        } else {
            text.extend(Text::from("No item selected".gray()));
            "Selection Info"
        };

        let info = Paragraph::new(text)
            .block(self.format_block(title, Panel::SelectedInfo))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });

        f.render_widget(info, area);
    }

    fn render_file_meta_tree_panel(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let Some(module_tree) = &self.tree_state else {
            return;
        };

        // Split the area into file info and metadata tree
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // File info (3 lines + 2 for borders)
                Constraint::Min(1),    // Metadata tree
            ])
            .split(area);

        // Render file info in top section
        let mut file_info = Text::default();
        file_info.push_line(vec![
            "Path: ".bold(),
            self.file_path
                .as_ref()
                .unwrap()
                .display()
                .to_string()
                .fg(TENSOR_FG),
        ]);
        file_info.push_line(vec![
            "Total Tensors: ".bold(),
            module_tree.data.total_tensors.to_string().fg(COUNT_FG),
        ]);
        file_info.push_line(vec![
            "Total Parameters: ".bold(),
            self.format_count(module_tree.data.total_params)
                .fg(COUNT_FG),
        ]);

        let file_info_widget = Paragraph::new(file_info)
            .block(Block::default().borders(Borders::ALL).title("File Info"))
            .style(Style::default().fg(Color::White));
        f.render_widget(file_info_widget, chunks[0]);

        // Render metadata tree in bottom section
        if let Some(tree) = &self.meta_tree_state {
            let lines: Vec<Line> = tree
                .visible_items
                .iter()
                .map(|item| {
                    let mut spans = Vec::new();

                    // Indentation
                    if item.depth > 0 {
                        spans.push("  ".repeat(item.depth as usize).into());
                    }

                    // Icon
                    let icon_span = if item.has_children() {
                        if item.is_expanded { "▼ " } else { "▶ " }
                    } else {
                        "📄 "
                    }
                    .into();
                    spans.push(icon_span);

                    // Name
                    let name_span = if item.has_children() {
                        item.name.as_str().fg(MODULE_FG).bold()
                    } else {
                        item.name.as_str().fg(TENSOR_FG)
                    };
                    spans.push(name_span);

                    // Value (for leaf nodes)
                    if shorten_value(&*item.info) {
                        spans.push(format!(" = ...").fg(Color::Gray));
                    } else {
                        match &*item.info {
                            Value::Null => {
                                spans.push(format!(" = null").fg(Color::Gray));
                            }
                            Value::Bool(bool) => {
                                spans.push(format!(" = {bool}").fg(Color::Blue));
                            }
                            Value::Number(number) => {
                                spans.push(format!(" = {number}").fg(Color::Blue));
                            }
                            Value::String(string) => {
                                spans.push(format!(" = {string}").fg(Color::White));
                            }
                            Value::Array(_) => {}
                            Value::Object(_) => {}
                        }
                    }

                    Line::from(spans)
                })
                .collect();

            let items: Vec<ListItem> = lines.into_iter().map(ListItem::new).collect();

            let list = List::new(items)
                .block(self.format_block("Metadata", Panel::FileInfo))
                .style(Style::default().fg(Color::White))
                .highlight_style(Style::default().bg(Color::Blue).fg(Color::White));
            list.render(
                chunks[1],
                f.buffer_mut(),
                &mut *tree.list_state.borrow_mut(),
            );
        } else {
            let no_metadata = Paragraph::new("No metadata available")
                .block(self.format_block("Metadata", Panel::FileInfo))
                .style(Style::default().fg(Color::Gray));
            f.render_widget(no_metadata, chunks[1]);
        }
    }

    fn format_block<'a>(&self, title: impl Into<Line<'a>>, panel: Panel) -> Block<'a> {
        let mut title: Line = title.into();
        let border_style = if self.selected_panel == panel {
            title += "*".into();
            Style::default().fg(PANEL_BORDER_SELECTED)
        } else {
            Style::default().fg(PANEL_BORDER)
        };
        title = title.bold();

        Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(title)
    }

    fn format_count(&self, count: u64) -> String {
        if count < 1000 {
            count.to_string()
        } else {
            self.count_formatter.format(count as f64)
        }
    }

    fn format_bytes(&self, bytes: u64) -> String {
        if bytes < 1000 {
            format!("{bytes} Bytes")
        } else {
            self.bytes_formatter.format(bytes as f64)
        }
    }

    fn should_show_analysis_panel(&self) -> bool {
        let Some(tree) = &self.tree_state else {
            return false;
        };
        tree.list_state
            .borrow()
            .selected()
            .and_then(|i| tree.visible_items.get(i))
            .is_some_and(|item| item.info.is_tensor())
    }

    fn render_analysis_panel(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let tensor_info = {
            let Some(tree) = &self.tree_state else { return };
            let selected_item = tree
                .list_state
                .borrow()
                .selected()
                .and_then(|i| tree.visible_items.get(i));

            let Some(item) = selected_item else {
                return;
            };

            let Some(tensor_info) = &item.info.tensor_info else {
                return;
            };

            tensor_info.clone()
        };

        let analysis_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50), // Histogram
                Constraint::Percentage(50), // Singular values (if 2D)
            ])
            .split(area);

        self.render_histogram(f, analysis_chunks[0]);

        if tensor_info.shape.len() == 2 {
            self.render_spectrum(f, analysis_chunks[1]);
        } else {
            let placeholder = Paragraph::new("SVD only possible on 2D tensors")
                .block(self.format_block("Matrix Spectrum", Panel::Analysis))
                .style(Style::default().fg(Color::Gray));
            f.render_widget(placeholder, analysis_chunks[1]);
        }
    }

    fn render_bar_chart(
        chart: &crate::analysis::BarChart,
        max_width: usize,
        color: Color,
        format_value: impl Fn(f32) -> String,
    ) -> Vec<Line<'static>> {
        let mut lines = Vec::new();

        if chart.bins.is_empty() {
            return lines;
        }

        let max_count = chart.bins.iter().max().cloned().unwrap_or(1) as f32;
        let bin_width = (chart.right - chart.left) / chart.bins.len() as f32;

        for (i, &count) in chart.bins.iter().enumerate() {
            let range_start = chart.left + i as f32 * bin_width;
            let range_end = chart.left + (i + 1) as f32 * bin_width;
            let bar_len = (count as f32 / max_count * max_width as f32) as usize;
            let bar = "█".repeat(bar_len);

            let label = if chart.continues_past_left && i == 0 {
                format!("       {}: ", format_value(range_end))
            } else if chart.continues_past_right && i + 1 == chart.bins.len() {
                format!("{}       : ", format_value(range_start))
            } else {
                format!(
                    "{} {}: ",
                    format_value(range_start),
                    format_value(range_end)
                )
            };

            lines.push(Line::from(vec![
                label.into(),
                bar.fg(color),
                format!(" ({count})").into(),
            ]));
        }

        lines
    }

    fn render_histogram_into(&mut self, text: &mut Text) {
        let Some(analysis) = self.current_analysis.as_ref() else {
            text.push_line("No analysis running");
            return;
        };

        if let Some(error) = analysis.error.get() {
            text.push_line(vec!["Error: ".fg(Color::Red), format!("{error}").into()]);
            return;
        }

        match (
            analysis.histogram.get(),
            analysis.histogram_go.load(Relaxed),
        ) {
            (Some(histogram), _) => {
                text.push_line(vec![
                    "Data range: ".bold(),
                    format!("{:.3} to {:.3}", histogram.min, histogram.max).into(),
                ]);
                text.push_line(Line::from(""));

                let chart_lines = Self::render_bar_chart(
                    &histogram.chart,
                    30, // max_width
                    Color::Blue,
                    |x| format!("{x:6.2}"),
                );
                text.extend(chart_lines);
            }
            (None, true) => {
                text.push_line(vec!["🔄 Computing histogram...".fg(Color::Yellow)]);
            }
            (None, false) => {
                text.push_line(vec!["Press \"y\" to compute histogram".fg(Color::Red)]);
            }
        }
    }

    fn render_histogram(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let mut text = Text::default();
        self.render_histogram_into(&mut text);
        let histogram_widget = Paragraph::new(text)
            .block(self.format_block("Histogram", Panel::Analysis))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });

        f.render_widget(histogram_widget, area);
    }

    fn render_spectrum_into(&mut self, text: &mut Text) {
        let Some(analysis) = self.current_analysis.as_ref() else {
            text.push_line("No analysis running");
            return;
        };

        if let Some(error) = analysis.error.get() {
            text.push_line(vec!["Error: ".fg(Color::Red), format!("{error}").into()]);
            return;
        }

        match (analysis.spectrum.get(), analysis.spectrum_go.load(Relaxed)) {
            (Some(spectrum), _) => {
                text.push_line(Line::from(""));

                let chart_lines = Self::render_bar_chart(
                    &spectrum.chart,
                    30, // max_width
                    Color::Blue,
                    |x| format!("{x:6.2}"),
                );
                text.extend(chart_lines);
            }
            (None, true) => {
                text.push_line(vec!["🔄 Computing SVD decomposition...".fg(Color::Yellow)]);
            }
            (None, false) => {
                text.push_line(vec![
                    "Press \"y\" to compute SVD decomposition".fg(Color::Red),
                ]);
            }
        }
    }

    fn render_spectrum(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let mut text = Text::default();
        self.render_spectrum_into(&mut text);

        let svd_widget = Paragraph::new(text)
            .block(self.format_block("Matrix Spectrum", Panel::Analysis))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });

        f.render_widget(svd_widget, area);
    }

    fn update_analysis_for_selected_tensor(&mut self) {
        let Some(tree) = &self.tree_state else { return };
        let selected_item = tree
            .list_state
            .borrow()
            .selected()
            .and_then(|i| tree.visible_items.get(i));

        let Some(item) = selected_item else { return };
        let Some(tensor_info) = &item.info.tensor_info else {
            return;
        };

        // Calculate total number of elements in the tensor
        let total_elements = tensor_info.shape.iter().copied().product::<u64>();

        let analysis = Own::new(Box::new(Analysis {
            tensor: tensor_info.clone(),
            histogram: OnceLock::new(),
            histogram_go: (total_elements <= self.histogram_size_limit).into(),
            spectrum: OnceLock::new(),
            spectrum_go: (total_elements <= self.spectrum_size_limit).into(),
            error: std::sync::OnceLock::new(),
            max_bin_count: 20,
        }));
        if let Some(sender) = self.analysis_sender.as_ref() {
            sender.set(analysis.refer());
        }
        self.current_analysis = Some(analysis);
    }

    fn handle_y_key(&mut self) {
        let Some(analysis) = &self.current_analysis else {
            return;
        };

        // Check if histogram is not set (not requested yet)
        if !analysis.histogram_go.load(Relaxed) {
            analysis.histogram_go.store(true, Relaxed);
        } else {
            // If histogram is already requested, check spectrum
            if !analysis.spectrum_go.load(Relaxed) {
                analysis.spectrum_go.store(true, Relaxed);
            }
        }
    }

    fn update_selected_metadata(&mut self, new_value: Option<Value>) {
        let Some(source) = &self.source else {
            return;
        };
        let Some(state) = &mut self.meta_tree_state else {
            return;
        };
        let Some(index) = state.list_state.borrow().selected() else {
            return;
        };
        let Some(item) = state.visible_items.get(index) else {
            return;
        };
        let root = &*state.data;
        let replace = &*item.info;
        let new_meta = clone_with_replacement(root, replace, new_value.as_ref()).unwrap();

        let mut data = source.lock().unwrap();
        match data.write_metadata(&new_meta).and_then(|_| data.metadata()) {
            Err(err) => {
                // Display error dialog
                self.dialog_type = Some(DialogType::Error(err.to_string()));
            }
            Ok(reloaded_meta) => {
                *state = TreeState::new(Arc::new(reloaded_meta).into());
                state.rebuild_visible_items();
            }
        }
    }

    fn get_selected_metadata_value_string(&self) -> Option<String> {
        let state = self.meta_tree_state.as_ref()?;
        let index = state.list_state.borrow().selected()?;
        let item = state.visible_items.get(index)?;

        // Convert value to a string that can be edited
        match &*item.info {
            Value::Null => Some("null".to_string()),
            Value::Bool(b) => Some(b.to_string()),
            Value::Number(n) => Some(n.to_string()),
            Value::String(s) => Some(s.clone()),
            Value::Array(_) | Value::Object(_) => None, // Can't edit complex types
        }
    }

    fn is_metadata_item_selected(&self) -> bool {
        let Some(state) = self.meta_tree_state.as_ref() else {
            return false;
        };
        state.list_state.borrow().selected().is_some()
    }

    fn parse_edit_draft(&self) -> Value {
        let draft = self.edit_draft.trim();

        // Keep as a string
        let force_string = (|| {
            let state = self.meta_tree_state.as_ref()?;
            let index = state.list_state.borrow().selected()?;
            let item = state.visible_items.get(index)?;
            Some(matches!(&*item.info, Value::String(_)))
        })();
        if force_string == Some(true) {
            return Value::String(draft.to_string());
        }

        // Try to parse as different types
        if draft == "null" {
            Value::Null
        } else if draft == "true" {
            Value::Bool(true)
        } else if draft == "false" {
            Value::Bool(false)
        } else if let Ok(num) = draft.parse::<i64>() {
            Value::Number(num.into())
        } else if let Some(num) = draft
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
        {
            Value::Number(num)
        } else {
            // Treat as string
            Value::String(draft.to_string())
        }
    }

    fn render_dialog(&self, f: &mut ratatui::Frame, area: Rect) {
        let Some(dialog_type) = &self.dialog_type else {
            return;
        };

        // Create a centered dialog
        let dialog_width = 60;
        let dialog_height = 7;
        let x = (area.width.saturating_sub(dialog_width)) / 2;
        let y = (area.height.saturating_sub(dialog_height)) / 2;

        let dialog_area = Rect {
            x: area.x + x,
            y: area.y + y,
            width: dialog_width.min(area.width),
            height: dialog_height.min(area.height),
        };

        // Clear the dialog area with a semi-transparent effect (using a filled block)
        f.render_widget(Clear, dialog_area);

        // Create dialog content
        let mut text = Text::default();
        let (title, border_color) = match dialog_type {
            DialogType::Edit => {
                text.push_line("Edit Value".bold().fg(Color::Yellow));
                text.push_line("");
                text.push_line(vec![
                    "Value: ".bold(),
                    self.edit_draft.clone().fg(Color::White),
                ]);
                text.push_line("");
                text.push_line("Enter: Confirm | Esc: Cancel".fg(Color::Gray));
                ("Metadata Editor", Color::Yellow)
            }
            DialogType::Delete => {
                text.push_line("Delete Value".bold().fg(Color::Red));
                text.push_line("");
                text.push_line("Are you sure you want to delete this value?".fg(Color::White));
                text.push_line("");
                text.push_line("Enter: Confirm | Esc: Cancel".fg(Color::Gray));
                ("Metadata Editor", Color::Yellow)
            }
            DialogType::Error(err) => {
                text.push_line("Error".bold().fg(Color::Red));
                text.push_line("");
                text.push_line(err.clone().fg(Color::White));
                text.push_line("");
                text.push_line("Enter/Esc: Close".fg(Color::Gray));
                ("Error", Color::Red)
            }
        };

        let dialog = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(title),
            )
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });

        f.render_widget(dialog, dialog_area);
    }
}

fn clone_with_replacement(value: &Value, replace: &Value, with: Option<&Value>) -> Option<Value> {
    if (value as *const Value) == (replace as *const Value) {
        return with.cloned();
    }
    Some(match value {
        Value::Array(values) => Value::Array(
            values
                .iter()
                .filter_map(|i| clone_with_replacement(i, replace, with))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .filter_map(|(k, v)| Some((k.clone(), clone_with_replacement(v, replace, with)?)))
                .collect(),
        ),
        _ => value.clone(),
    })
}

pub fn setup_terminal() -> Result<Terminal<Backend>, Error> {
    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

pub fn restore_terminal(terminal: &mut Terminal<Backend>) -> Result<(), Error> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}
