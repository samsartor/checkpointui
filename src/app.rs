use anyhow::{Error, bail};
use async_cell::sync::AsyncCell;
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
    Block, Borders, List, ListItem, ListState, Paragraph, StatefulWidget, Wrap,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashSet;
use std::io::{Stdout, stdout};
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use weakref::{Own, refer};

use crate::analysis::{Analysis, AnalysisCell, Req, start_analysis_thread};
use crate::gguf::Gguf;
use crate::model::{Key, ModuleInfo, ModuleSource, PathSplit, TensorInfo, shorten_value};
use crate::safetensors::Safetensors;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
enum Panel {
    #[default]
    Tree,
    SelectedInfo,
    FileInfo,
    Analysis,
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
    tree_state: Option<TreeState>,
    extra_metadata: Option<Value>,
    source: Option<Box<dyn ModuleSource + Send>>,
    formatted_extra: String,
    count_formatter: Formatter,
    bytes_formatter: Formatter,
    selected_panel: Panel,
    pub helptext: String,
    pub path_split: PathSplit,
    analysis_sender: Option<Own<Box<AnalysisCell>>>,
    current_analysis: Option<Own<Box<Analysis>>>,
}

struct TreeState {
    data: ArcRef<ModuleInfo>,
    data_history: Vec<ArcRef<ModuleInfo>>,
    expanded: HashSet<Key>,
    visible_items: Vec<TreeItem>,
    list_state: RefCell<ListState>,
}

#[derive(Clone)]
struct TreeItem {
    name: String,
    depth: i32,
    is_expanded: bool,
    info: ArcRef<ModuleInfo>,
}

impl TreeItem {
    pub fn has_children(&self) -> bool {
        !self.info.children.is_empty()
    }

    pub fn is_tensor(&self) -> bool {
        self.info.tensor_info.is_some()
    }
}

impl TreeState {
    fn new(root: ArcRef<ModuleInfo>) -> Self {
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
            let is_expanded = depth < 0 || self.expanded.contains(&info.full_name);
            if is_expanded {
                let stack_at = stack.len();
                for key in info.children.keys() {
                    let child = info.clone().map(|m| &m.children[key]);
                    stack.push((child, key.to_string(), depth + 1));
                }
                stack[stack_at..].sort_by(|(a_info, a_name, ..), (b_info, b_name, ..)| {
                    a_info
                        .tensor_info
                        .is_some()
                        .cmp(&b_info.tensor_info.is_some())
                        .then(natural_lexical_cmp(b_name, a_name))
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
        if self.expanded.contains(&item.info.full_name) {
            self.expanded.remove(&item.info.full_name);
        } else {
            self.expanded.insert(item.info.full_name.clone());
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
        this
    }

    pub fn load_file(&mut self, file_path: PathBuf) -> Result<(), Error> {
        let ext = file_path.extension().and_then(|ext| ext.to_str());
        if ext == Some("safetensors") {
            self.source = Some(Box::new(Safetensors::open_file(&file_path)?));
        } else if ext == Some("gguf") {
            self.source = Some(Box::new(Gguf::open_file(&file_path)?));
        } else {
            bail!("could not infer file type");
        }
        self.file_path = Some(file_path);
        self.rebuild_module()
    }

    pub fn rebuild_module(&mut self) -> Result<(), Error> {
        let Some(data) = &mut self.source else {
            return Ok(());
        };
        let mut module = data.module(&self.path_split)?;
        module.flatten_single_children();
        let mut extra_metadata = data.metadata()?;
        shorten_value(&mut extra_metadata);
        if let Ok(formatted) =
            colored_json::to_colored_json(&extra_metadata, colored_json::ColorMode::On)
        {
            self.formatted_extra = formatted;
        }
        self.extra_metadata = Some(extra_metadata);
        let mut state = TreeState::new(Arc::new(module).into());
        state.rebuild_visible_items();
        self.tree_state = Some(state);

        // Now that we have the tree, move the source to the analysis thread
        let source = self.source.take().unwrap();
        let sender = self
            .analysis_sender
            .insert(Own::new_box(AnalysisCell::new()))
            .refer();
        start_analysis_thread(source, sender);

        // Start analysis for the initially selected tensor
        self.update_analysis_for_selected_tensor();
        Ok(())
    }

    pub fn handle_events(&mut self) -> Result<(), Error> {
        if let Event::Key(key) = event::read()? {
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
                        Constraint::Percentage(50), // Selected item info
                        Constraint::Percentage(50), // File info
                    ])
                    .split(main_chunks[1]);

                self.render_selected_info_panel(f, info_chunks[0]);
                self.render_file_info_panel(f, info_chunks[1]);
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
                        Constraint::Percentage(50), // Selected item info
                        Constraint::Percentage(50), // File info
                    ])
                    .split(main_chunks[1]);

                self.render_selected_info_panel(f, info_chunks[0]);
                self.render_file_info_panel(f, info_chunks[1]);
            }
        } else {
            let help = Paragraph::new(self.helptext.as_str())
                .block(Block::default().borders(Borders::ALL).title("Help"))
                .style(Style::default().fg(Color::White));
            f.render_widget(help, chunks[1]);
        }

        // Bottom bar
        let help_text = if self.tree_state.is_some() {
            "↑/↓: Navigate | ←/→: Enter/Exit Module | Space/Enter: Expand/Collapse | Tab/Shift+Tab: Switch Panel | q/Esc: Quit"
        } else {
            "q/Esc: Quit"
        };

        let bottom_bar = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Gray));
        f.render_widget(bottom_bar, chunks[2]);
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
                } else if item.is_tensor() {
                    "📄 "
                } else {
                    "  "
                }
                .into();
                spans.push(icon_span);

                // Name
                let name_span = if item.is_tensor() {
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
            title += tree.data.full_name.fg(MODULE_FG)
        };

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

    fn render_file_info_panel(&self, f: &mut ratatui::Frame, area: Rect) {
        let Some(tree) = &self.tree_state else { return };

        let mut text = Text::default();

        // File info section
        text.push_line(vec![
            "Path: ".bold(),
            self.file_path
                .as_ref()
                .unwrap()
                .display()
                .to_string()
                .fg(TENSOR_FG),
        ]);
        text.push_line(vec![
            "Total Tensors: ".bold(),
            tree.data.total_tensors.to_string().fg(COUNT_FG),
        ]);
        text.push_line(vec![
            "Total Parameters: ".bold(),
            self.format_count(tree.data.total_params).fg(COUNT_FG),
        ]);

        // Add metadata section if available
        if self.extra_metadata.is_some() {
            if let Ok(mut metadata_text) = ansi_to_tui::IntoText::into_text(&self.formatted_extra) {
                if let Some(first) = metadata_text.lines.get_mut(0) {
                    first.spans.insert(0, "Metadata: ".bold());
                    text.extend(metadata_text);
                }
            }
        }

        let info = Paragraph::new(text)
            .block(self.format_block("File Info", Panel::FileInfo))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });

        f.render_widget(info, area);
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
            .is_some_and(|item| item.is_tensor())
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
            let placeholder = Paragraph::new("Spectrum only available for 2D tensors")
                .block(self.format_block("Spectrum (SVD)", Panel::Analysis))
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
                format!(" ({})", count).into(),
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

        match analysis.histogram.try_get() {
            Some(Req::Completed(histogram)) => {
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
            Some(Req::Requested) => {
                text.push_line(vec!["🔄 Computing histogram...".fg(Color::Yellow)]);
            }
            None => {
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

        match analysis.spectrum.try_get() {
            Some(Req::Completed(spectrum)) => {
                text.push_line(Line::from(""));

                let chart_lines = Self::render_bar_chart(
                    &spectrum.chart,
                    30, // max_width
                    Color::Blue,
                    |x| format!("{x:6.2}"),
                );
                text.extend(chart_lines);
            }
            Some(Req::Requested) => {
                text.push_line(vec!["🔄 Computing SVD decomposition...".fg(Color::Yellow)]);
            }
            None => {
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
            .block(self.format_block("Singular Values", Panel::Analysis))
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
        let analysis = Own::new(Box::new(Analysis {
            tensor: tensor_info.clone(),
            histogram: AsyncCell::new_with(Req::Requested),
            spectrum: AsyncCell::new_with(Req::Requested),
            error: std::sync::OnceLock::new(),
            max_bin_count: 20,
        }));
        if let Some(sender) = self.analysis_sender.as_ref() {
            sender.set(analysis.refer());
        }
        self.current_analysis = Some(analysis);
    }
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
