use anyhow::Error;
use human_format::{Formatter, Scales};
use owning_ref::ArcRef;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::text::{Line, Text};
use ratatui::widgets::Wrap;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Margin},
    style::{Color, Style},
    widgets::{
        Block, Borders, List, ListItem, ListState, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState, StatefulWidget,
    },
};
use serde_json::Value;
use std::collections::HashSet;
use std::io::{Stdout, stdout};
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use crate::model::{Key, ModuleInfo, SafeTensorsData, shorten_value};

pub type Backend = CrosstermBackend<Stdout>;

pub struct App {
    should_quit: bool,
    file_path: Option<PathBuf>,
    tree_state: Option<TreeState>,
    extra_metadata: Option<Value>,
    formatted_extra: String,
    count_formatter: Formatter,
    bytes_formatter: Formatter,
}

struct TreeState {
    current_path: Vec<Key>,
    path_history: Vec<Vec<Key>>,
    expanded: HashSet<Vec<Key>>,
    visible_items: Vec<TreeItem>,
    data: ArcRef<ModuleInfo>,
    list_state: ListState,
}

#[derive(Clone)]
struct TreeItem {
    path: Vec<Key>,
    name: String,
    depth: usize,
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
    fn new(data: ArcRef<ModuleInfo>) -> Self {
        Self {
            current_path: Vec::new(),
            path_history: Vec::new(),
            expanded: HashSet::new(),
            visible_items: Vec::new(),
            data,
            list_state: ListState::default(),
        }
    }

    fn rebuild_visible_items(&mut self) {
        self.visible_items.clear();

        // Navigate to current module
        let mut current_module = self.data.clone();
        for key in &self.current_path {
            if current_module.children.contains_key(key) {
                current_module = current_module.map(|m| &m.children[key]);
            } else {
                // Path no longer exists, reset to root
                self.current_path.clear();
                current_module = self.data.clone();
                break;
            }
        }

        self.build_visible_items(current_module, self.current_path.clone(), 0);
    }

    fn build_visible_items(&mut self, module: ArcRef<ModuleInfo>, path: Vec<Key>, depth: usize) {
        for key in module.children.keys() {
            let info = module.clone().map(|m| &m.children[key]);
            let mut item_path = path.clone();
            item_path.push(key.clone());

            let is_expanded = self.expanded.contains(&item_path);

            self.visible_items.push(TreeItem {
                path: item_path.clone(),
                name: key.to_string(),
                depth,
                is_expanded,
                info: info.clone(),
            });

            if is_expanded {
                self.build_visible_items(info, item_path, depth + 1);
            }
        }
    }

    fn toggle_expanded(&mut self) {
        let Some(index) = self.list_state.selected() else {
            return;
        };
        let Some(item) = self.visible_items.get(index) else {
            return;
        };
        if !item.has_children() {
            return;
        }
        if self.expanded.contains(&item.path) {
            self.expanded.remove(&item.path);
        } else {
            self.expanded.insert(item.path.clone());
        }
    }

    fn move_up(&mut self) {
        self.list_state.select_previous();
    }

    fn move_down(&mut self) {
        self.list_state.select_next();
    }

    fn move_right(&mut self) {
        let Some(index) = self.list_state.selected() else {
            return;
        };
        let Some(item) = self.visible_items.get(index) else {
            return;
        };
        if !item.has_children() {
            return;
        }
        let prev_path = mem::replace(&mut self.current_path, item.path.clone());
        self.path_history.push(prev_path);
        self.rebuild_visible_items();
        self.list_state.select(Some(0));
    }

    fn move_left(&mut self) {
        let goto_path = self.path_history.pop().unwrap_or_default();
        let prev_path = mem::replace(&mut self.current_path, goto_path);
        self.rebuild_visible_items();
        let index = self.visible_items.iter().position(|i| i.path == prev_path);
        self.list_state.select(index);
    }
}

impl App {
    pub fn new() -> Self {
        let mut count_formatter = Formatter::new();
        let mut count_scales = Scales::new();
        count_scales
            .with_base(1000)
            .with_suffixes(vec!["", "K", "M", "B", "T"]);
        count_formatter.with_separator("").with_scales(count_scales);
        let mut bytes_formatter = Formatter::new();
        bytes_formatter
            .with_scales(Scales::Binary())
            .with_units("B");
        Self {
            should_quit: false,
            file_path: None,
            tree_state: None,
            extra_metadata: None,
            formatted_extra: String::new(),
            count_formatter,
            bytes_formatter,
        }
    }

    pub fn load_file(&mut self, file_path: PathBuf) -> Result<(), Error> {
        let data = SafeTensorsData::from_file(&file_path)?;
        self.file_path = Some(file_path);
        let mut extra_metadata = data.parse_metadata();
        shorten_value(&mut extra_metadata);
        if let Ok(formatted) =
            colored_json::to_colored_json(&extra_metadata, colored_json::ColorMode::On)
        {
            self.formatted_extra = formatted;
        }
        self.extra_metadata = Some(extra_metadata);
        let mut state = TreeState::new(Arc::new(data.tree).into());
        state.rebuild_visible_items();
        self.tree_state = Some(state);
        Ok(())
    }

    pub fn handle_events(&mut self) -> Result<(), Error> {
        if let Event::Key(key) = event::read()? {
            match (key.code, &mut self.tree_state) {
                (KeyCode::Char('q') | KeyCode::Esc, _) => self.should_quit = true,
                (KeyCode::Up, Some(s)) => s.move_up(),
                (KeyCode::Down, Some(s)) => s.move_down(),
                (KeyCode::Left, Some(s)) => s.move_left(),
                (KeyCode::Right, Some(s)) => s.move_right(),
                (KeyCode::Char(' ') | KeyCode::Enter, Some(s)) => {
                    s.toggle_expanded();
                    s.rebuild_visible_items();
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn run(&mut self, terminal: &mut Terminal<Backend>) -> Result<(), Error> {
        while !self.should_quit {
            terminal.draw(|f| self.render_ui(f))?;
            self.handle_events()?;
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
            format!("SafeTensors Inspector - {}", path.display())
        } else {
            "SafeTensors Inspector - No file loaded".to_string()
        };

        let top_bar = Paragraph::new(title)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(top_bar, chunks[0]);

        // Main content area
        if self.tree_state.is_some() {
            let main_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50), // Tree panel
                    Constraint::Percentage(50), // Info panel
                ])
                .split(chunks[1]);

            self.render_tree_panel(f, main_chunks[0]);
            self.render_info_panel(f, main_chunks[1]);
        } else {
            let help_text = "No file loaded.\n\nUsage: checkpointui <safetensors_file>\n\nPress 'q' or Esc to quit";
            let help = Paragraph::new(help_text)
                .block(Block::default().borders(Borders::ALL).title("Help"))
                .style(Style::default().fg(Color::White));
            f.render_widget(help, chunks[1]);
        }

        // Bottom bar
        let help_text = if self.tree_state.is_some() {
            "↑/↓: Navigate | ←/→: Enter/Exit Module | Space/Enter: Expand/Collapse | q/Esc: Quit"
        } else {
            "q/Esc: Quit"
        };

        let bottom_bar = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Gray));
        f.render_widget(bottom_bar, chunks[2]);
    }

    fn render_tree_panel(&mut self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let Some(tree) = &self.tree_state else {
            return;
        };
        let items: Vec<ListItem> = tree
            .visible_items
            .iter()
            .map(|item| {
                let indent = "  ".repeat(item.depth);
                let icon = if item.has_children() {
                    if item.is_expanded { "▼ " } else { "▶ " }
                } else if item.is_tensor() {
                    "📄 "
                } else {
                    "  "
                };

                let text = format!(
                    "{}{}{} ({})",
                    indent,
                    icon,
                    item.name,
                    self.format_count(item.info.total_params),
                );
                ListItem::new(text)
            })
            .collect();

        let current_path_str = if tree.current_path.is_empty() {
            "Root".to_string()
        } else {
            tree.current_path
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(".")
        };

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!("Module Tree - {}", current_path_str)),
            )
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().bg(Color::Blue).fg(Color::White));

        let tree = self.tree_state.as_mut().unwrap();
        f.render_stateful_widget(list, area, &mut tree.list_state);
    }

    fn render_info_panel(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        use std::fmt::Write;

        let Some(tree) = &self.tree_state else { return };
        let selected_item = tree
            .list_state
            .selected()
            .and_then(|i| tree.visible_items.get(i));
        let mut info_text = String::new();
        if let Some(item) = selected_item {
            if let Some(tensor_info) = &item.info.tensor_info {
                writeln!(
                    &mut info_text,
                    "Tensor: {}\nShape: {:?}\nData Type: {:?}\nParameters: {}\nSize: {}",
                    item.info.full_name,
                    tensor_info.shape,
                    tensor_info.dtype,
                    self.format_count(item.info.total_params),
                    self.format_bytes(tensor_info.data_offsets.1 - tensor_info.data_offsets.0),
                )
                .unwrap();
            } else {
                writeln!(
                    &mut info_text,
                    "Module: {}\nTensors: {}\nParameters: {}",
                    item.info.full_name,
                    item.info.total_tensors,
                    self.format_count(item.info.total_params),
                )
                .unwrap();
            }
        }

        if !info_text.is_empty() {
            writeln!(&mut info_text).unwrap();
        }

        writeln!(
            &mut info_text,
            "File: {}\nTotal Tensors: {}\nTotal Parameters: {}",
            self.file_path.as_ref().unwrap().display(),
            tree.data.total_tensors,
            self.format_count(tree.data.total_params)
        )
        .unwrap();

        let mut info_text = Text::from(info_text);

        if self.extra_metadata.is_some() {
            if let Ok(mut text) = ansi_to_tui::IntoText::into_text(&self.formatted_extra) {
                if let Some(first) = text.lines.get_mut(0) {
                    first.spans.insert(0, "Metadata: ".into());
                    info_text.extend(text);
                }
            }
        }

        let info = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Information"))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });

        f.render_widget(info, area);
    }

    fn format_count(&self, count: usize) -> String {
        if count < 1000 {
            count.to_string()
        } else {
            self.count_formatter.format(count as f64)
        }
    }

    fn format_bytes(&self, bytes: usize) -> String {
        if bytes < 1000 {
            format!("{bytes} Bytes")
        } else {
            self.bytes_formatter.format(bytes as f64)
        }
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
