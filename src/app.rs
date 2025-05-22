use anyhow::Error;
use owning_ref::ArcRef;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use std::collections::HashSet;
use std::io::{Stdout, stdout};
use std::path::PathBuf;
use std::sync::Arc;

use crate::model::{Key, Metadata, ModuleInfo, SafeTensorsData};

pub type Backend = CrosstermBackend<Stdout>;

pub struct App {
    should_quit: bool,
    file_path: Option<PathBuf>,
    tree_state: Option<TreeState>,
    safetensors_metadata: Option<Metadata>,
}

struct TreeState {
    current_path: Vec<Key>,
    expanded: HashSet<Vec<Key>>,
    selected_index: usize,
    visible_items: Vec<TreeItem>,
    data: ArcRef<ModuleInfo>,
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
            expanded: HashSet::new(),
            selected_index: 0,
            visible_items: Vec::new(),
            data,
        }
    }

    fn rebuild_visible_items(&mut self) {
        self.visible_items.clear();
        self.build_visible_items(self.data.clone(), Vec::new(), 0);

        if self.selected_index >= self.visible_items.len() {
            self.selected_index = self.visible_items.len().saturating_sub(1);
        }
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
        if let Some(item) = self.visible_items.get(self.selected_index) {
            if item.has_children() {
                if self.expanded.contains(&item.path) {
                    self.expanded.remove(&item.path);
                } else {
                    self.expanded.insert(item.path.clone());
                }
            }
        }
    }

    fn move_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        }
    }

    fn move_down(&mut self) {
        if self.selected_index < self.visible_items.len().saturating_sub(1) {
            self.selected_index += 1;
        }
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            should_quit: false,
            file_path: None,
            tree_state: None,
            safetensors_metadata: None,
        }
    }

    pub fn load_file(&mut self, file_path: PathBuf) -> Result<(), Error> {
        let SafeTensorsData { metadata, tree, .. } = SafeTensorsData::from_file(&file_path)?;
        self.file_path = Some(file_path);
        self.safetensors_metadata = Some(metadata);
        let mut state = TreeState::new(Arc::new(tree).into());
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

    fn render_ui(&self, f: &mut ratatui::Frame) {
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
            "↑/↓: Navigate | Space/Enter: Expand/Collapse | q/Esc: Quit"
        } else {
            "q/Esc: Quit"
        };

        let bottom_bar = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Gray));
        f.render_widget(bottom_bar, chunks[2]);
    }

    fn render_tree_panel(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let Some(tree) = &self.tree_state else { return };
        let items: Vec<ListItem> = tree
            .visible_items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let indent = "  ".repeat(item.depth);
                let icon = if item.has_children() {
                    if item.is_expanded { "▼ " } else { "▶ " }
                } else if item.is_tensor() {
                    "📄 "
                } else {
                    "  "
                };

                let style = if i == tree.selected_index {
                    Style::default().bg(Color::Blue).fg(Color::White)
                } else {
                    Style::default().fg(Color::White)
                };

                let text = format!(
                    "{}{}{} ({})",
                    indent,
                    icon,
                    item.name,
                    human_format::Formatter::new().format(item.info.total_params as f64)
                );
                ListItem::new(text).style(style)
            })
            .collect();

        let tree = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Module Tree"))
            .style(Style::default().fg(Color::White));

        f.render_widget(tree, area);
    }

    fn render_info_panel(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        let Some(tree) = &self.tree_state else { return };
        let selected_item = tree.visible_items.get(tree.selected_index);
        let info_text = if let Some(item) = selected_item {
            if let Some(tensor_info) = &item.info.tensor_info {
                format!(
                    "Tensor: {}\n\nShape: {:?}\nData Type: {:?}\nParameters: {}\nSize: {} bytes",
                    item.info.full_name,
                    tensor_info.shape,
                    tensor_info.dtype,
                    human_format::Formatter::new().format(item.info.total_params as f64),
                    tensor_info.data_offsets.1 - tensor_info.data_offsets.0
                )
            } else {
                format!(
                    "Module: {}\nParameters: {}",
                    item.info.full_name,
                    human_format::Formatter::new().format(item.info.total_params as f64)
                )
            }
        } else {
            format!(
                "File: {}\nTotal Tensors: {}\nTotal Parameters: {}",
                self.file_path.as_ref().unwrap().display(),
                tree.data.total_tensors,
                human_format::Formatter::new().format(tree.data.total_params as f64)
            )
        };

        let info = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Information"))
            .style(Style::default().fg(Color::White));

        f.render_widget(info, area);
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
