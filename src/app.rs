use anyhow::Error;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::{Terminal, backend::CrosstermBackend, layout::{Constraint, Direction, Layout}, widgets::{Block, Borders, List, ListItem, Paragraph}, style::{Color, Style}};
use std::io::{Stdout, stdout};
use std::path::PathBuf;

use crate::model::{SafeTensorsData, ModuleInfo, Key};

pub type Backend = CrosstermBackend<Stdout>;

pub struct App {
    should_quit: bool,
    data: Option<SafeTensorsData>,
    file_path: Option<PathBuf>,
    tree_state: TreeState,
}

struct TreeState {
    current_path: Vec<Key>,
    expanded: Vec<Vec<Key>>,
    selected_index: usize,
    visible_items: Vec<TreeItem>,
}

#[derive(Clone)]
struct TreeItem {
    path: Vec<Key>,
    name: String,
    depth: usize,
    is_expanded: bool,
    has_children: bool,
    params: usize,
    is_tensor: bool,
}

impl TreeState {
    fn new() -> Self {
        Self {
            current_path: Vec::new(),
            expanded: Vec::new(),
            selected_index: 0,
            visible_items: Vec::new(),
        }
    }

    fn rebuild_visible_items(&mut self, root: &ModuleInfo) {
        self.visible_items.clear();
        self.build_visible_items(root, Vec::new(), 0);
        
        if self.selected_index >= self.visible_items.len() {
            self.selected_index = self.visible_items.len().saturating_sub(1);
        }
    }

    fn build_visible_items(&mut self, module: &ModuleInfo, path: Vec<Key>, depth: usize) {
        for (key, child) in &module.children {
            let mut item_path = path.clone();
            item_path.push(key.clone());
            
            let is_expanded = self.expanded.contains(&item_path);
            let has_children = !child.children.is_empty();
            let is_tensor = child.tensor_info.is_some();
            
            self.visible_items.push(TreeItem {
                path: item_path.clone(),
                name: key.to_string(),
                depth,
                is_expanded,
                has_children,
                params: child.params,
                is_tensor,
            });
            
            if is_expanded {
                self.build_visible_items(child, item_path, depth + 1);
            }
        }
    }

    fn toggle_expanded(&mut self) {
        if let Some(item) = self.visible_items.get(self.selected_index) {
            if item.has_children {
                let path = item.path.clone();
                if let Some(pos) = self.expanded.iter().position(|p| *p == path) {
                    self.expanded.remove(pos);
                } else {
                    self.expanded.push(path);
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
            data: None,
            file_path: None,
            tree_state: TreeState::new(),
        }
    }

    pub fn load_file(&mut self, file_path: PathBuf) -> Result<(), Error> {
        let data = SafeTensorsData::from_file(&file_path)?;
        self.tree_state.rebuild_visible_items(&data.tree);
        self.data = Some(data);
        self.file_path = Some(file_path);
        Ok(())
    }

    pub fn handle_events(&mut self) -> Result<(), Error> {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                KeyCode::Up => self.tree_state.move_up(),
                KeyCode::Down => self.tree_state.move_down(),
                KeyCode::Char(' ') | KeyCode::Enter => {
                    self.tree_state.toggle_expanded();
                    if let Some(data) = &self.data {
                        self.tree_state.rebuild_visible_items(&data.tree);
                    }
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
        if let Some(data) = &self.data {
            let main_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50), // Tree panel
                    Constraint::Percentage(50), // Info panel
                ])
                .split(chunks[1]);

            self.render_tree_panel(f, main_chunks[0]);
            self.render_info_panel(f, main_chunks[1], data);
        } else {
            let help_text = "No file loaded.\n\nUsage: checkpointui <safetensors_file>\n\nPress 'q' or Esc to quit";
            let help = Paragraph::new(help_text)
                .block(Block::default().borders(Borders::ALL).title("Help"))
                .style(Style::default().fg(Color::White));
            f.render_widget(help, chunks[1]);
        }

        // Bottom bar
        let help_text = if self.data.is_some() {
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
        let items: Vec<ListItem> = self.tree_state.visible_items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let indent = "  ".repeat(item.depth);
                let icon = if item.has_children {
                    if item.is_expanded { "▼ " } else { "▶ " }
                } else if item.is_tensor {
                    "📄 "
                } else {
                    "  "
                };
                
                let style = if i == self.tree_state.selected_index {
                    Style::default().bg(Color::Blue).fg(Color::White)
                } else {
                    Style::default().fg(Color::White)
                };
                
                let text = format!("{}{}{} ({})", indent, icon, item.name, human_format::Formatter::new().format(item.params as f64));
                ListItem::new(text).style(style)
            })
            .collect();

        let tree = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Module Tree"))
            .style(Style::default().fg(Color::White));
        
        f.render_widget(tree, area);
    }

    fn render_info_panel(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect, data: &SafeTensorsData) {
        let selected_item = self.tree_state.visible_items.get(self.tree_state.selected_index);
        
        let info_text = if let Some(item) = selected_item {
            if item.is_tensor {
                // Find the tensor info by following the path
                if let Some(tensor_info) = self.get_tensor_info_at_path(&data.tree, &item.path) {
                    format!(
                        "Tensor: {}\n\nShape: {:?}\nData Type: {:?}\nParameters: {}\nSize: {} bytes",
                        item.name,
                        tensor_info.shape,
                        tensor_info.dtype,
                        human_format::Formatter::new().format(item.params as f64),
                        tensor_info.data_offsets.1 - tensor_info.data_offsets.0
                    )
                } else {
                    format!("Module: {}\nParameters: {}", item.name, human_format::Formatter::new().format(item.params as f64))
                }
            } else {
                format!("Module: {}\nParameters: {}", item.name, human_format::Formatter::new().format(item.params as f64))
            }
        } else {
            format!(
                "File: {}\nTotal Tensors: {}\nTotal Parameters: {}",
                self.file_path.as_ref().unwrap().display(),
                data.metadata.tensors().len(),
                human_format::Formatter::new().format(data.tree.params as f64)
            )
        };

        let info = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Information"))
            .style(Style::default().fg(Color::White));
        
        f.render_widget(info, area);
    }

    fn get_tensor_info_at_path<'a>(&self, module: &'a ModuleInfo, path: &[Key]) -> Option<&'a safetensors::tensor::TensorInfo> {
        if path.is_empty() {
            return module.tensor_info.as_ref();
        }
        
        let key = &path[0];
        if let Some(child) = module.children.get(key) {
            self.get_tensor_info_at_path(child, &path[1..])
        } else {
            None
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
