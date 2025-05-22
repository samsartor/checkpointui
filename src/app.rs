use anyhow::Error;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{Stdout, stdout};
use std::path::PathBuf;

use crate::model::SafeTensorsData;

pub type Backend = CrosstermBackend<Stdout>;

pub struct App {
    should_quit: bool,
    data: Option<SafeTensorsData>,
    file_path: Option<PathBuf>,
}

impl App {
    pub fn new() -> Self {
        Self { 
            should_quit: false,
            data: None,
            file_path: None,
        }
    }

    pub fn load_file(&mut self, file_path: PathBuf) -> Result<(), Error> {
        let data = SafeTensorsData::from_file(&file_path)?;
        self.data = Some(data);
        self.file_path = Some(file_path);
        Ok(())
    }

    pub fn handle_events(&mut self) -> Result<(), Error> {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                _ => {}
            }
        }
        Ok(())
    }

    pub fn run(&mut self, terminal: &mut Terminal<Backend>) -> Result<(), Error> {
        while !self.should_quit {
            terminal.draw(|f| {
                let area = f.area();
                let content = if let Some(data) = &self.data {
                    format!(
                        "SafeTensors TUI Inspector\n\nFile: {}\nTensors: {}\nParameters: {}\n\nPress 'q' or Esc to quit",
                        self.file_path.as_ref().unwrap().display(),
                        data.metadata.tensors().len(),
                        data.tree.params
                    )
                } else {
                    "SafeTensors TUI Inspector\n\nNo file loaded\n\nPress 'q' or Esc to quit".to_string()
                };
                f.render_widget(
                    ratatui::widgets::Paragraph::new(content),
                    area,
                );
            })?;

            self.handle_events()?;
        }
        Ok(())
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
