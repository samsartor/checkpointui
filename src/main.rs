use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, stdout, Stdout};
use ratatui::{
    backend::CrosstermBackend,
    crossterm::{
        event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    Terminal,
};
use safetensors::tensor::TensorInfo;

type Backend = CrosstermBackend<Stdout>;

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

struct App {
    should_quit: bool,
}

impl App {
    fn new() -> Self {
        Self {
            should_quit: false,
        }
    }

    fn handle_events(&mut self) -> io::Result<()> {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                _ => {}
            }
        }
        Ok(())
    }

    fn run(&mut self, terminal: &mut Terminal<Backend>) -> io::Result<()> {
        while !self.should_quit {
            terminal.draw(|f| {
                let area = f.area();
                f.render_widget(
                    ratatui::widgets::Paragraph::new("SafeTensors TUI Inspector\n\nPress 'q' or Esc to quit"),
                    area,
                );
            })?;

            self.handle_events()?;
        }
        Ok(())
    }
}

fn setup_terminal() -> io::Result<Terminal<Backend>> {
    let mut stdout = stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<Backend>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    let result = app.run(&mut terminal);
    restore_terminal(&mut terminal)?;
    result
}
