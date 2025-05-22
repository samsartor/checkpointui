mod app;
mod model;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "checkpointui")]
#[command(about = "TUI for inspecting safetensors files")]
struct Cli {
    #[arg(help = "Path to the safetensors file")]
    file_path: Option<PathBuf>,
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    
    let mut terminal = app::setup_terminal()?;
    let mut app = app::App::new();
    
    if let Some(file_path) = cli.file_path {
        if let Err(e) = app.load_file(file_path) {
            app::restore_terminal(&mut terminal)?;
            eprintln!("Error loading file: {}", e);
            return Err(e);
        }
    }
    
    let result = app.run(&mut terminal);
    app::restore_terminal(&mut terminal)?;
    result
}
