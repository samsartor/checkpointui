mod analysis;
mod app;
mod gguf;
mod model;
mod safetensors;
mod storage;

use clap::{CommandFactory as _, Parser};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "checkpointui")]
#[command(about = "TUI for inspecting safetensors files")]
struct Cli {
    #[arg(help = "Path to the safetensors file")]
    file_path: Option<PathBuf>,
    #[arg(
        help = "The character which separates modules in tensor paths",
        short = 'd',
        long,
        default_value_t = '.'
    )]
    module_delim: char,
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();

    let mut app = app::App::new();
    app.helptext = Cli::command().render_long_help().to_string();
    app.path_split = model::PathSplit::Delim(cli.module_delim);

    if let Some(file_path) = cli.file_path {
        if let Err(e) = app.load_file(file_path) {
            eprintln!("Error loading file: {}", e);
            return Err(e);
        }
    }

    let mut terminal = app::setup_terminal()?;
    let result = app.run(&mut terminal);
    app::restore_terminal(&mut terminal)?;
    result
}
