mod app;
mod model;

fn main() -> Result<(), anyhow::Error> {
    let mut terminal = app::setup_terminal()?;
    let mut app = app::App::new();
    let result = app.run(&mut terminal);
    app::restore_terminal(&mut terminal)?;
    result
}
