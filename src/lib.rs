#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::SolverApp;
use egui_web::{resize_canvas_to_screen_size, console_log};

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::{self, prelude::*};
use eframe::egui::Vec2;
use eframe::epi::App;

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start(canvas_id: &str) -> Result<(), eframe::wasm_bindgen::JsValue> {
    let app = SolverApp::new();
    resize_canvas_to_screen_size(canvas_id, Vec2::new(1400., 1100.));
    eframe::start_web(canvas_id, Box::new(app))
}