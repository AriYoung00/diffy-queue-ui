mod app;

fn main() {
    let application = app::GrapherApp::new();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(application), native_options);
}
