//! Desktop viewer entry point.
//!
//! The eframe application is wired to `System` once the tracking thread and
//! snapshot transport are available.

fn main() -> eframe::Result {
    let options = eframe::NativeOptions::default();
    eframe::run_simple_native("ORB-SLAM3-RS", options, |ctx, _frame| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("ORB-SLAM3-RS");
            ui.label("Viewer scaffold is ready; waiting for the SLAM runtime.");
        });
    })
}
