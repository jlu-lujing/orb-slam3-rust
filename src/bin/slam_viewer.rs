//! ORB-SLAM3-RS desktop viewer.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use clap::{Arg, ArgAction, Command};
use eframe::egui;
use nalgebra::Vector3;
use orb_slam3_rs::euroc::{EurocRunStats, EurocSequence, run_sequences};
use orb_slam3_rs::system::{Sensor, System};
use orb_slam3_rs::viewer::{ViewerImage, ViewerSnapshot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let matches = Command::new("slam_viewer")
        .about("Run ORB-SLAM3-RS stereo-inertial EuRoC with the desktop viewer")
        .arg(Arg::new("vocabulary").required(true).index(1))
        .arg(Arg::new("settings").required(true).index(2))
        .arg(
            Arg::new("sequences")
                .required(true)
                .num_args(2..)
                .index(3)
                .trailing_var_arg(false),
        )
        .arg(
            Arg::new("realtime")
                .long("realtime")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let vocabulary = PathBuf::from(matches.get_one::<String>("vocabulary").expect("required"));
    let settings = PathBuf::from(matches.get_one::<String>("settings").expect("required"));
    let paths: Vec<_> = matches
        .get_many::<String>("sequences")
        .expect("required")
        .map(String::as_str)
        .collect();
    if paths.len() % 2 != 0 {
        return Err("sequence paths and timestamp files must be paired".into());
    }
    let sequences = paths
        .chunks_exact(2)
        .map(|pair| EurocSequence::load(pair[0], pair[1]))
        .collect::<Result<Vec<_>, _>>()?;
    let system = System::new(&vocabulary, &settings, Sensor::IMUStereo, true, None)?;
    let result = Arc::new(Mutex::new(None));
    let worker_system = system.clone();
    let worker_result = result.clone();
    let realtime = matches.get_flag("realtime");
    let worker = std::thread::Builder::new()
        .name("euroc-gui-runner".to_owned())
        .spawn(move || {
            let value = run_sequences(worker_system, &sequences, realtime);
            *worker_result.lock().unwrap() = Some(value);
        })?;

    let app = ViewerApp {
        system,
        worker: Some(worker),
        result,
        snapshot: None,
        texture: None,
        last_snapshot: Instant::now() - Duration::from_secs(1),
        last_image: Instant::now() - Duration::from_secs(1),
        camera: MapCamera::default(),
        show_points: true,
        show_keyframes: true,
        show_graph: true,
        follow_camera: true,
        localization_mode: false,
        paused: false,
        save_status: None,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native("ORB-SLAM3-RS", options, Box::new(|_cc| Ok(Box::new(app))))?;
    Ok(())
}

struct ViewerApp {
    system: Arc<System>,
    worker: Option<JoinHandle<()>>,
    result: Arc<Mutex<Option<Result<EurocRunStats, orb_slam3_rs::euroc::EurocError>>>>,
    snapshot: Option<ViewerSnapshot>,
    texture: Option<egui::TextureHandle>,
    last_snapshot: Instant,
    last_image: Instant,
    camera: MapCamera,
    show_points: bool,
    show_keyframes: bool,
    show_graph: bool,
    follow_camera: bool,
    localization_mode: bool,
    paused: bool,
    save_status: Option<String>,
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.last_snapshot.elapsed() >= Duration::from_millis(33) {
            self.snapshot = self.system.viewer_snapshot();
            self.last_snapshot = Instant::now();
            ui.ctx().request_repaint_after(Duration::from_millis(33));
        }

        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(245.0)
            .show(ui, |ui| self.controls(ui));

        egui::Panel::right("current_frame")
            .resizable(true)
            .default_size(390.0)
            .size_range(260.0..=720.0)
            .show(ui, |ui| self.image_panel(ui));

        egui::CentralPanel::default().show(ui, |ui| self.map_panel(ui));
        self.update_texture(ui.ctx());
    }
}

impl ViewerApp {
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("ORB-SLAM3-RS");
        ui.add_space(8.0);
        let state = self
            .snapshot
            .as_ref()
            .map(|snapshot| format!("{:?}", snapshot.state))
            .unwrap_or_else(|| "No frame".to_owned());
        ui.label(format!("Tracking: {state}"));

        if let Some(snapshot) = &self.snapshot {
            ui.label(format!("Map points: {}", snapshot.map_point_count));
            ui.label(format!("Keyframes: {}", snapshot.keyframes.len()));
            ui.label(format!("Features: {}", snapshot.feature_count));
            ui.label(format!("Inliers: {}", snapshot.inlier_count));
            ui.label(format!("IMU initialized: {}", snapshot.imu_initialized));
            if snapshot.map_changed {
                ui.colored_label(egui::Color32::LIGHT_GREEN, "Map updated");
            }
        }
        if let Some(result) = self.result.lock().unwrap().as_ref() {
            match result {
                Ok(stats) => {
                    ui.label(format!("Images: {}", stats.images));
                    ui.label(format!("Mean track: {:.2} ms", stats.mean_tracking_ms));
                }
                Err(error) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("{error:?}"));
                }
            }
        } else {
            ui.label("Sequence running...");
        }

        ui.separator();
        ui.checkbox(&mut self.show_points, "Show map points");
        ui.checkbox(&mut self.show_keyframes, "Show keyframes");
        ui.checkbox(&mut self.show_graph, "Show covisibility graph");
        ui.checkbox(&mut self.follow_camera, "Follow camera");

        if ui
            .checkbox(&mut self.localization_mode, "Localization mode")
            .changed()
        {
            if self.localization_mode {
                self.system.activate_localization_mode();
            } else {
                self.system.deactivate_localization_mode();
            }
        }

        ui.separator();
        if ui.checkbox(&mut self.paused, "Pause tracking").changed() {
            if self.paused {
                self.system.pause_tracking();
            } else {
                self.system.resume_tracking();
            }
        }
        if ui
            .add_enabled(self.paused, egui::Button::new("Step one frame"))
            .clicked()
        {
            self.system.step_tracking();
        }
        if ui.button("Reset active map").clicked() {
            self.system.reset_active_map();
        }
        if ui.button("Reset session").clicked() {
            self.system.reset();
        }
        if ui.button("Save trajectory").clicked() {
            self.save_status = Some(
                self.system
                    .save_trajectory_euroc("CameraTrajectory.txt")
                    .and_then(|_| {
                        self.system
                            .save_keyframe_trajectory_euroc("KeyFrameTrajectory.txt")
                    })
                    .and_then(|_| {
                        self.system
                            .save_atlas("Atlas")
                            .map_err(std::io::Error::other)
                    })
                    .map(|_| "Saved trajectory and Atlas".to_owned())
                    .unwrap_or_else(|error| format!("Save failed: {error}")),
            );
        }
        if ui.button("Stop").clicked() {
            self.system.shutdown();
        }
        if let Some(status) = &self.save_status {
            ui.label(status);
        }
        ui.add_space(8.0);
        ui.small("Drag: orbit\nScroll: zoom");
    }

    fn image_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Current Frame");
        ui.separator();
        if let Some(texture) = &self.texture {
            let max = ui.available_size();
            let aspect = texture.size_vec2().x / texture.size_vec2().y.max(1.0);
            let width = max.x.min(max.y * aspect);
            let size = egui::vec2(width, width / aspect);
            let response = ui.add(egui::Image::new(egui::load::SizedTexture::new(
                texture.id(),
                size,
            )));
            if let Some(snapshot) = &self.snapshot
                && let Some(image) = &snapshot.image
            {
                let scale = egui::vec2(
                    response.rect.width() / image.width.max(1) as f32,
                    response.rect.height() / image.height.max(1) as f32,
                );
                let painter = ui.painter();
                for feature in &snapshot.features {
                    let position =
                        response.rect.min + egui::vec2(feature.x * scale.x, feature.y * scale.y);
                    let color = if feature.outlier {
                        egui::Color32::RED
                    } else if feature.has_map_point {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::YELLOW
                    };
                    painter.circle_stroke(position, 2.0, egui::Stroke::new(1.0, color));
                }
            }
        } else {
            ui.centered_and_justified(|ui| ui.label("Waiting for image data"));
        }
    }

    fn map_panel(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click_and_drag());
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(7, 11, 18));

        if response.dragged() {
            let delta = response.drag_delta();
            self.camera.yaw += delta.x * 0.008;
            self.camera.pitch = (self.camera.pitch + delta.y * 0.008).clamp(-1.45, 1.45);
            self.follow_camera = false;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.camera.distance =
                    (self.camera.distance * (-scroll * 0.002).exp()).clamp(0.05, 100_000.0);
            }
        }

        let Some(snapshot) = &self.snapshot else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Waiting for SLAM data",
                egui::FontId::proportional(18.0),
                egui::Color32::GRAY,
            );
            return;
        };

        if self.follow_camera
            && let Some(pose) = snapshot.current_pose
        {
            self.camera.target = pose.inverse().translation.vector;
        }
        let project = |point: Vector3<f32>| {
            self.camera
                .project(point, rect, snapshot.current_pose.as_ref())
        };
        let point_color = egui::Color32::from_rgb(95, 185, 255);
        if self.show_points {
            let stride = (snapshot.map_points.len() / 50_000).max(1);
            for point in snapshot.map_points.iter().step_by(stride) {
                if let Some(position) = project(*point) {
                    painter.circle_filled(position, 1.15, point_color);
                }
            }
        }

        let mut keyframe_positions = HashMap::new();
        for keyframe in &snapshot.keyframes {
            let center = keyframe.pose.inverse().translation.vector;
            keyframe_positions.insert(keyframe.id, center);
            if self.show_keyframes
                && let Some(position) = project(center)
            {
                painter.circle_filled(position, 3.0, egui::Color32::YELLOW);
            }
        }
        if self.show_graph {
            for keyframe in &snapshot.keyframes {
                let Some(&source) = keyframe_positions.get(&keyframe.id) else {
                    continue;
                };
                for connected in &keyframe.connections {
                    if connected <= &keyframe.id {
                        continue;
                    }
                    let Some(&target) = keyframe_positions.get(connected) else {
                        continue;
                    };
                    if let (Some(a), Some(b)) = (project(source), project(target)) {
                        painter.line_segment(
                            [a, b],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 85, 110)),
                        );
                    }
                }
            }
        }

        if let Some(pose) = snapshot.current_pose {
            let inverse = pose.inverse();
            let center = inverse.translation.vector;
            let rotation = inverse.rotation.to_rotation_matrix().into_inner();
            let axis_length = (self.camera.distance * 0.04).clamp(0.01, 1.5);
            for (axis, color) in [
                (rotation.column(0).into_owned(), egui::Color32::RED),
                (rotation.column(1).into_owned(), egui::Color32::GREEN),
                (rotation.column(2).into_owned(), egui::Color32::BLUE),
            ] {
                if let (Some(a), Some(b)) = (project(center), project(center + axis * axis_length))
                {
                    painter.line_segment([a, b], egui::Stroke::new(2.0, color));
                }
            }
        }

        painter.text(
            rect.left_top() + egui::vec2(12.0, 10.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Map  |  {} points  |  {} keyframes",
                snapshot.map_point_count,
                snapshot.keyframes.len()
            ),
            egui::FontId::monospace(12.0),
            egui::Color32::LIGHT_GRAY,
        );
    }

    fn update_texture(&mut self, context: &egui::Context) {
        if self.last_image.elapsed() < Duration::from_millis(100) {
            return;
        }
        let Some(image) = self
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.image.as_ref())
        else {
            return;
        };
        self.texture = Some(image_texture(context, image));
        self.last_image = Instant::now();
    }
}

impl Drop for ViewerApp {
    fn drop(&mut self) {
        self.system.shutdown();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn image_texture(context: &egui::Context, image: &ViewerImage) -> egui::TextureHandle {
    let color = egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.rgba);
    context.load_texture("current-frame", color, egui::TextureOptions::LINEAR)
}

#[derive(Debug, Clone)]
struct MapCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vector3<f32>,
}

impl Default for MapCamera {
    fn default() -> Self {
        Self {
            yaw: 0.65,
            pitch: 0.45,
            distance: 8.0,
            target: Vector3::zeros(),
        }
    }
}

impl MapCamera {
    fn project(
        &self,
        point: Vector3<f32>,
        rect: egui::Rect,
        _pose: Option<&nalgebra::Isometry3<f32>>,
    ) -> Option<egui::Pos2> {
        let relative = point - self.target;
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let x = cos_yaw * relative.x - sin_yaw * relative.z;
        let z0 = sin_yaw * relative.x + cos_yaw * relative.z;
        let y = cos_pitch * relative.y - sin_pitch * z0;
        let z = sin_pitch * relative.y + cos_pitch * z0 + self.distance;
        if z <= 0.02 || !z.is_finite() {
            return None;
        }
        let focal = rect.width().min(rect.height()) * 1.15;
        let position = egui::pos2(
            rect.center().x + focal * x / z,
            rect.center().y - focal * y / z,
        );
        rect.contains(position).then_some(position)
    }
}
