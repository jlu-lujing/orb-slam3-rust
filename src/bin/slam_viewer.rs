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
        vocabulary,
        settings,
        worker: Some(worker),
        result,
        snapshot: None,
        texture: None,
        texture_right: None,
        last_snapshot: Instant::now() - Duration::from_secs(1),
        last_image: Instant::now() - Duration::from_secs(1),
        camera: MapCamera::default(),
        show_points: true,
        show_keyframes: true,
        show_graph: true,
        show_inertial_graph: true,
        follow_camera: true,
        localization_mode: false,
        paused: false,
        drag_hover: false,
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
    vocabulary: PathBuf,
    settings: PathBuf,
    worker: Option<JoinHandle<()>>,
    result: Arc<Mutex<Option<Result<EurocRunStats, orb_slam3_rs::euroc::EurocError>>>>,
    snapshot: Option<ViewerSnapshot>,
    texture: Option<egui::TextureHandle>,
    texture_right: Option<egui::TextureHandle>,
    last_snapshot: Instant,
    last_image: Instant,
    camera: MapCamera,
    show_points: bool,
    show_keyframes: bool,
    show_graph: bool,
    show_inertial_graph: bool,
    follow_camera: bool,
    localization_mode: bool,
    paused: bool,
    drag_hover: bool,
    save_status: Option<String>,
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let (hovered_files, dropped_files) = ui.ctx().input(|input| {
            (
                input.raw.hovered_files.len(),
                input.raw.dropped_files.clone(),
            )
        });
        self.drag_hover = hovered_files > 0;
        if let Some(path) = dropped_files
            .into_iter()
            .map(|file| file.path().to_path_buf())
            .next()
        {
            self.start_sequence(path);
        }
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
        if self.drag_hover {
            egui::Area::new(egui::Id::unique("drop-sequence")).show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.heading("Drop a EuRoC sequence directory");
                    ui.label("Expected: <sequence>/times.txt and <sequence>/mav0/...");
                });
            });
        }
        self.update_texture(ui.ctx());
    }
}

impl ViewerApp {
    fn start_sequence(&mut self, sequence_path: PathBuf) {
        let times = sequence_path.join("times.txt");
        if !times.is_file() {
            self.save_status = Some(format!(
                "Cannot open {}: times.txt not found",
                sequence_path.display()
            ));
            return;
        }
        let sequence = match EurocSequence::load(&sequence_path, &times) {
            Ok(sequence) => sequence,
            Err(error) => {
                self.save_status = Some(format!("Cannot load sequence: {error}"));
                return;
            }
        };
        let system = match System::new(
            &self.vocabulary,
            &self.settings,
            Sensor::IMUStereo,
            true,
            None,
        ) {
            Ok(system) => system,
            Err(error) => {
                self.save_status = Some(format!("Cannot restart SLAM: {error:?}"));
                return;
            }
        };

        self.system.shutdown();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }

        let result = Arc::new(Mutex::new(None));
        let worker_system = system.clone();
        let worker_result = result.clone();
        let worker = std::thread::Builder::new()
            .name("euroc-gui-runner".to_owned())
            .spawn(move || {
                let value = run_sequences(worker_system, &[sequence], false);
                *worker_result.lock().unwrap() = Some(value);
            });
        match worker {
            Ok(worker) => {
                self.system = system;
                self.worker = Some(worker);
                self.result = result;
                self.snapshot = None;
                self.texture = None;
                self.texture_right = None;
                self.paused = false;
                self.localization_mode = false;
                self.save_status = Some(format!("Loaded {}", sequence_path.display()));
            }
            Err(error) => {
                system.shutdown();
                self.save_status = Some(format!("Cannot start sequence worker: {error}"));
            }
        }
    }

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
            ui.label(format!(
                "Keyframes: {} active / {} atlas",
                snapshot
                    .keyframes
                    .iter()
                    .filter(|keyframe| keyframe.active_map)
                    .count(),
                snapshot.keyframes.len()
            ));
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
        ui.checkbox(&mut self.show_inertial_graph, "Show inertial links");
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
        let has_right = self
            .snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.image_right.is_some());
        if let Some(texture) = &self.texture {
            let available = ui.available_size();
            let image_size = if has_right {
                egui::vec2(available.x, (available.y - 8.0) * 0.5)
            } else {
                available
            };
            let aspect = texture.size_vec2().x / texture.size_vec2().y.max(1.0);
            let width = image_size.x.min(image_size.y * aspect);
            let size = egui::vec2(width, width / aspect);
            let response = ui.add(egui::Image::new(egui::load::SizedTexture::new(
                texture.id(),
                size,
            )));
            self.draw_features(ui, &response, false);
            if has_right && let Some(texture_right) = &self.texture_right {
                let aspect = texture_right.size_vec2().x / texture_right.size_vec2().y.max(1.0);
                let width = image_size.x.min(image_size.y * aspect);
                let size = egui::vec2(width, width / aspect);
                let response = ui.add(egui::Image::new(egui::load::SizedTexture::new(
                    texture_right.id(),
                    size,
                )));
                self.draw_features(ui, &response, true);
            }
        } else {
            ui.centered_and_justified(|ui| ui.label("Waiting for image data"));
        }
    }

    fn draw_features(&self, ui: &egui::Ui, response: &egui::Response, right: bool) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let image = if right {
            snapshot.image_right.as_ref()
        } else {
            snapshot.image.as_ref()
        };
        let Some(image) = image else {
            return;
        };
        let scale = egui::vec2(
            response.rect.width() / image.width.max(1) as f32,
            response.rect.height() / image.height.max(1) as f32,
        );
        let painter = ui.painter();
        for feature in snapshot
            .features
            .iter()
            .filter(|feature| feature.right == right)
        {
            let position = response.rect.min + egui::vec2(feature.x * scale.x, feature.y * scale.y);
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
        let draw_frustum = |center: Vector3<f32>,
                            rotation: nalgebra::Matrix3<f32>,
                            size: f32,
                            color: egui::Color32| {
            let half_width = size;
            let half_height = size * 0.75;
            let depth = size * 0.6;
            let vertices = [
                center + rotation * Vector3::new(half_width, half_height, depth),
                center + rotation * Vector3::new(half_width, -half_height, depth),
                center + rotation * Vector3::new(-half_width, -half_height, depth),
                center + rotation * Vector3::new(-half_width, half_height, depth),
            ];
            if let Some(center_2d) = project(center) {
                for vertex in vertices {
                    if let Some(vertex_2d) = project(vertex) {
                        painter.line_segment([center_2d, vertex_2d], egui::Stroke::new(1.0, color));
                    }
                }
            }
            for index in 0..vertices.len() {
                let next = (index + 1) % vertices.len();
                if let (Some(a), Some(b)) = (project(vertices[index]), project(vertices[next])) {
                    painter.line_segment([a, b], egui::Stroke::new(1.0, color));
                }
            }
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
            let inverse = keyframe.pose.inverse();
            let center = inverse.translation.vector;
            keyframe_positions.insert(keyframe.id, center);
            if self.show_keyframes {
                if let Some(position) = project(center) {
                    painter.circle_filled(
                        position,
                        2.5,
                        if keyframe.active_map {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::from_rgb(120, 170, 185)
                        },
                    );
                }
                draw_frustum(
                    center,
                    inverse.rotation.to_rotation_matrix().into_inner(),
                    (self.camera.distance * 0.018).clamp(0.02, 0.8),
                    if keyframe.active_map {
                        egui::Color32::from_rgb(230, 205, 70)
                    } else {
                        egui::Color32::from_rgb(80, 125, 145)
                    },
                );
            }
        }
        if self.show_graph {
            for keyframe in &snapshot.keyframes {
                if !keyframe.active_map {
                    continue;
                }
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
            for keyframe in &snapshot.keyframes {
                if !keyframe.active_map {
                    continue;
                }
                let Some(&source) = keyframe_positions.get(&keyframe.id) else {
                    continue;
                };
                if let Some(parent) = keyframe.parent
                    && let Some(&target) = keyframe_positions.get(&parent)
                    && let (Some(a), Some(b)) = (project(source), project(target))
                {
                    painter.line_segment(
                        [a, b],
                        egui::Stroke::new(1.8, egui::Color32::from_rgb(90, 190, 120)),
                    );
                }
                for connected in &keyframe.loop_edges {
                    if connected <= &keyframe.id {
                        continue;
                    }
                    let Some(&target) = keyframe_positions.get(connected) else {
                        continue;
                    };
                    if let (Some(a), Some(b)) = (project(source), project(target)) {
                        painter.line_segment(
                            [a, b],
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(240, 170, 60)),
                        );
                    }
                }
            }
        }
        if self.show_inertial_graph && snapshot.imu_initialized {
            for keyframe in &snapshot.keyframes {
                if !keyframe.active_map {
                    continue;
                }
                let Some(next) = keyframe.next_keyframe else {
                    continue;
                };
                let Some(&source) = keyframe_positions.get(&keyframe.id) else {
                    continue;
                };
                let Some(&target) = keyframe_positions.get(&next) else {
                    continue;
                };
                if let (Some(a), Some(b)) = (project(source), project(target)) {
                    painter.line_segment(
                        [a, b],
                        egui::Stroke::new(1.2, egui::Color32::from_rgb(225, 85, 85)),
                    );
                }
            }
        }

        if let Some(pose) = snapshot.current_pose {
            let inverse = pose.inverse();
            let center = inverse.translation.vector;
            let rotation = inverse.rotation.to_rotation_matrix().into_inner();
            let axis_length = (self.camera.distance * 0.04).clamp(0.01, 1.5);
            draw_frustum(
                center,
                rotation,
                axis_length,
                egui::Color32::from_rgb(255, 245, 170),
            );
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
                snapshot
                    .keyframes
                    .iter()
                    .filter(|keyframe| keyframe.active_map)
                    .count()
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
        let image_right = self
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.image_right.as_ref());
        if let Some(image) = image_right {
            self.texture_right = Some(context.load_texture(
                "current-frame-right",
                egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.rgba),
                egui::TextureOptions::LINEAR,
            ));
        } else {
            self.texture_right = None;
        }
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
