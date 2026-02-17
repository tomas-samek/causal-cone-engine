// Causal Cone Engine — v0.1
//
// The observer swims through a persistent diff field.
// Entities deposit into the field. The field spreads at c (one cell per tick).
// The observer samples the field to produce each frame.
//
// There are no rays. There are no meshes. There is only the field.

mod field;
mod observer;
mod renderer;

use observer::Observer;
use renderer::Renderer;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

struct App {
    state: Option<AppState>,
    keys_held: HashSet<KeyCode>,
}

struct AppState {
    window: Arc<Window>,
    renderer: Renderer,
    observer: Observer,
    last_frame: Instant,
    tick_accumulator: f64,
    tick_count: u64,
    frame_count: u64,
    fps_timer: Instant,
    current_fps: f64,
    mouse_captured: bool,
}

impl App {
    fn new() -> Self {
        Self {
            state: None,
            keys_held: HashSet::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("Causal Cone Engine v0.1 — diff field renderer")
            .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let renderer = pollster::block_on(Renderer::new(window.clone()));
        let observer = Observer::new();

        self.state = Some(AppState {
            window,
            renderer,
            observer,
            last_frame: Instant::now(),
            tick_accumulator: 0.0,
            tick_count: 0,
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            mouse_captured: false,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                state.renderer.resize(new_size);
                state.window.request_redraw();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                match key_state {
                    ElementState::Pressed => {
                        self.keys_held.insert(key);

                        match key {
                            KeyCode::Escape => {
                                if state.mouse_captured {
                                    state.mouse_captured = false;
                                    state
                                        .window
                                        .set_cursor_grab(winit::window::CursorGrabMode::None)
                                        .ok();
                                    state.window.set_cursor_visible(true);
                                } else {
                                    event_loop.exit();
                                }
                            }
                            _ => {}
                        }
                    }
                    ElementState::Released => {
                        self.keys_held.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if !state.mouse_captured {
                    state.mouse_captured = true;
                    state
                        .window
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                        .or_else(|_| {
                            state
                                .window
                                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        })
                        .ok();
                    state.window.set_cursor_visible(false);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f64();
                state.last_frame = now;

                // FPS counter
                state.frame_count += 1;
                let fps_elapsed = (now - state.fps_timer).as_secs_f64();
                if fps_elapsed >= 1.0 {
                    state.current_fps = state.frame_count as f64 / fps_elapsed;
                    state.frame_count = 0;
                    state.fps_timer = now;

                    state.window.set_title(&format!(
                        "Causal Cone Engine v0.1 — {:.0} FPS — tick {} — observer v={:.3}c",
                        state.current_fps,
                        state.tick_count,
                        state.observer.speed()
                    ));
                }

                // Update observer from input
                state.observer.update(&self.keys_held, dt);

                // Tick simulation (fixed timestep, 30 ticks/sec)
                state.tick_accumulator += dt;
                let tick_interval = 1.0 / 30.0;
                while state.tick_accumulator >= tick_interval {
                    state.renderer.tick(&state.observer);
                    state.tick_count += 1;
                    state.tick_accumulator -= tick_interval;
                }

                // Render
                match state.renderer.render(&state.observer) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        let size = state.window.inner_size();
                        state.renderer.resize(size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        event_loop.exit();
                    }
                    Err(e) => {
                        eprintln!("Render error: {:?}", e);
                    }
                }

                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        if let DeviceEvent::MouseMotion { delta } = event {
            if state.mouse_captured {
                state.observer.mouse_look(delta.0 as f32, delta.1 as f32);
            }
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
