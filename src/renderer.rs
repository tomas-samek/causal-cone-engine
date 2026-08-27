// Renderer — the observer's retina, uploaded.
//
// A fullscreen quad. A fragment shader. The receptor array already holds the
// image: the CPU retina sums what arrived along entity pipes. The renderer
// copies it into two W×H textures and the shader thresholds, shades, composites.
//
// No meshes. No draw calls. No lights. No shadows. No marching.

use crate::field::{DiffField, FIELD_SIZE};
use crate::observer::Observer;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};
use std::sync::Arc;

/// Uniform data sent to the GPU each frame
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    /// Inverse view-projection matrix — transforms screen coords to world rays
    inv_view_proj: [f32; 16],
    /// Observer position in field space
    observer_pos: [f32; 3],
    /// Observer speed as fraction of c (affects FOV, aberration)
    observer_speed: f32,
    /// Field dimensions
    field_size: [f32; 3],
    /// Current tick
    tick: f32,
    /// AABB of active solid geometry (for ray march culling)
    aabb_min: [f32; 3],
    _pad1: f32,
    aabb_max: [f32; 3],
    _pad2: f32,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Pipeline
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // The retina — two W×H textures on GPU
    retina_dc: wgpu::Texture,
    retina_nd: wgpu::Texture,
    retina_bind_group: wgpu::BindGroup,
    retina_layout: wgpu::BindGroupLayout,
    retina_sampler: wgpu::Sampler,
    retina_size: (u32, u32),

    // The field data on CPU
    diff_field: DiffField,
    upload_buf: Vec<u16>, // f16 staging buffer for both textures (padded rows)
}

/// Allocate the two receptor textures at `w`×`h` and bind them with `sampler`.
fn create_retina_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::Texture, wgpu::BindGroup) {
    let make = |label: &str| device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let dc = make("RetinaDC");
    let nd = make("RetinaND");
    let dc_view = dc.create_view(&Default::default());
    let nd_view = nd.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("RetinaBindGroup"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dc_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&nd_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });
    (dc, nd, bind_group)
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // Create wgpu instance and surface
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        log::info!("GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("CausalConeDevice"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let retina_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("RetinaSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Uniform buffer ---
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::cast_slice(&[Uniforms {
                inv_view_proj: glam::Mat4::IDENTITY.to_cols_array(),
                observer_pos: [128.0, 128.0, 190.0],
                observer_speed: 0.0,
                field_size: [FIELD_SIZE as f32; 3],
                tick: 0.0,
                aabb_min: [0.0; 3],
                _pad1: 0.0,
                aabb_max: [FIELD_SIZE as f32; 3],
                _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group layouts ---
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("UniformLayout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let retina_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RetinaLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UniformBindGroup"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // --- The retina textures — sized by the CPU receptor array ---
        let diff_field = DiffField::new();
        let retina_size = (diff_field.retina.width, diff_field.retina.height);
        let (retina_dc, retina_nd, retina_bind_group) = create_retina_textures(
            &device, &retina_layout, &retina_sampler, retina_size.0, retina_size.1,
        );

        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RetinaDisplay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/retina.wgsl").into()),
        });

        // --- Pipeline ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RenderPipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &retina_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RetinaDisplayPipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // fullscreen quad — no vertex buffer needed
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // fullscreen quad, no culling
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
            retina_dc,
            retina_nd,
            retina_bind_group,
            retina_layout,
            retina_sampler,
            retina_size,
            diff_field,
            upload_buf: Vec::new(),
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Run one simulation tick with reactive pipeline
    pub fn tick(&mut self, observer: &Observer) {
        let aspect = self.size.width as f32 / self.size.height.max(1) as f32;
        let view = observer.view_matrix();
        let proj = observer.projection_matrix(aspect);
        let view_proj = proj * view;
        self.diff_field.tick(view_proj);
    }

    /// Render one frame — sample the field from the observer's perspective
    pub fn render(&mut self, observer: &Observer) -> Result<(), wgpu::SurfaceError> {
        // Resolution changed (keys 7/8)? Recreate the textures.
        let (rw, rh) = (self.diff_field.retina.width, self.diff_field.retina.height);
        if (rw, rh) != self.retina_size {
            let (dc, nd, bg) = create_retina_textures(&self.device, &self.retina_layout, &self.retina_sampler, rw, rh);
            self.retina_dc = dc;
            self.retina_nd = nd;
            self.retina_bind_group = bg;
            self.retina_size = (rw, rh);
            self.diff_field.retina.dirty = true;
        }
        if self.diff_field.retina.dirty {
            // Rows are padded to a 256-byte multiple. `write_texture` does not
            // require it (only `copy_buffer_to_texture` does), but keeping the
            // staging rows 256-aligned costs a few bytes at odd resolutions
            // (keys 7/8) and matches what the driver wants anyway.
            let row_bytes = ((rw * 8 + 255) / 256) * 256;
            let stride = (row_bytes / 2) as usize; // u16 per padded row
            let total = stride * rh as usize;
            self.upload_buf.resize(total * 2, 0);
            let (dc_buf, nd_buf) = self.upload_buf.split_at_mut(total);
            let f = |x: f32| half::f16::from_f32(x).to_bits();
            for (i, r) in self.diff_field.retina.receptors.iter().enumerate() {
                let (x, y) = ((i % rw as usize), (i / rw as usize));
                let o = y * stride + x * 4;
                let d = r.density.clamp(0.0, 60000.0);
                let inv = if r.density > 1e-6 { 1.0 / r.density } else { 0.0 };
                let nl = r.normal.length();
                let nrm = if nl > 1e-6 { r.normal / nl } else { glam::Vec3::Y };
                dc_buf[o] = f(d);
                dc_buf[o + 1] = f((r.color[0] * inv).min(60000.0));
                dc_buf[o + 2] = f((r.color[1] * inv).min(60000.0));
                dc_buf[o + 3] = f((r.color[2] * inv).min(60000.0));
                nd_buf[o] = f(nrm.x);
                nd_buf[o + 1] = f(nrm.y);
                nd_buf[o + 2] = f(nrm.z);
                nd_buf[o + 3] = f((r.depth * inv).min(60000.0));
            }
            for (tex, buf) in [(&self.retina_dc, &*dc_buf), (&self.retina_nd, &*nd_buf)] {
                self.queue.write_texture(
                    wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    bytemuck::cast_slice(buf),
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(rh) },
                    wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
                );
            }
            self.diff_field.retina.dirty = false;
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update uniforms
        let aspect = self.size.width as f32 / self.size.height as f32;
        let view_matrix = observer.view_matrix();
        let proj_matrix = observer.projection_matrix(aspect);
        let view_proj = proj_matrix * view_matrix;
        let inv_view_proj = view_proj.inverse();

        let uniforms = Uniforms {
            inv_view_proj: inv_view_proj.to_cols_array(),
            observer_pos: observer.position.to_array(),
            observer_speed: observer.speed(),
            field_size: [FIELD_SIZE as f32; 3],
            tick: self.diff_field.tick as f32,
            aabb_min: self.diff_field.aabb_min.to_array(),
            _pad1: 0.0,
            aabb_max: self.diff_field.aabb_max.to_array(),
            _pad2: 0.0,
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Encode render commands
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderEncoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FieldSamplePass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &self.retina_bind_group, &[]);

            // Draw fullscreen triangle (3 vertices, no buffer)
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn toggle_trie_depth_viz(&mut self) {
        self.diff_field.show_trie_depth = !self.diff_field.show_trie_depth;
        log::info!("Trie depth visualization: {}", self.diff_field.show_trie_depth);
    }

    pub fn dump_trie_info(&self) {
        let entity_count = self.diff_field.entities.len();
        let cs = &self.diff_field.consumption_states;
        for i in 0..entity_count.min(cs.len()) {
            if let Some(ref s) = cs[i] {
                if s.consumed > 0 || !s.learning {
                    log::info!(
                        "Entity {} (group {}): depth={}, spectrum={}, consumed={}, rejected={}",
                        i, self.diff_field.entities[i].group, s.depth, s.spectrum.len(),
                        s.consumed, s.rejected
                    );
                }
            }
        }
        let extra = cs.len().saturating_sub(entity_count);
        if extra > 0 {
            log::info!("+ {} trie-only states (no spatial entity)", extra);
        }
    }

    pub fn decrease_render_depth(&mut self) {
        self.diff_field.render_depth_cutoff = self.diff_field.render_depth_cutoff.saturating_sub(1);
        log::info!("Render depth cutoff: {}", self.diff_field.render_depth_cutoff);
    }

    pub fn increase_render_depth(&mut self) {
        self.diff_field.render_depth_cutoff = self.diff_field.render_depth_cutoff.saturating_add(1).min(u16::MAX);
        log::info!("Render depth cutoff: {}", self.diff_field.render_depth_cutoff);
    }

    pub fn time_lapse(&self) -> u64 {
        self.diff_field.walker.time_lapse
    }

    pub fn halve_time_lapse(&mut self) {
        self.diff_field.walker.halve_time_lapse();
        log::info!("Time lapse: ×{}", self.diff_field.walker.time_lapse);
    }

    pub fn double_time_lapse(&mut self) {
        self.diff_field.walker.double_time_lapse();
        log::info!("Time lapse: ×{}", self.diff_field.walker.time_lapse);
    }

    pub fn dump_field_stats(&self) {
        self.diff_field.dump_field_stats();
    }

    pub fn tune_density_scale(&mut self, factor: f32) {
        self.diff_field.tune_density = crate::field::scale_tune(self.diff_field.tune_density, factor);
        self.log_tuning();
    }

    pub fn tune_color_scale(&mut self, factor: f32) {
        self.diff_field.tune_color = crate::field::scale_tune(self.diff_field.tune_color, factor);
        self.log_tuning();
    }

    pub fn tune_atten_scale(&mut self, factor: f32) {
        self.diff_field.atten_k = crate::field::scale_tune(self.diff_field.atten_k, factor);
        self.diff_field.retina_force_relink = true;
        self.diff_field.compute_edge_atten_public();
        self.log_tuning();
    }

    pub fn scale_retina(&mut self, factor: f32) {
        let r = &mut self.diff_field.retina;
        let w = (r.width as f32 * factor).round() as u32;
        let h = (r.height as f32 * factor).round() as u32;
        r.resize(w, h);
        log::info!("Retina resolution: {}x{}", r.width, r.height);
    }

    fn log_tuning(&self) {
        log::info!(
            "Tuning: density ×{:.4}, color ×{:.4}, atten_k ×{:.4}",
            self.diff_field.tune_density, self.diff_field.tune_color, self.diff_field.atten_k
        );
    }
}
