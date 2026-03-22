// Renderer — the observer's retina.
//
// A fullscreen quad. A fragment shader. For each pixel, cast a direction
// from the observer into the field. Sample. First hit above threshold = color.
//
// No meshes. No draw calls. No lights. No shadows.
// Just a 3D texture and a camera swimming through it.

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

    // The field — 3D texture on GPU
    field_texture: wgpu::Texture,
    field_bind_group: wgpu::BindGroup,

    // The field data on CPU
    diff_field: DiffField,
    last_uploaded_tick: u64,
    upload_buf: Vec<u16>, // f16 staging buffer for slab-by-slab upload
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

        // --- Create 3D field texture ---
        let field_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DiffField3D"),
            size: wgpu::Extent3d {
                width: FIELD_SIZE,
                height: FIELD_SIZE,
                depth_or_array_layers: FIELD_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let field_texture_view = field_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let field_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FieldSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
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

        let field_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FieldLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let field_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FieldBindGroup"),
            layout: &field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&field_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&field_sampler),
                },
            ],
        });

        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FieldSampler"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/field_sample.wgsl").into()),
        });

        // --- Pipeline ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RenderPipelineLayout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &field_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FieldSamplePipeline"),
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

        let diff_field = DiffField::new();

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
            field_texture,
            field_bind_group,
            diff_field,
            last_uploaded_tick: 0,
            upload_buf: vec![0u16; (FIELD_SIZE * FIELD_SIZE * 4) as usize],
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
        // Upload dirty slabs when simulation has advanced — AABB-restricted f32→f16 per slab
        if self.diff_field.tick != self.last_uploaded_tick {
            let margin = 40.0f32;
            let fs = FIELD_SIZE as usize;
            let dx_min = (self.diff_field.aabb_min.x - margin).max(0.0) as usize;
            let dx_max = ((self.diff_field.aabb_max.x + margin) as usize + 1).min(fs);
            let dy_min = (self.diff_field.aabb_min.y - margin).max(0.0) as usize;
            let dy_max = ((self.diff_field.aabb_max.y + margin) as usize + 1).min(fs);
            let sub_w = dx_max - dx_min;
            let sub_h = dy_max - dy_min;
            let sub_bytes_per_row = sub_w as u32 * 4 * 2; // 4 channels × 2 bytes (f16)

            for z in 0..FIELD_SIZE as usize {
                if !self.diff_field.dirty_slabs[z] { continue; }

                // Convert f32 cells to f16 — only the AABB sub-rectangle
                let slab_base = z * fs * fs;
                let mut buf_idx = 0;
                for y in dy_min..dy_max {
                    let row_base = slab_base + y * fs;
                    for x in dx_min..dx_max {
                        let cell = &self.diff_field.cells[row_base + x];
                        self.upload_buf[buf_idx]     = half::f16::from_f32(cell.density).to_bits();
                        self.upload_buf[buf_idx + 1] = half::f16::from_f32(cell.color_r).to_bits();
                        self.upload_buf[buf_idx + 2] = half::f16::from_f32(cell.color_g).to_bits();
                        self.upload_buf[buf_idx + 3] = half::f16::from_f32(cell.color_b).to_bits();
                        buf_idx += 4;
                    }
                }

                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.field_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: dx_min as u32, y: dy_min as u32, z: z as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&self.upload_buf[..buf_idx]),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(sub_bytes_per_row),
                        rows_per_image: Some(sub_h as u32),
                    },
                    wgpu::Extent3d {
                        width: sub_w as u32,
                        height: sub_h as u32,
                        depth_or_array_layers: 1,
                    },
                );
            }
            self.last_uploaded_tick = self.diff_field.tick;
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
            render_pass.set_bind_group(1, &self.field_bind_group, &[]);

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
}
