//! Rendering of the donut model.
//!
//! The donut is the `donut.glb` model shipped with the app: a torus whose
//! symmetry axis is the glTF Y axis. It is rotated a quarter turn around X so
//! the torus ring lies flat in the view plane: the camera sees the whole ring
//! instead of only the tube around its rim.
//!
//! It is lit by a single yellow point light placed at the camera, so the light
//! always shines from the direction the camera is facing.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::Result;
use crate::gltf::{self, Vertex};
use crate::math::Mat4;

/// The model that ships next to the sources.
const MODEL_GLB: &[u8] = include_bytes!("../donut.glb");

/// Depth format used by the render pass.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Vertical field of view of the camera in degrees.
const FIELD_OF_VIEW_Y: f32 = 45.0;
/// Distance of the near clipping plane.
const NEAR_PLANE: f32 = 0.1;
/// Distance of the far clipping plane, comfortably beyond any camera distance.
const FAR_PLANE: f32 = 100.0;
/// Extra room around the donut, as a multiple of its bounding radius.
const FRAME_MARGIN: f32 = 1.12;

/// Base (albedo) color of the donut's dough and glaze.
const DONUT_COLOR: [f32; 3] = [0.94, 0.76, 0.69];
/// Color of the point light. This is what makes the donut shine yellow.
const LIGHT_COLOR: [f32; 3] = [1.0, 0.86, 0.2];
/// Intensity of the point light before distance attenuation.
const LIGHT_INTENSITY: f32 = 2.5;
/// Color of the faint ambient fill, so the unlit side is not pitch black.
const AMBIENT_COLOR: [f32; 3] = [0.10, 0.11, 0.13];
/// Specular exponent (shininess) of the dough.
const SHININESS: f32 = 48.0;
/// How strong the light's specular highlight is.
const SPECULAR_STRENGTH: f32 = 0.6;
/// Background color of the canvas.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.043,
    g: 0.047,
    b: 0.059,
    a: 1.0,
};

/// Uniform block shared with `donut.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_position: [f32; 4],
    light_position: [f32; 4],
    light_color: [f32; 4],
    material: [f32; 4],
    ambient: [f32; 4],
}

impl Uniforms {
    /// Build the uniforms for one frame viewed with the given aspect ratio.
    fn new(mesh: &gltf::Mesh, aspect_ratio: f32) -> Self {
        let half_fov_y = FIELD_OF_VIEW_Y.to_radians() * 0.5;
        // Fit the bounding sphere on both axes, with a little margin.
        let half_fov_x = (half_fov_y.tan() * aspect_ratio).atan();
        let distance =
            (mesh.radius / half_fov_y.tan()).max(mesh.radius / half_fov_x.tan()) * FRAME_MARGIN;

        // The camera sits on the +Z axis and looks at the origin. The point
        // light shares its position, so it shines along the view direction.
        let eye = [0.0, 0.0, distance];
        let view = Mat4::look_at_rh(eye, [0.0; 3], [0.0, 1.0, 0.0]);
        let projection = Mat4::perspective_rh(
            FIELD_OF_VIEW_Y.to_radians(),
            aspect_ratio,
            NEAR_PLANE,
            FAR_PLANE,
        );
        // Center the mesh, then turn the torus' Y axis towards the camera so
        // the whole ring is parallel to the screen.
        let model = Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2)
            * Mat4::from_translation([-mesh.center[0], -mesh.center[1], -mesh.center[2]]);
        let view_proj = projection * view;

        Self {
            view_proj: view_proj.cols,
            model: model.cols,
            camera_position: [eye[0], eye[1], eye[2], 1.0],
            light_position: [eye[0], eye[1], eye[2], LIGHT_INTENSITY],
            light_color: [LIGHT_COLOR[0], LIGHT_COLOR[1], LIGHT_COLOR[2], 0.0],
            material: [DONUT_COLOR[0], DONUT_COLOR[1], DONUT_COLOR[2], SHININESS],
            ambient: [
                AMBIENT_COLOR[0],
                AMBIENT_COLOR[1],
                AMBIENT_COLOR[2],
                SPECULAR_STRENGTH,
            ],
        }
    }
}

/// A render pipeline paired with the target format it was built for.
struct Pipeline {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
}

/// The depth buffer, recreated whenever the target is resized.
struct DepthTarget {
    width: u32,
    height: u32,
    view: wgpu::TextureView,
}

/// Draws the donut model into a texture provided by a [`wgpu::Device`].
pub struct DonutRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: Option<Pipeline>,
    depth: Option<DepthTarget>,
    mesh: gltf::Mesh,
}

impl DonutRenderer {
    /// Upload the embedded donut model and build its resources.
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let mesh = gltf::load_glb(MODEL_GLB)?;
        let index_count = u32::try_from(mesh.indices.len())
            .map_err(|_| color_eyre::eyre::eyre!("the donut model has too many indices"))?;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Donut Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Donut Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Donut Uniform Buffer"),
            size: size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Donut Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Donut Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            pipeline: None,
            depth: None,
            mesh,
        })
    }

    /// Number of triangles in the model.
    pub fn triangle_count(&self) -> usize {
        self.mesh.triangle_count()
    }

    /// Draw the donut into `view`, which must be `size` pixels large.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let (width, height) = size;
        if width == 0 || height == 0 {
            return;
        }

        if self.pipeline.as_ref().map(|pipeline| pipeline.format) != Some(format) {
            self.pipeline = Some(Pipeline {
                format,
                pipeline: create_pipeline(device, &self.bind_group_layout, format),
            });
        }
        self.ensure_depth_target(device, width, height);

        let uniforms = Uniforms::new(&self.mesh, width as f32 / height as f32);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let pipeline = &self.pipeline.as_ref().expect("created above").pipeline;
        let depth_view = &self.depth.as_ref().expect("created above").view;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Donut Command Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Donut Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        queue.submit([encoder.finish()]);
    }

    /// Make sure the depth buffer matches the size of the render target.
    fn ensure_depth_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self
            .depth
            .as_ref()
            .is_some_and(|depth| depth.width == width && depth.height == height)
        {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Donut Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth = Some(DepthTarget {
            width,
            height,
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        });
    }
}

/// Build the render pipeline for a color target in `format`.
fn create_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Donut Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("donut.wgsl").into()),
    });

    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Donut Pipeline Layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Donut Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &ATTRIBUTES,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
