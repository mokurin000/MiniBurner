//! Rendering of the donut model.
//!
//! The donut is the `donut.glb` model shipped with the app: a torus whose
//! symmetry axis is the glTF Y axis. It is tilted 7 degrees away from the
//! screen plane and then spun around the screen's Z axis, so the camera always
//! sees the whole ring while it turns.
//!
//! On top of the base surface a stack of fur shells is drawn, each one pushed
//! further along the surface normals. The shells only keep the strands that are
//! long enough to reach them, and both the fur length and its coverage fall off
//! towards grazing angles, so the fur is dense in the middle of the donut and
//! thins out around its silhouette. Drawing every shell is the main GPU load of
//! the app.
//!
//! The scene is lit by three point lights: a yellow key light at the camera, an
//! orange light in the center of the world and a red light behind the donut,
//! opposite the camera. Each fur fragment evaluates all three.
//!
//! Rendering happens into a 4x multisampled target that is resolved into the
//! swapchain texture, which smooths the silhouette and adds more GPU work.

use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::Result;
use crate::gltf::{self, Vertex};
use crate::math::Mat4;

/// The model that ships next to the sources.
const MODEL_GLB: &[u8] = include_bytes!("../donut.glb");

/// Depth format used by the render pass.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Number of samples per pixel of the multisampled render target.
const MSAA_SAMPLES: u32 = 4;

/// Vertical field of view of the camera in degrees.
const FIELD_OF_VIEW_Y: f32 = 45.0;
/// Angle between the torus plane and the screen plane, in degrees.
const TILT_DEGREES: f32 = 7.0;
/// Distance of the near clipping plane.
const NEAR_PLANE: f32 = 0.1;
/// Distance of the far clipping plane, comfortably beyond any camera distance.
const FAR_PLANE: f32 = 100.0;
/// Extra room around the donut, as a multiple of its bounding radius.
const FRAME_MARGIN: f32 = 1.12;

/// Base (albedo) color of the donut's dough and glaze.
const DONUT_COLOR: [f32; 3] = [0.94, 0.76, 0.69];
/// Color of the yellow point light at the camera. This is what makes the donut
/// shine yellow.
const KEY_LIGHT_COLOR: [f32; 3] = [1.0, 0.86, 0.2];
/// Intensity of the camera light before distance attenuation.
const KEY_LIGHT_INTENSITY: f32 = 2.5;
/// Color of the orange point light at the center of the world.
const CENTER_LIGHT_COLOR: [f32; 3] = [1.0, 0.42, 0.04];
/// Intensity of the center light before distance attenuation.
const CENTER_LIGHT_INTENSITY: f32 = 1.8;
/// Color of the red point light behind the donut, opposite the camera.
const BACK_LIGHT_COLOR: [f32; 3] = [1.0, 0.05, 0.03];
/// Intensity of the back light before distance attenuation.
const BACK_LIGHT_INTENSITY: f32 = 2.5;
/// Color of the faint ambient fill, so the unlit side is not pitch black.
const AMBIENT_COLOR: [f32; 3] = [0.10, 0.11, 0.13];
/// Specular exponent (shininess) of the dough.
const SHININESS: f32 = 48.0;
/// How strong the light's specular highlight is.
const SPECULAR_STRENGTH: f32 = 0.6;
/// Length of the fur along the surface normal, in model units.
const FUR_LENGTH: f32 = 0.24;
/// Number of fur strands per UV unit.
const FUR_DENSITY: f32 = 350.0;
/// Fur length scale at grazing angles, where the fur thins out.
const FUR_EDGE_LENGTH: f32 = 0.18;
/// Color of the fur tips.
const FUR_COLOR: [f32; 3] = [1.0, 0.72, 0.28];
/// Alpha of a single fur shell.
const FUR_LAYER_ALPHA: f32 = 0.18;
/// Background color of the canvas.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.043,
    g: 0.047,
    b: 0.059,
    a: 1.0,
};

/// Seconds between two frame-rate measurements.
const FPS_WINDOW: Duration = Duration::from_secs(1);
/// Number of vertices the frame-rate overlay buffer can hold.
const HUD_CAPACITY: usize = 4096;
/// Size of one font pixel of the frame-rate overlay, in pixels.
const HUD_SCALE: f32 = 4.0;
/// Color of the frame-rate text.
const HUD_TEXT_COLOR: [f32; 4] = [1.0, 0.86, 0.3, 0.95];
/// Color of the panel behind the frame-rate text.
const HUD_PANEL_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.45];
/// 5x7 bitmap font: digits `0`-`9` followed by `F`, `P` and `S`.
const HUD_FONT: [[u8; 7]; 13] = [
    [
        0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
    ],
    [
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ],
    [
        0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
    ],
    [
        0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
    ],
    [
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
    ],
    [
        0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
    ],
    [
        0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
    ],
    [
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
    ],
    [
        0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
    ],
    [
        0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
    ],
    [
        0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
    ],
    [
        0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
    ],
];

/// A vertex of the frame-rate overlay, already in clip space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct HudVertex {
    position: [f32; 2],
    color: [f32; 4],
}

/// Uniform block shared with `donut.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_position: [f32; 4],
    /// xyz: world-space position, w: intensity.
    light_positions: [[f32; 4]; 3],
    /// rgb: light color.
    light_colors: [[f32; 4]; 3],
    material: [f32; 4],
    ambient: [f32; 4],
    fur: [f32; 4],
    fur_color: [f32; 4],
}

impl Uniforms {
    /// Build the uniforms for one frame viewed with the given aspect ratio and
    /// the donut spun by `spin` radians around the Z axis.
    fn new(mesh: &gltf::Mesh, aspect_ratio: f32, spin: f32) -> Self {
        let half_fov_y = FIELD_OF_VIEW_Y.to_radians() * 0.5;
        // Fit the bounding sphere on both axes, with a little margin.
        let half_fov_x = (half_fov_y.tan() * aspect_ratio).atan();
        let distance =
            (mesh.radius / half_fov_y.tan()).max(mesh.radius / half_fov_x.tan()) * FRAME_MARGIN;

        // The camera sits on the +Z axis and looks at the origin. The yellow
        // key light shares its position, so it shines along the view
        // direction. An orange light sits at the center of the world (inside
        // the donut's hole) and a red light sits behind the donut, opposite the
        // camera.
        let eye = [0.0, 0.0, distance];
        let view = Mat4::look_at_rh(eye, [0.0; 3], [0.0, 1.0, 0.0]);
        let projection = Mat4::perspective_rh(
            FIELD_OF_VIEW_Y.to_radians(),
            aspect_ratio,
            NEAR_PLANE,
            FAR_PLANE,
        );
        // Center the mesh, tilt it 7 degrees away from the screen plane, then
        // spin it around the screen's Z axis.
        let model = Mat4::from_rotation_z(spin)
            * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2 + TILT_DEGREES.to_radians())
            * Mat4::from_translation([-mesh.center[0], -mesh.center[1], -mesh.center[2]]);
        let view_proj = projection * view;

        Self {
            view_proj: view_proj.cols,
            model: model.cols,
            camera_position: [eye[0], eye[1], eye[2], 1.0],
            light_positions: [
                [eye[0], eye[1], eye[2], KEY_LIGHT_INTENSITY],
                [0.0, 0.0, 0.0, CENTER_LIGHT_INTENSITY],
                [0.0, 0.0, -distance, BACK_LIGHT_INTENSITY],
            ],
            light_colors: [
                [
                    KEY_LIGHT_COLOR[0],
                    KEY_LIGHT_COLOR[1],
                    KEY_LIGHT_COLOR[2],
                    0.0,
                ],
                [
                    CENTER_LIGHT_COLOR[0],
                    CENTER_LIGHT_COLOR[1],
                    CENTER_LIGHT_COLOR[2],
                    0.0,
                ],
                [
                    BACK_LIGHT_COLOR[0],
                    BACK_LIGHT_COLOR[1],
                    BACK_LIGHT_COLOR[2],
                    0.0,
                ],
            ],
            material: [DONUT_COLOR[0], DONUT_COLOR[1], DONUT_COLOR[2], SHININESS],
            ambient: [
                AMBIENT_COLOR[0],
                AMBIENT_COLOR[1],
                AMBIENT_COLOR[2],
                SPECULAR_STRENGTH,
            ],
            fur: [FUR_LENGTH, FUR_DENSITY, FUR_EDGE_LENGTH, 0.0],
            fur_color: [FUR_COLOR[0], FUR_COLOR[1], FUR_COLOR[2], FUR_LAYER_ALPHA],
        }
    }
}

/// The render pipelines paired with the target format they were built for.
struct Pipelines {
    format: wgpu::TextureFormat,
    /// The opaque donut surface.
    base: wgpu::RenderPipeline,
    /// The blended fur shells, one instance per layer.
    fur: wgpu::RenderPipeline,
    /// The frame-rate overlay.
    hud: wgpu::RenderPipeline,
}

/// The depth buffer, recreated whenever the target is resized.
struct DepthTarget {
    width: u32,
    height: u32,
    view: wgpu::TextureView,
}

/// The multisampled color buffer, resolved into the swapchain every frame.
struct MultisampleTarget {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

/// Draws the donut model into a texture provided by a [`wgpu::Device`].
pub struct DonutRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// One normalized shell height per fur layer.
    fur_layer_buffer: wgpu::Buffer,
    /// Number of fur shells drawn over the body; this is the main GPU load.
    fur_layers: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipelines: Option<Pipelines>,
    depth: Option<DepthTarget>,
    msaa: Option<MultisampleTarget>,
    mesh: gltf::Mesh,
    /// Vertices of the frame-rate overlay, rebuilt when the rate changes.
    hud_buffer: wgpu::Buffer,
    hud_vertices: Vec<HudVertex>,
    hud_vertex_count: u32,
    /// The target size the HUD vertices were built for.
    hud_size: (u32, u32),
    hud_dirty: bool,
    /// Frames submitted since the last measurement of the frame rate.
    frames: u32,
    /// The last measured, actually submitted frames per second.
    fps: u32,
    fps_start: Instant,
}

impl DonutRenderer {
    /// Upload the embedded donut model and build its resources.
    ///
    /// `fur_layers` is the number of fur shells drawn over the body; it is the
    /// main GPU load of the app.
    pub fn new(device: &wgpu::Device, fur_layers: u32) -> Result<Self> {
        if fur_layers == 0 {
            return Err(color_eyre::eyre::eyre!(
                "the donut needs at least one fur layer"
            ));
        }

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

        // One instance per fur shell, carrying the shell's normalized height.
        let shells = (1..=fur_layers)
            .map(|layer| layer as f32 / fur_layers as f32)
            .collect::<Vec<_>>();
        let fur_layer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Donut Fur Layer Buffer"),
            contents: bytemuck::cast_slice(&shells),
            usage: wgpu::BufferUsages::VERTEX,
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

        let hud_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Donut HUD Buffer"),
            size: (HUD_CAPACITY * size_of::<HudVertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count,
            fur_layer_buffer,
            fur_layers,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            pipelines: None,
            depth: None,
            msaa: None,
            mesh,
            hud_buffer,
            hud_vertices: Vec::with_capacity(HUD_CAPACITY),
            hud_vertex_count: 0,
            hud_size: (0, 0),
            hud_dirty: true,
            frames: 0,
            fps: 0,
            fps_start: Instant::now(),
        })
    }

    /// Number of triangles in the model.
    pub fn triangle_count(&self) -> usize {
        self.mesh.triangle_count()
    }

    /// Draw the donut into `view`, which must be `size` pixels large.
    ///
    /// `spin` is the rotation around the screen's Z axis, in radians.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        spin: f32,
    ) {
        let (width, height) = size;
        if width == 0 || height == 0 {
            return;
        }

        self.tick_fps_counter();

        if self.pipelines.as_ref().map(|pipeline| pipeline.format) != Some(format) {
            self.pipelines = Some(create_pipelines(device, &self.bind_group_layout, format));
        }
        self.ensure_depth_target(device, width, height);
        self.ensure_multisample_target(device, width, height, format);

        let uniforms = Uniforms::new(&self.mesh, width as f32 / height as f32, spin);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        if self.hud_dirty || self.hud_size != (width, height) {
            self.rebuild_hud(width, height);
            queue.write_buffer(
                &self.hud_buffer,
                0,
                bytemuck::cast_slice(&self.hud_vertices),
            );
            self.hud_size = (width, height);
            self.hud_dirty = false;
        }

        let pipelines = self.pipelines.as_ref().expect("created above");
        let depth_view = &self.depth.as_ref().expect("created above").view;
        let multisample_view = &self.msaa.as_ref().expect("created above").view;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Donut Command Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Donut Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Render into the multisampled buffer and resolve it into
                    // the swapchain texture.
                    view: multisample_view,
                    depth_slice: None,
                    resolve_target: Some(view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Discard,
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

            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            // The opaque body first, so it writes the depth buffer.
            render_pass.set_pipeline(&pipelines.base);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);

            // Then every fur shell, back to front, blended over the body.
            render_pass.set_pipeline(&pipelines.fur);
            render_pass.set_vertex_buffer(1, self.fur_layer_buffer.slice(..));
            render_pass.draw_indexed(0..self.index_count, 0, 0..self.fur_layers);

            // Finally the frame-rate overlay, ignoring the depth buffer.
            render_pass.set_pipeline(&pipelines.hud);
            render_pass.set_vertex_buffer(0, self.hud_buffer.slice(..));
            render_pass.draw(0..self.hud_vertex_count, 0..1);
        }
        queue.submit([encoder.finish()]);
        self.frames += 1;
    }

    /// Recompute the displayed frame rate once per [`FPS_WINDOW`].
    fn tick_fps_counter(&mut self) {
        let elapsed = self.fps_start.elapsed();
        if elapsed >= FPS_WINDOW {
            self.fps = (self.frames as f32 / elapsed.as_secs_f32()).round() as u32;
            self.frames = 0;
            self.fps_start = Instant::now();
            self.hud_dirty = true;
        }
    }

    /// Rebuild the frame-rate overlay for a target of `width` x `height` pixels.
    fn rebuild_hud(&mut self, width: u32, height: u32) {
        self.hud_vertices.clear();

        let text = format!("FPS {}", self.fps);
        let scale = HUD_SCALE;
        let glyph_width = 5.0 * scale;
        let glyph_height = 7.0 * scale;
        let spacing = scale;
        let margin = 16.0;
        let padding = 10.0;
        let text_width = text.chars().count() as f32 * (glyph_width + spacing) - spacing;

        push_hud_quad(
            &mut self.hud_vertices,
            width,
            height,
            margin,
            margin,
            margin + text_width + padding * 2.0,
            margin + glyph_height + padding * 2.0,
            HUD_PANEL_COLOR,
        );

        for (index, character) in text.chars().enumerate() {
            let glyph = hud_glyph(character);
            let origin_x = margin + padding + index as f32 * (glyph_width + spacing);
            let origin_y = margin + padding;
            for (row, bits) in glyph.iter().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) == 0 {
                        continue;
                    }
                    let x = origin_x + column as f32 * scale;
                    let y = origin_y + row as f32 * scale;
                    push_hud_quad(
                        &mut self.hud_vertices,
                        width,
                        height,
                        x,
                        y,
                        x + scale,
                        y + scale,
                        HUD_TEXT_COLOR,
                    );
                }
            }
        }

        self.hud_vertex_count = self.hud_vertices.len() as u32;
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
            sample_count: MSAA_SAMPLES,
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

    /// Make sure the multisampled color buffer matches the target size and format.
    fn ensure_multisample_target(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) {
        if self.msaa.as_ref().is_some_and(|target| {
            target.width == width && target.height == height && target.format == format
        }) {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Donut MSAA Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.msaa = Some(MultisampleTarget {
            width,
            height,
            format,
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        });
    }
}

/// The 5x7 bitmap of a character of the HUD font.
fn hud_glyph(character: char) -> [u8; 7] {
    match character {
        '0'..='9' => HUD_FONT[(character as u8 - b'0') as usize],
        'F' => HUD_FONT[10],
        'P' => HUD_FONT[11],
        'S' => HUD_FONT[12],
        _ => [0; 7],
    }
}

/// Append a rectangle given in pixels to the HUD vertices, in clip space.
fn push_hud_quad(
    vertices: &mut Vec<HudVertex>,
    width: u32,
    height: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    let width = width as f32;
    let height = height as f32;
    let to_clip = |x: f32, y: f32| [x / width * 2.0 - 1.0, 1.0 - y / height * 2.0];

    let top_left = to_clip(x0, y0);
    let top_right = to_clip(x1, y0);
    let bottom_right = to_clip(x1, y1);
    let bottom_left = to_clip(x0, y1);

    for position in [
        top_left,
        top_right,
        bottom_right,
        top_left,
        bottom_right,
        bottom_left,
    ] {
        vertices.push(HudVertex { position, color });
    }
}

/// Build the render pipelines for a color target in `format`.
fn create_pipelines(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> Pipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Donut Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("donut.wgsl").into()),
    });

    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
    const LAYER_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![3 => Float32];

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Donut Pipeline Layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    };
    let layer_layout = wgpu::VertexBufferLayout {
        array_stride: size_of::<f32>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &LAYER_ATTRIBUTES,
    };

    let base = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Donut Base Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(vertex_layout.clone())],
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
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    });

    let fur = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Donut Fur Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_fur"),
            buffers: &[Some(vertex_layout), Some(layer_layout)],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_fur"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    });

    let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Donut HUD Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("hud.wgsl").into()),
    });
    const HUD_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

    let hud = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Donut HUD Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &hud_shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<HudVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &HUD_ATTRIBUTES,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &hud_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    });

    Pipelines {
        format,
        base,
        fur,
        hud,
    }
}
