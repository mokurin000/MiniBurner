//! Donut shading with layered fur shells, lit by three point lights: a yellow
//! key light at the camera, an orange light in the center of the world and a
//! red light behind the donut, opposite the camera.
//!
//! The fur is drawn as many copies of the mesh, each one pushed a little
//! further along the surface normal. Every shell keeps only the strands that
//! are long enough to reach it, so the stack forms fur of varying length. The
//! strand coverage and the fur length both fall off towards grazing angles,
//! which makes the fur sparser around the silhouette.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_position: vec4<f32>,
    // xyz: world-space position, w: intensity
    light_positions: array<vec4<f32>, 3>,
    // rgb: light color
    light_colors: array<vec4<f32>, 3>,
    // rgb: base color, w: shininess
    material: vec4<f32>,
    // rgb: ambient color, w: specular strength
    ambient: vec4<f32>,
    // x: fur length, y: strand density, z: fur length scale at the silhouette
    fur: vec4<f32>,
    // rgb: fur tip color, w: alpha of one shell
    fur_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct FurVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Normalized shell height in `(0, 1]`, one value per instance.
    @location(3) layer: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

struct FurOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) layer: f32,
};

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn shade(base_color: vec3<f32>, world_position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let view_direction = normalize(uniforms.camera_position.xyz - world_position);
    var color = base_color * uniforms.ambient.rgb;

    for (var index = 0u; index < 3u; index += 1u) {
        let light_position = uniforms.light_positions[index].xyz;
        let intensity = uniforms.light_positions[index].w;
        let to_light = light_position - world_position;
        let light_distance = max(length(to_light), 1e-4);
        let light_direction = to_light / light_distance;
        let half_direction = normalize(light_direction + view_direction);

        // Inverse-square falloff softened by constant and linear terms, so the
        // donut stays evenly lit at a comfortable camera distance.
        let attenuation =
            1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance);
        let radiance = uniforms.light_colors[index].rgb * intensity * attenuation;

        let n_dot_l = max(dot(normal, light_direction), 0.0);
        color += base_color * radiance * n_dot_l;
        color += radiance
            * pow(max(dot(normal, half_direction), 0.0), uniforms.material.w)
            * uniforms.ambient.w
            * step(1e-4, n_dot_l);
    }

    return color;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);

    var output: VertexOutput;
    output.clip_position = uniforms.view_proj * world_position;
    output.world_position = world_position.xyz;
    output.world_normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
    return output;
}

@vertex
fn vs_fur(input: FurVertexInput) -> FurOutput {
    // Give the strands some length variation so the fur does not look like a
    // smooth inflated shell.
    let clump = 0.6 + 0.4 * hash21(floor(input.uv * 12.0));
    let displaced = input.position + input.normal * (uniforms.fur.x * input.layer * clump);
    let world_position = uniforms.model * vec4<f32>(displaced, 1.0);

    var output: FurOutput;
    output.clip_position = uniforms.view_proj * world_position;
    output.world_position = world_position.xyz;
    output.world_normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
    output.uv = input.uv;
    output.layer = input.layer;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);
    return vec4<f32>(shade(uniforms.material.rgb, input.world_position, normal), 1.0);
}

@fragment
fn fs_fur(input: FurOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);
    let view_direction = normalize(uniforms.camera_position.xyz - input.world_position);
    let facing = abs(dot(normal, view_direction));
    // The fur thins out where the surface turns away from the camera.
    let edge = smoothstep(0.0, 0.85, facing);

    // Each strand column has its own length; a shell only keeps the strands
    // that are long enough to reach it.
    let cell = floor(input.uv * uniforms.fur.y);
    let strand = hash21(cell);
    let height = input.layer;
    let hair_length = mix(0.4, 1.0, strand) * mix(uniforms.fur.z, 1.0, edge);
    if height > hair_length {
        discard;
    }

    // Around the silhouette fewer strands survive, so the fur visibly thins
    // out into separate tufts instead of a smooth halo.
    let coverage = mix(0.35, 1.0, edge);
    if hash21(cell + vec2(37.0, 11.0)) > coverage {
        discard;
    }

    // Fade the last shells of a strand out instead of cutting them off.
    let tip_fade = 1.0 - smoothstep(hair_length - 0.25, hair_length, height);
    let root = uniforms.material.rgb * 0.45;
    let hair_color = mix(root, uniforms.fur_color.rgb, height);
    let alpha = uniforms.fur_color.w * tip_fade * mix(0.12, 1.0, edge);

    return vec4<f32>(shade(hair_color, input.world_position, normal), alpha);
}
