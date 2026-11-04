//! Renders the donut with a single yellow point light placed at the camera,
//! so the light always shines from the direction the camera faces.

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_position: vec4<f32>,
    // xyz: world-space position, w: intensity
    light_position: vec4<f32>,
    // rgb: color
    light_color: vec4<f32>,
    // rgb: base color, w: shininess
    material: vec4<f32>,
    // rgb: ambient color, w: specular strength
    ambient: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);

    var output: VertexOutput;
    output.clip_position = uniforms.view_proj * world_position;
    output.world_position = world_position.xyz;
    output.world_normal = (uniforms.model * vec4<f32>(input.normal, 0.0)).xyz;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);

    let to_light = uniforms.light_position.xyz - input.world_position;
    let light_distance = max(length(to_light), 1e-4);
    let light_direction = to_light / light_distance;
    let view_direction = normalize(uniforms.camera_position.xyz - input.world_position);
    let half_direction = normalize(light_direction + view_direction);

    // Inverse-square falloff softened by constant and linear terms, so the
    // donut stays evenly lit at a comfortable camera distance.
    let attenuation = 1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance);
    let radiance = uniforms.light_color.rgb * uniforms.light_position.w * attenuation;

    let n_dot_l = max(dot(normal, light_direction), 0.0);
    let diffuse = uniforms.material.rgb * radiance * n_dot_l;

    let specular = radiance
        * pow(max(dot(normal, half_direction), 0.0), uniforms.material.w)
        * uniforms.ambient.w
        * step(1e-4, n_dot_l);

    let ambient = uniforms.material.rgb * uniforms.ambient.rgb;

    return vec4<f32>(diffuse + specular + ambient, 1.0);
}
