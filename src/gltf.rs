//! A small glTF 2.0 loader for the model embedded in this app.
//!
//! The donut ships as a binary glTF (`.glb`) with a single triangle mesh in
//! the embedded buffer and no materials or animations. This module reads
//! exactly that subset: one mesh, without node transforms, made of
//! `POSITION` and `NORMAL` (and optionally `TEXCOORD_0`) attributes plus an
//! index buffer. The JSON document is handled by the minimal parser further
//! down, so the app does not need a serialization dependency.

use bytemuck::{Pod, Zeroable};
use color_eyre::eyre::{Result, ensure, eyre};

/// A single mesh vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Position in model space.
    pub position: [f32; 3],
    /// Normal in model space.
    pub normal: [f32; 3],
    /// First texture coordinate set, if the model provides one.
    pub uv: [f32; 2],
}

/// Geometry of the loaded glTF mesh.
#[derive(Debug)]
pub struct Mesh {
    /// Interleaved vertices of all primitives.
    pub vertices: Vec<Vertex>,
    /// Triangle indices into [`Mesh::vertices`].
    pub indices: Vec<u32>,
    /// Center of the axis-aligned bounding box of the model.
    pub center: [f32; 3],
    /// Radius of the bounding sphere around [`Mesh::center`].
    pub radius: f32,
}

impl Mesh {
    /// Number of triangles in the mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

// glTF component type constants used by the accessors.
const COMPONENT_UNSIGNED_BYTE: usize = 5121;
const COMPONENT_UNSIGNED_SHORT: usize = 5123;
const COMPONENT_UNSIGNED_INT: usize = 5125;
const COMPONENT_FLOAT: usize = 5126;

/// Load a mesh from the bytes of a binary glTF file.
pub fn load_glb(bytes: &[u8]) -> Result<Mesh> {
    ensure!(bytes.len() >= 12, "the GLB file is shorter than its header");
    ensure!(&bytes[0..4] == b"glTF", "the file is not a binary glTF");

    let version = u32::from_le_bytes(bytes[4..8].try_into()?);
    ensure!(version == 2, "unsupported glTF version {version}");

    let length = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    ensure!(
        length <= bytes.len(),
        "the GLB file is truncated: header announces {length} bytes, got {}",
        bytes.len()
    );
    let bytes = &bytes[..length];

    let mut json = None;
    let mut bin = None;
    let mut chunks = &bytes[12..];
    while chunks.len() >= 8 {
        let chunk_length = u32::from_le_bytes(chunks[0..4].try_into()?) as usize;
        let chunk_type = &chunks[4..8];
        ensure!(
            chunks.len() >= 8 + chunk_length,
            "the GLB file has a truncated chunk"
        );
        let data = &chunks[8..8 + chunk_length];
        match chunk_type {
            b"JSON" => json = Some(data),
            b"BIN\0" => bin = Some(data),
            _ => {}
        }
        chunks = &chunks[8 + chunk_length..];
    }

    let json = parse_json(json.ok_or_else(|| eyre!("the GLB file has no JSON chunk"))?)?;
    let bin = bin.unwrap_or_default();
    Document { root: &json, bin }.mesh()
}

/// A glTF document together with its embedded binary buffer.
struct Document<'a> {
    root: &'a Json,
    bin: &'a [u8],
}

impl<'a> Document<'a> {
    /// Extract the first mesh referenced by a node, flattened into one mesh.
    fn mesh(&self) -> Result<Mesh> {
        let asset_version = self
            .root
            .get("asset")
            .and_then(|asset| asset.get("version"))
            .and_then(Json::as_str)
            .ok_or_else(|| eyre!("the glTF document has no asset version"))?;
        ensure!(
            asset_version.starts_with("2."),
            "unsupported glTF asset version {asset_version}"
        );

        let mesh_index = self.mesh_index()?;
        let mesh = self
            .root
            .get("meshes")
            .and_then(Json::as_array)
            .and_then(|meshes| meshes.get(mesh_index))
            .ok_or_else(|| eyre!("the glTF document has no mesh {mesh_index}"))?;
        let primitives = mesh
            .get("primitives")
            .and_then(Json::as_array)
            .ok_or_else(|| eyre!("mesh {mesh_index} has no primitives"))?;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (primitive_index, primitive) in primitives.iter().enumerate() {
            let mode = primitive.get("mode").and_then(Json::as_usize).unwrap_or(4);
            ensure!(
                mode == 4,
                "primitive {primitive_index} is not a triangle list (mode {mode})"
            );

            let attributes = primitive
                .get("attributes")
                .ok_or_else(|| eyre!("primitive {primitive_index} has no attributes"))?;
            let positions = self.attribute(attributes, "POSITION", "VEC3", 3, 4)?;
            let normals = self.attribute(attributes, "NORMAL", "VEC3", 3, 4)?;
            let uvs = self.optional_attribute(attributes, "TEXCOORD_0", "VEC2", 2, 4)?;

            let vertex_count = positions.len() / 3;
            ensure!(
                positions.len() == vertex_count * 3,
                "primitive {primitive_index} has a malformed position buffer"
            );
            ensure!(
                normals.len() == positions.len(),
                "primitive {primitive_index} has {} normals for {vertex_count} vertices",
                normals.len() / 3
            );
            if let Some(uvs) = &uvs {
                ensure!(
                    uvs.len() == vertex_count * 2,
                    "primitive {primitive_index} has a malformed UV buffer"
                );
            }

            let first_vertex = vertices.len() as u32;
            for vertex in 0..vertex_count {
                let position = &positions[vertex * 3..vertex * 3 + 3];
                let normal = &normals[vertex * 3..vertex * 3 + 3];
                let uv = uvs
                    .as_ref()
                    .map(|uvs| [uvs[vertex * 2], uvs[vertex * 2 + 1]])
                    .unwrap_or([0.0, 0.0]);
                vertices.push(Vertex {
                    position: [position[0], position[1], position[2]],
                    normal: [normal[0], normal[1], normal[2]],
                    uv,
                });
            }

            match primitive.get("indices").and_then(Json::as_usize) {
                Some(accessor) => {
                    let mut primitive_indices = self.indices(accessor)?;
                    ensure!(
                        primitive_indices.len() % 3 == 0,
                        "primitive {primitive_index} is not a triangle list"
                    );
                    for index in &mut primitive_indices {
                        ensure!(
                            (*index as usize) < vertex_count,
                            "primitive {primitive_index} has an index out of bounds"
                        );
                        *index += first_vertex;
                    }
                    indices.append(&mut primitive_indices);
                }
                None => {
                    ensure!(
                        vertex_count % 3 == 0,
                        "primitive {primitive_index} without indices is not a triangle list"
                    );
                    indices.extend((0..vertex_count as u32).map(|index| first_vertex + index));
                }
            }
        }

        ensure!(!vertices.is_empty(), "the glTF model contains no geometry");
        ensure!(!indices.is_empty(), "the glTF model contains no triangles");

        let (center, radius) = bounds(&vertices);
        Ok(Mesh {
            vertices,
            indices,
            center,
            radius,
        })
    }

    /// Find the mesh referenced by the first node that has one.
    fn mesh_index(&self) -> Result<usize> {
        let Some(nodes) = self.root.get("nodes").and_then(Json::as_array) else {
            return Ok(0);
        };
        let Some(node) = nodes.iter().find(|node| node.get("mesh").is_some()) else {
            return Ok(0);
        };
        let transformed = ["matrix", "translation", "rotation", "scale"]
            .iter()
            .any(|key| node.get(*key).is_some());
        ensure!(
            !transformed,
            "the built-in glTF loader does not support node transforms"
        );
        node.get("mesh")
            .and_then(Json::as_usize)
            .ok_or_else(|| eyre!("the glTF node has an invalid mesh index"))
    }

    /// Read a required float attribute as a flat list of components.
    fn attribute(
        &self,
        attributes: &Json,
        name: &str,
        kind: &str,
        components: usize,
        component_size: usize,
    ) -> Result<Vec<f32>> {
        let accessor = attributes
            .get(name)
            .and_then(Json::as_usize)
            .ok_or_else(|| eyre!("the glTF primitive has no {name} attribute"))?;
        self.floats(accessor, kind, components, component_size)
    }

    /// Read an optional float attribute as a flat list of components.
    fn optional_attribute(
        &self,
        attributes: &Json,
        name: &str,
        kind: &str,
        components: usize,
        component_size: usize,
    ) -> Result<Option<Vec<f32>>> {
        match attributes.get(name).and_then(Json::as_usize) {
            Some(accessor) => self
                .floats(accessor, kind, components, component_size)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Read a `FLOAT` accessor.
    fn floats(
        &self,
        accessor: usize,
        kind: &str,
        components: usize,
        component_size: usize,
    ) -> Result<Vec<f32>> {
        let view = self.accessor(accessor, kind, components, COMPONENT_FLOAT, component_size)?;
        let mut values = Vec::with_capacity(view.count * components);
        for element in 0..view.count {
            let start = view.base + element * view.stride;
            for component in 0..components {
                let offset = start + component * component_size;
                let bytes: [u8; 4] = self.bin[offset..offset + 4]
                    .try_into()
                    .expect("the accessor bounds were checked");
                values.push(f32::from_le_bytes(bytes));
            }
        }
        Ok(values)
    }

    /// Read an unsigned integer index accessor.
    fn indices(&self, accessor: usize) -> Result<Vec<u32>> {
        let component_type = self
            .accessor_json(accessor)?
            .get("componentType")
            .and_then(Json::as_usize)
            .unwrap_or_default();
        let component_size = match component_type {
            COMPONENT_UNSIGNED_BYTE => 1,
            COMPONENT_UNSIGNED_SHORT => 2,
            COMPONENT_UNSIGNED_INT => 4,
            other => {
                return Err(eyre!(
                    "accessor {accessor} has an unsupported index type {other}"
                ));
            }
        };
        let view = self.accessor(accessor, "SCALAR", 1, component_type, component_size)?;
        let mut values = Vec::with_capacity(view.count);
        for element in 0..view.count {
            let offset = view.base + element * view.stride;
            let index = match component_size {
                1 => self.bin[offset] as u32,
                2 => u16::from_le_bytes(
                    self.bin[offset..offset + 2]
                        .try_into()
                        .expect("the accessor bounds were checked"),
                ) as u32,
                4 => u32::from_le_bytes(
                    self.bin[offset..offset + 4]
                        .try_into()
                        .expect("the accessor bounds were checked"),
                ),
                _ => unreachable!("component sizes are validated above"),
            };
            values.push(index);
        }
        Ok(values)
    }

    /// Validate and describe an accessor.
    fn accessor(
        &self,
        index: usize,
        kind: &str,
        components: usize,
        component_type: usize,
        component_size: usize,
    ) -> Result<Accessor> {
        let accessor = self.accessor_json(index)?;
        let actual_kind = accessor
            .get("type")
            .and_then(Json::as_str)
            .ok_or_else(|| eyre!("accessor {index} has no type"))?;
        ensure!(
            actual_kind == kind,
            "accessor {index} is {actual_kind}, expected {kind}"
        );
        let actual_component_type = accessor
            .get("componentType")
            .and_then(Json::as_usize)
            .ok_or_else(|| eyre!("accessor {index} has no component type"))?;
        ensure!(
            actual_component_type == component_type,
            "accessor {index} has component type {actual_component_type}, expected {component_type}"
        );
        ensure!(
            accessor.get("sparse").is_none(),
            "accessor {index} is sparse, which is not supported"
        );

        let count = accessor
            .get("count")
            .and_then(Json::as_usize)
            .ok_or_else(|| eyre!("accessor {index} has no count"))?;
        let view_index = accessor
            .get("bufferView")
            .and_then(Json::as_usize)
            .ok_or_else(|| eyre!("accessor {index} has no buffer view"))?;
        let view = self
            .root
            .get("bufferViews")
            .and_then(|views| views.index(view_index))
            .ok_or_else(|| eyre!("the glTF document has no buffer view {view_index}"))?;
        ensure!(
            view.get("buffer").and_then(Json::as_usize) == Some(0),
            "only the embedded glTF buffer is supported"
        );

        let view_offset = view.get("byteOffset").and_then(Json::as_usize).unwrap_or(0);
        let view_length = view
            .get("byteLength")
            .and_then(Json::as_usize)
            .ok_or_else(|| eyre!("buffer view {view_index} has no length"))?;
        ensure!(
            view_offset + view_length <= self.bin.len(),
            "buffer view {view_index} is outside of the binary chunk"
        );

        let accessor_offset = accessor
            .get("byteOffset")
            .and_then(Json::as_usize)
            .unwrap_or(0);
        ensure!(
            accessor_offset <= view_length,
            "accessor {index} starts outside of its buffer view"
        );
        let element_size = components * component_size;
        let stride = view
            .get("byteStride")
            .and_then(Json::as_usize)
            .unwrap_or(element_size);
        ensure!(
            stride >= element_size,
            "buffer view {view_index} has a too small stride"
        );

        let needed = accessor_offset + count.saturating_sub(1) * stride + element_size;
        ensure!(
            count == 0 || needed <= view_length,
            "accessor {index} is out of bounds of its buffer view"
        );

        Ok(Accessor {
            base: view_offset + accessor_offset,
            stride,
            count,
        })
    }

    /// Fetch the raw JSON of an accessor.
    fn accessor_json(&self, index: usize) -> Result<&'a Json> {
        self.root
            .get("accessors")
            .and_then(|accessors| accessors.index(index))
            .ok_or_else(|| eyre!("the glTF document has no accessor {index}"))
    }
}

/// A validated view into a buffer view.
struct Accessor {
    /// Offset of the first element in the binary chunk.
    base: usize,
    /// Distance between consecutive elements in bytes.
    stride: usize,
    /// Number of elements.
    count: usize,
}

/// Compute the bounding sphere of the vertices.
fn bounds(vertices: &[Vertex]) -> ([f32; 3], f32) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex.position[axis]);
            max[axis] = max[axis].max(vertex.position[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let radius = vertices
        .iter()
        .map(|vertex| {
            let offset = [
                vertex.position[0] - center[0],
                vertex.position[1] - center[1],
                vertex.position[2] - center[2],
            ];
            (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt()
        })
        .fold(0.0f32, f32::max);
    (center, radius)
}

// ---------------------------------------------------------------------------
// Minimal JSON parser
// ---------------------------------------------------------------------------

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Look up a member of an object.
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// Look up an element of an array.
    fn index(&self, index: usize) -> Option<&Json> {
        match self {
            Self::Array(items) => items.get(index),
            _ => None,
        }
    }

    /// Borrow the array, if this is an array.
    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Borrow the string, if this is a string.
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// The value as a non-negative integer, if it is one.
    fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Number(value) if *value >= 0.0 && value.fract() == 0.0 => Some(*value as usize),
            _ => None,
        }
    }
}

/// Parse a JSON document.
fn parse_json(bytes: &[u8]) -> Result<Json> {
    Parser::new(bytes).document()
}

/// A recursive-descent JSON parser.
struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, pos: 0 }
    }

    fn document(mut self) -> Result<Json> {
        let value = self.value()?;
        self.skip_whitespace();
        ensure!(
            self.pos == self.input.len(),
            "unexpected data after the JSON document"
        );
        Ok(value)
    }

    fn value(&mut self) -> Result<Json> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(byte) => {
                return Err(eyre!(
                    "unexpected character {:?} in JSON at byte {}",
                    byte as char,
                    self.pos
                ));
            }
            None => return Err(eyre!("unexpected end of the JSON document")),
        }
    }

    fn object(&mut self) -> Result<Json> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            entries.push((key, self.value()?));
            self.skip_whitespace();
            match self.bump()? {
                b',' => {}
                b'}' => break,
                byte => {
                    return Err(eyre!(
                        "unexpected character {:?} in a JSON object at byte {}",
                        byte as char,
                        self.pos
                    ));
                }
            }
        }
        Ok(Json::Object(entries))
    }

    fn array(&mut self) -> Result<Json> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_whitespace();
            match self.bump()? {
                b',' => {}
                b']' => break,
                byte => {
                    return Err(eyre!(
                        "unexpected character {:?} in a JSON array at byte {}",
                        byte as char,
                        self.pos
                    ));
                }
            }
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut output = Vec::new();
        loop {
            match self.bump()? {
                b'"' => break,
                b'\\' => match self.bump()? {
                    b'"' => output.push(b'"'),
                    b'\\' => output.push(b'\\'),
                    b'/' => output.push(b'/'),
                    b'b' => output.push(0x08),
                    b'f' => output.push(0x0c),
                    b'n' => output.push(b'\n'),
                    b'r' => output.push(b'\r'),
                    b't' => output.push(b'\t'),
                    b'u' => {
                        let high = self.hex4()?;
                        let codepoint = if (0xd800..=0xdbff).contains(&high) {
                            ensure!(
                                self.bump()? == b'\\' && self.bump()? == b'u',
                                "unpaired UTF-16 surrogate in a JSON string"
                            );
                            let low = self.hex4()?;
                            ensure!(
                                (0xdc00..=0xdfff).contains(&low),
                                "invalid UTF-16 surrogate pair in a JSON string"
                            );
                            0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
                        } else {
                            high
                        };
                        let character = char::from_u32(codepoint)
                            .ok_or_else(|| eyre!("invalid unicode escape in a JSON string"))?;
                        let mut buffer = [0u8; 4];
                        output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                    }
                    byte => {
                        return Err(eyre!(
                            "invalid escape {:?} in a JSON string at byte {}",
                            byte as char,
                            self.pos
                        ));
                    }
                },
                byte if byte < 0x20 => {
                    return Err(eyre!(
                        "control character in a JSON string at byte {}",
                        self.pos
                    ));
                }
                byte => output.push(byte),
            }
        }
        String::from_utf8(output).map_err(|error| eyre!("invalid UTF-8 in a JSON string: {error}"))
    }

    fn hex4(&mut self) -> Result<u32> {
        let mut value = 0;
        for _ in 0..4 {
            let byte = self.bump()?;
            let digit = (byte as char)
                .to_digit(16)
                .ok_or_else(|| eyre!("invalid hex digit in a JSON escape"))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.input[start..self.pos])?;
        let value = text
            .parse::<f64>()
            .map_err(|error| eyre!("invalid JSON number {text:?}: {error}"))?;
        Ok(Json::Number(value))
    }

    fn literal(&mut self, literal: &str) -> Result<()> {
        let end = self.pos + literal.len();
        ensure!(
            self.input.get(self.pos..end) == Some(literal.as_bytes()),
            "invalid JSON literal at byte {}",
            self.pos
        );
        self.pos = end;
        Ok(())
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn bump(&mut self) -> Result<u8> {
        let byte = self
            .peek()
            .ok_or_else(|| eyre!("unexpected end of the JSON document"))?;
        self.pos += 1;
        Ok(byte)
    }

    fn expect(&mut self, expected: u8) -> Result<()> {
        let actual = self.bump()?;
        ensure!(
            actual == expected,
            "expected {:?} at byte {}, found {:?}",
            expected as char,
            self.pos - 1,
            actual as char
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model that ships with the app.
    const DONUT: &[u8] = include_bytes!("../donut.glb");

    #[test]
    fn loads_the_embedded_donut() {
        let mesh = load_glb(DONUT).expect("donut.glb should load");
        assert_eq!(mesh.vertices.len(), 3185);
        assert_eq!(mesh.indices.len(), 18432);
        assert_eq!(mesh.triangle_count(), 6144);
        assert!(mesh.radius > 0.0 && mesh.radius.is_finite());
        for axis in 0..3 {
            assert!(mesh.center[axis].abs() < 1e-6, "center: {:?}", mesh.center);
        }
        for index in &mesh.indices {
            assert!((*index as usize) < mesh.vertices.len());
        }
        for vertex in &mesh.vertices {
            assert!(vertex.position.iter().all(|value| value.is_finite()));
            assert!(vertex.normal.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn parses_json_escapes_and_nesting() {
        let json = parse_json(
            br#"{"name":"\u73af\u4f53","values":[1,-2.5,1e3,true,false,null],"nested":{"key":"a\\b\"c"}}"#,
        )
        .expect("valid JSON");
        assert_eq!(json.get("name").and_then(Json::as_str), Some("环体"));
        assert_eq!(
            json.get("values")
                .and_then(Json::as_array)
                .map(|values| values.len()),
            Some(6)
        );
        assert_eq!(
            json.get("nested")
                .and_then(|nested| nested.get("key"))
                .and_then(Json::as_str),
            Some("a\\b\"c")
        );
        assert!(parse_json(br#"{"missing": }"#).is_err());
    }

    #[test]
    fn rejects_non_glb_input() {
        assert!(load_glb(b"not a glb").is_err());
    }
}
