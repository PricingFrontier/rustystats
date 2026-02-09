// =============================================================================
// ONNX Export — From-scratch protobuf serialization
// =============================================================================
//
// Builds valid ONNX model files without any external protobuf dependency.
// Implements the subset of the protobuf wire format needed by ONNX IR v8.
//
// Two entry points exposed to Python:
//   - build_onnx_glm_scoring_py: Level 1 — design-matrix → prediction
//   - serialize_onnx_graph_py:   Generic — Python builds graph, Rust serializes
// =============================================================================

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use numpy::PyReadonlyArray1;

// ── Protobuf wire-format primitives ────────────────────────────────────────

/// Encode an unsigned integer as a protobuf varint.
#[inline]
fn varint(mut v: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        buf.push(if v > 0 { byte | 0x80 } else { byte });
        if v == 0 {
            break;
        }
    }
    buf
}

/// Field key = (field_number << 3) | wire_type.
#[inline]
fn tag(field: u32, wire: u32) -> Vec<u8> {
    varint(((field as u64) << 3) | (wire as u64))
}

/// Varint field (wire type 0).
fn pb_uint(field: u32, val: u64) -> Vec<u8> {
    let mut b = tag(field, 0);
    b.extend(varint(val));
    b
}

/// Length-delimited field (wire type 2).
fn pb_len(field: u32, data: &[u8]) -> Vec<u8> {
    let mut b = tag(field, 2);
    b.extend(varint(data.len() as u64));
    b.extend_from_slice(data);
    b
}

/// String field (length-delimited UTF-8).
#[inline]
fn pb_str(field: u32, s: &str) -> Vec<u8> {
    pb_len(field, s.as_bytes())
}

/// Embedded message field (length-delimited).
#[inline]
fn pb_msg(field: u32, msg: &[u8]) -> Vec<u8> {
    pb_len(field, msg)
}

/// Float32 field (wire type 5 — 32-bit).
fn pb_f32(field: u32, v: f32) -> Vec<u8> {
    let mut b = tag(field, 5);
    b.extend_from_slice(&v.to_le_bytes());
    b
}

// ── ONNX constants ─────────────────────────────────────────────────────────

// TensorProto.DataType
const DT_DOUBLE: u64 = 11;
const DT_INT64: u64 = 7;

// AttributeProto.AttributeType
const AT_FLOAT: u64 = 1;
const AT_INT: u64 = 2;

// ── ONNX message helpers ───────────────────────────────────────────────────
//
// Field numbers come from the official onnx.proto3 spec:
//   https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

/// AttributeProto with a single int value.
fn onnx_attr_int(name: &str, val: i64) -> Vec<u8> {
    let mut a = pb_str(1, name);          // name
    a.extend(pb_uint(3, val as u64));     // i  (field 3)
    a.extend(pb_uint(20, AT_INT));        // type (field 20)
    a
}

/// AttributeProto with a single float value.
fn onnx_attr_float(name: &str, val: f32) -> Vec<u8> {
    let mut a = pb_str(1, name);          // name
    a.extend(pb_f32(2, val));             // f  (field 2)
    a.extend(pb_uint(20, AT_FLOAT));      // type (field 20)
    a
}

/// TensorShapeProto from a slice of dimension values (-1 → symbolic "batch").
fn onnx_shape(dims: &[i64]) -> Vec<u8> {
    let mut buf = Vec::new();
    for &d in dims {
        let dim = if d < 0 {
            pb_str(2, "batch")            // dim_param (field 2)
        } else {
            pb_uint(1, d as u64)          // dim_value (field 1)
        };
        buf.extend(pb_msg(1, &dim));      // Dimension (field 1 of TensorShapeProto)
    }
    buf
}

/// TypeProto for a tensor with the given element type and shape.
fn onnx_type_tensor(elem_type: u64, dims: &[i64]) -> Vec<u8> {
    let mut t = pb_uint(1, elem_type);            // elem_type
    t.extend(pb_msg(2, &onnx_shape(dims)));       // shape
    pb_msg(1, &t)                                 // TypeProto.tensor_type (field 1)
}

/// ValueInfoProto.
fn onnx_value_info(name: &str, elem_type: u64, dims: &[i64]) -> Vec<u8> {
    let mut v = pb_str(1, name);                             // name
    v.extend(pb_msg(2, &onnx_type_tensor(elem_type, dims))); // type
    v
}

/// TensorProto holding float64 raw data.
fn onnx_tensor_f64(name: &str, data: &[f64], dims: &[i64]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        t.extend(pb_uint(1, d as u64));           // dims (field 1, repeated)
    }
    t.extend(pb_uint(2, DT_DOUBLE));              // data_type (field 2)
    t.extend(pb_str(8, name));                    // name (field 8)
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    t.extend(pb_len(9, &raw));                    // raw_data (field 9)
    t
}

/// TensorProto holding int64 raw data.
fn onnx_tensor_i64(name: &str, data: &[i64], dims: &[i64]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        t.extend(pb_uint(1, d as u64));
    }
    t.extend(pb_uint(2, DT_INT64));
    t.extend(pb_str(8, name));
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    t.extend(pb_len(9, &raw));
    t
}

/// NodeProto.
fn onnx_node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        n.extend(pb_str(1, i));                   // input (field 1, repeated)
    }
    for o in outputs {
        n.extend(pb_str(2, o));                   // output (field 2, repeated)
    }
    n.extend(pb_str(4, op));                      // op_type (field 4)
    for a in attrs {
        n.extend(pb_msg(5, a));                   // attribute (field 5, repeated)
    }
    n
}

/// StringStringEntryProto.
fn onnx_kv(key: &str, val: &str) -> Vec<u8> {
    let mut e = pb_str(1, key);
    e.extend(pb_str(2, val));
    e
}

/// GraphProto.
fn onnx_graph(
    name: &str,
    nodes: &[Vec<u8>],
    inits: &[Vec<u8>],
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Vec<u8> {
    let mut g = Vec::new();
    for n in nodes {
        g.extend(pb_msg(1, n));                   // node (field 1)
    }
    g.extend(pb_str(2, name));                    // name (field 2)
    for i in inits {
        g.extend(pb_msg(5, i));                   // initializer (field 5)
    }
    for i in inputs {
        g.extend(pb_msg(11, i));                  // input (field 11)
    }
    for o in outputs {
        g.extend(pb_msg(12, o));                  // output (field 12)
    }
    g
}

/// ModelProto.
fn onnx_model(
    graph: &[u8],
    ir_ver: u64,
    opset_ver: u64,
    producer: &str,
    doc: &str,
    meta: &[(&str, &str)],
) -> Vec<u8> {
    let mut m = pb_uint(1, ir_ver);               // ir_version (field 1)
    m.extend(pb_str(2, producer));                // producer_name (field 2)
    if !doc.is_empty() {
        m.extend(pb_str(6, doc));                 // doc_string (field 6)
    }
    m.extend(pb_msg(7, graph));                   // graph (field 7)
    // opset_import (field 8) — default domain
    let opset = pb_uint(2, opset_ver);            // OperatorSetIdProto.version
    m.extend(pb_msg(8, &opset));
    for &(k, v) in meta {
        m.extend(pb_msg(14, &onnx_kv(k, v)));    // metadata_props (field 14)
    }
    m
}

// ── Level 1: GLM scoring model (design matrix input) ───────────────────────

/// Build a minimal GLM scoring graph:
///   Input  X (batch, n)  →  MatMul(W)  →  Add(b)  →  inverse_link  →  prediction
fn glm_scoring_bytes(
    coefs: &[f64],
    intercept: f64,
    n_features: usize,
    link: &str,
    family: &str,
    meta: &[(&str, &str)],
) -> Vec<u8> {
    let mut nodes = Vec::new();
    let mut inits = Vec::new();

    // Input / output specs
    let inputs = vec![onnx_value_info("X", DT_DOUBLE, &[-1, n_features as i64])];
    let outputs = vec![onnx_value_info("prediction", DT_DOUBLE, &[-1, 1])];

    // Coefficient column-vector  (n, 1)
    inits.push(onnx_tensor_f64("W", coefs, &[n_features as i64, 1]));
    nodes.push(onnx_node("MatMul", &["X", "W"], &["eta_raw"], &[]));

    // Intercept
    inits.push(onnx_tensor_f64("b", &[intercept], &[1]));
    nodes.push(onnx_node("Add", &["eta_raw", "b"], &["eta"], &[]));

    // Inverse link function
    match link {
        "log"      => nodes.push(onnx_node("Exp",        &["eta"], &["prediction"], &[])),
        "logit"    => nodes.push(onnx_node("Sigmoid",    &["eta"], &["prediction"], &[])),
        "identity" => nodes.push(onnx_node("Identity",   &["eta"], &["prediction"], &[])),
        "inverse"  => nodes.push(onnx_node("Reciprocal", &["eta"], &["prediction"], &[])),
        "sqrt"     => nodes.push(onnx_node("Mul",        &["eta", "eta"], &["prediction"], &[])),
        _          => nodes.push(onnx_node("Exp",        &["eta"], &["prediction"], &[])),
    }

    let graph = onnx_graph("rustystats_glm", &nodes, &inits, &inputs, &outputs);
    let doc = format!("RustyStats GLM: {} family, {} link", family, link);
    onnx_model(&graph, 8, 18, "RustyStats", &doc, meta)
}

// ── PyO3 wrappers ──────────────────────────────────────────────────────────

/// Build a Level-1 ONNX model (design-matrix → prediction).
///
/// The resulting model expects a float64 input tensor ``X`` of shape
/// ``(batch, n_features)`` representing the pre-built design matrix
/// (without an intercept column — the intercept is added internally).
///
/// Returns raw ONNX protobuf bytes.
#[pyfunction]
#[pyo3(signature = (coefficients, intercept, n_features, link, family, metadata_keys, metadata_values))]
pub fn build_onnx_glm_scoring_py<'py>(
    py: Python<'py>,
    coefficients: PyReadonlyArray1<f64>,
    intercept: f64,
    n_features: usize,
    link: &str,
    family: &str,
    metadata_keys: Vec<String>,
    metadata_values: Vec<String>,
) -> PyResult<Bound<'py, PyBytes>> {
    let coefs = coefficients.as_slice()?;
    let meta: Vec<(&str, &str)> = metadata_keys
        .iter()
        .zip(metadata_values.iter())
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let bytes = glm_scoring_bytes(coefs, intercept, n_features, link, family, &meta);
    Ok(PyBytes::new_bound(py, &bytes))
}

/// Generic ONNX graph serializer — Python builds the graph description,
/// Rust handles protobuf encoding.
///
/// Accepts parallel vectors describing every node, initializer, input and
/// output of the computation graph.
#[pyfunction]
#[pyo3(signature = (
    node_ops, node_inputs, node_outputs,
    node_attr_names, node_attr_types, node_attr_ints, node_attr_floats,
    init_names_f64, init_data_f64, init_shapes_f64,
    init_names_i64, init_data_i64, init_shapes_i64,
    input_names, input_types, input_shapes,
    output_names, output_types, output_shapes,
    ir_version, opset_version, producer, doc_string,
    meta_keys, meta_values
))]
pub fn serialize_onnx_graph_py<'py>(
    py: Python<'py>,
    // ── Nodes ──
    node_ops: Vec<String>,
    node_inputs: Vec<Vec<String>>,
    node_outputs: Vec<Vec<String>>,
    node_attr_names: Vec<Vec<String>>,
    node_attr_types: Vec<Vec<String>>,   // "int" | "float"
    node_attr_ints: Vec<Vec<i64>>,
    node_attr_floats: Vec<Vec<f32>>,
    // ── Initializers (float64) ──
    init_names_f64: Vec<String>,
    init_data_f64: Vec<Vec<f64>>,
    init_shapes_f64: Vec<Vec<i64>>,
    // ── Initializers (int64) ──
    init_names_i64: Vec<String>,
    init_data_i64: Vec<Vec<i64>>,
    init_shapes_i64: Vec<Vec<i64>>,
    // ── Graph inputs ──
    input_names: Vec<String>,
    input_types: Vec<u64>,
    input_shapes: Vec<Vec<i64>>,
    // ── Graph outputs ──
    output_names: Vec<String>,
    output_types: Vec<u64>,
    output_shapes: Vec<Vec<i64>>,
    // ── Model metadata ──
    ir_version: u64,
    opset_version: u64,
    producer: String,
    doc_string: String,
    meta_keys: Vec<String>,
    meta_values: Vec<String>,
) -> PyResult<Bound<'py, PyBytes>> {
    // ── Nodes ──
    let mut nodes = Vec::with_capacity(node_ops.len());
    for i in 0..node_ops.len() {
        let ins: Vec<&str> = node_inputs[i].iter().map(|s| s.as_str()).collect();
        let outs: Vec<&str> = node_outputs[i].iter().map(|s| s.as_str()).collect();
        let mut attrs = Vec::new();
        for j in 0..node_attr_names[i].len() {
            let a = match node_attr_types[i][j].as_str() {
                "int" => onnx_attr_int(&node_attr_names[i][j], node_attr_ints[i][j]),
                "float" => onnx_attr_float(&node_attr_names[i][j], node_attr_floats[i][j]),
                _ => Vec::new(),
            };
            attrs.push(a);
        }
        nodes.push(onnx_node(&node_ops[i], &ins, &outs, &attrs));
    }

    // ── Initializers ──
    let mut inits = Vec::with_capacity(init_names_f64.len() + init_names_i64.len());
    for i in 0..init_names_f64.len() {
        inits.push(onnx_tensor_f64(
            &init_names_f64[i],
            &init_data_f64[i],
            &init_shapes_f64[i],
        ));
    }
    for i in 0..init_names_i64.len() {
        inits.push(onnx_tensor_i64(
            &init_names_i64[i],
            &init_data_i64[i],
            &init_shapes_i64[i],
        ));
    }

    // ── Graph I/O ──
    let inputs_pb: Vec<Vec<u8>> = input_names
        .iter()
        .enumerate()
        .map(|(i, n)| onnx_value_info(n, input_types[i], &input_shapes[i]))
        .collect();
    let outputs_pb: Vec<Vec<u8>> = output_names
        .iter()
        .enumerate()
        .map(|(i, n)| onnx_value_info(n, output_types[i], &output_shapes[i]))
        .collect();

    // ── Assemble ──
    let graph = onnx_graph("rustystats_glm", &nodes, &inits, &inputs_pb, &outputs_pb);
    let meta: Vec<(&str, &str)> = meta_keys
        .iter()
        .zip(meta_values.iter())
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let model = onnx_model(&graph, ir_version, opset_version, &producer, &doc_string, &meta);

    Ok(PyBytes::new_bound(py, &model))
}
