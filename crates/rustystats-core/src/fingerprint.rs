//! Order-sensitive column fingerprints (`specs/out-of-core-fit.md` §9).
//!
//! An out-of-core fit reads its source once per pass and must see the same rows in
//! the same order every time. The freeze pass records a fingerprint of every model
//! input column; each later pass recomputes the fingerprints of the columns it
//! reads and compares them. A fingerprint is a streaming 128-bit XXH3 hash of a
//! tagged byte encoding of the values in row order, so it does not depend on how
//! the rows were batched.

use xxhash_rust::xxh3::Xxh3;

/// What a value is, the first byte of its encoding: values of different kinds never
/// share an encoding.
pub mod kind {
    /// A float64, by its exact bit pattern.
    pub const FLOAT: u8 = 1;
    /// An integer, by its 64-bit pattern.
    pub const INTEGER: u8 = 2;
    pub const BOOLEAN: u8 = 3;
    /// A date, datetime, duration or time, by its physical integer.
    pub const TEMPORAL: u8 = 4;
    /// A string, categorical or enum, by its UTF-8 label.
    pub const STRING: u8 = 5;
    /// A binary value, by its bytes.
    pub const BINARY: u8 = 6;
    /// Any other scalar (decimal, 128-bit integer), by its text.
    pub const TEXT: u8 = 7;
}

/// Bytes buffered before they go to the hash: the encoder's only allocation, fixed
/// whatever the input's size.
const CHUNK_BYTES: usize = 8192;

/// A streaming fingerprint of one column.
///
/// Each value is encoded as its kind byte, a validity byte, then (for a valid value)
/// its 8-byte pattern or its 8-byte length and bytes; the encodings of the rows are
/// hashed in row order through a fixed buffer. XXH3 is a streaming hash, so the
/// digest does not depend on how the rows are split into calls.
#[derive(Clone)]
pub struct ColumnFingerprint {
    hasher: Xxh3,
    rows: u64,
    buffer: Vec<u8>,
}

impl Default for ColumnFingerprint {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnFingerprint {
    pub fn new() -> Self {
        Self {
            hasher: Xxh3::new(),
            rows: 0,
            buffer: Vec::with_capacity(CHUNK_BYTES),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        if self.buffer.len() + bytes.len() > CHUNK_BYTES {
            self.flush();
            if bytes.len() >= CHUNK_BYTES {
                self.hasher.update(bytes);
                return;
            }
        }
        self.buffer.extend_from_slice(bytes);
    }

    fn flush(&mut self) {
        if !self.buffer.is_empty() {
            self.hasher.update(&self.buffer);
            self.buffer.clear();
        }
    }

    /// Add values of `kind` given as 64-bit patterns, in row order. `valid[i] ==
    /// false` marks a null, encoded as its kind and a zero validity byte.
    pub fn update_words(&mut self, kind: u8, values: &[u64], valid: Option<&[bool]>) {
        for (i, &value) in values.iter().enumerate() {
            if valid.is_some_and(|valid| !valid[i]) {
                self.push(&[kind, 0]);
            } else {
                self.push(&[kind, 1]);
                self.push(&value.to_le_bytes());
            }
        }
        self.rows += values.len() as u64;
    }

    /// Add integers by their 64-bit patterns.
    pub fn update_bits(&mut self, values: &[u64], valid: Option<&[bool]>) {
        self.update_words(kind::INTEGER, values, valid);
    }

    /// Add float64 values, by their exact bit patterns.
    pub fn update_f64(&mut self, values: &[f64], valid: Option<&[bool]>) {
        for (i, value) in values.iter().enumerate() {
            if valid.is_some_and(|valid| !valid[i]) {
                self.push(&[kind::FLOAT, 0]);
            } else {
                self.push(&[kind::FLOAT, 1]);
                self.push(&value.to_bits().to_le_bytes());
            }
        }
        self.rows += values.len() as u64;
    }

    /// Add one byte-string value of `kind` (None: a null): its kind, validity byte,
    /// 8-byte length and bytes, streamed through the fixed buffer.
    pub fn update_bytes(&mut self, kind: u8, value: Option<&[u8]>) {
        match value {
            None => self.push(&[kind, 0]),
            Some(bytes) => {
                self.push(&[kind, 1]);
                self.push(&(bytes.len() as u64).to_le_bytes());
                self.push(bytes);
            }
        }
        self.rows += 1;
    }

    /// Rows seen so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The 128-bit digest of everything seen so far.
    pub fn digest(&self) -> u128 {
        let mut hasher = self.hasher.clone();
        hasher.update(&self.buffer);
        hasher.digest128()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_does_not_depend_on_batching() {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let valid: Vec<bool> = (0..1000).map(|i| i % 7 != 0).collect();
        let mut whole = ColumnFingerprint::new();
        whole.update_f64(&values, Some(&valid));
        let mut parts = ColumnFingerprint::new();
        for (chunk, mask) in values.chunks(33).zip(valid.chunks(33)) {
            parts.update_f64(chunk, Some(mask));
        }
        assert_eq!(whole.digest(), parts.digest());
        assert_eq!(whole.rows(), 1000);
    }

    #[test]
    fn byte_values_are_length_delimited_and_tagged() {
        let digest = |values: &[Option<&[u8]>], kind: u8| {
            let mut fp = ColumnFingerprint::new();
            for value in values {
                fp.update_bytes(kind, *value);
            }
            (fp.digest(), fp.rows())
        };
        let ab_c = digest(&[Some(b"ab"), Some(b"c")], kind::STRING);
        assert_eq!(ab_c.1, 2);
        // Lengths delimit the values; a null differs from an empty value; kinds differ.
        assert_ne!(ab_c, digest(&[Some(b"a"), Some(b"bc")], kind::STRING));
        assert_ne!(
            digest(&[Some(b"")], kind::STRING),
            digest(&[None], kind::STRING)
        );
        assert_ne!(ab_c, digest(&[Some(b"ab"), Some(b"c")], kind::BINARY));
    }

    #[test]
    fn kinds_and_long_values_are_distinguished() {
        let mut integer = ColumnFingerprint::new();
        integer.update_bits(&[1.5f64.to_bits()], None);
        let mut float = ColumnFingerprint::new();
        float.update_f64(&[1.5], None);
        assert_ne!(integer.digest(), float.digest());
        // A value longer than the buffer streams straight through, and gives the
        // digest of the same bytes pushed in pieces.
        let long = vec![b'x'; 3 * CHUNK_BYTES + 5];
        let mut one = ColumnFingerprint::new();
        one.update_bytes(kind::BINARY, Some(&long));
        let mut pieces = ColumnFingerprint::new();
        pieces.push(&[kind::BINARY, 1]);
        pieces.push(&(long.len() as u64).to_le_bytes());
        for piece in long.chunks(1000) {
            pieces.push(piece);
        }
        assert_eq!(one.digest(), pieces.digest());
        // The digest reads the buffer without consuming it.
        assert_eq!(one.digest(), one.digest());
    }

    #[test]
    fn digest_changes_with_value_order_and_nulls() {
        let digest = |values: &[f64], valid: Option<&[bool]>| {
            let mut fp = ColumnFingerprint::new();
            fp.update_f64(values, valid);
            fp.digest()
        };
        let base = digest(&[1.0, 2.0, 3.0], None);
        assert_ne!(base, digest(&[1.0, 2.0, 3.5], None));
        assert_ne!(base, digest(&[2.0, 1.0, 3.0], None));
        assert_ne!(base, digest(&[1.0, 2.0], None));
        assert_ne!(base, digest(&[1.0, 2.0, 3.0], Some(&[true, false, true])));
        assert_ne!(digest(&[0.0], None), digest(&[-0.0], None));
    }
}
