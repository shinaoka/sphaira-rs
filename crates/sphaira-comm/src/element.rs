//! Plain-old-data element trait for communicated buffers (data-structures §7.1).

use num_complex::Complex;

/// A plain-old-data element that can be communicated and summed.
///
/// Every element is flattened to `WORDS` 64-bit words for transport; this keeps all communicator
/// implementations free of `unsafe` and lets MPI move everything as `u64`.
/// Implemented for `f64`, `i64`, `u64`, `usize` and `Complex<f64>`.
///
/// `add` is the operation used by the ordered reductions. Floating-point types use ordinary IEEE
/// addition; integer types use wrapping addition (so reductions never panic and stay deterministic).
pub trait Element: Copy + Send + Sync + 'static {
    /// Number of 64-bit words per element.
    const WORDS: usize;
    /// Human-readable type name used in diagnostics.
    const NAME: &'static str;
    /// Writes the element into `out` (`out.len() == WORDS`).
    fn to_words(self, out: &mut [u64]);
    /// Reads an element from `w` (`w.len() == WORDS`).
    fn from_words(w: &[u64]) -> Self;
    /// The reduction operation `self + rhs`.
    fn add(self, rhs: Self) -> Self;
}

impl Element for f64 {
    const WORDS: usize = 1;
    const NAME: &'static str = "f64";
    fn to_words(self, out: &mut [u64]) {
        out[0] = self.to_bits();
    }
    fn from_words(w: &[u64]) -> Self {
        f64::from_bits(w[0])
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Element for i64 {
    const WORDS: usize = 1;
    const NAME: &'static str = "i64";
    fn to_words(self, out: &mut [u64]) {
        out[0] = self as u64;
    }
    fn from_words(w: &[u64]) -> Self {
        w[0] as i64
    }
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl Element for u64 {
    const WORDS: usize = 1;
    const NAME: &'static str = "u64";
    fn to_words(self, out: &mut [u64]) {
        out[0] = self;
    }
    fn from_words(w: &[u64]) -> Self {
        w[0]
    }
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl Element for usize {
    const WORDS: usize = 1;
    const NAME: &'static str = "usize";
    fn to_words(self, out: &mut [u64]) {
        out[0] = self as u64;
    }
    fn from_words(w: &[u64]) -> Self {
        w[0] as usize
    }
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl Element for Complex<f64> {
    const WORDS: usize = 2;
    const NAME: &'static str = "Complex<f64>";
    fn to_words(self, out: &mut [u64]) {
        out[0] = self.re.to_bits();
        out[1] = self.im.to_bits();
    }
    fn from_words(w: &[u64]) -> Self {
        Complex::new(f64::from_bits(w[0]), f64::from_bits(w[1]))
    }
    fn add(self, rhs: Self) -> Self {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

/// Flattens a slice of elements to words.
pub(crate) fn encode<T: Element>(xs: &[T]) -> Vec<u64> {
    let mut out = vec![0u64; xs.len() * T::WORDS];
    for (x, o) in xs.iter().zip(out.chunks_exact_mut(T::WORDS)) {
        x.to_words(o);
    }
    out
}

/// Reads words into a slice of elements (`words.len() == out.len() * WORDS`).
pub(crate) fn decode_into<T: Element>(words: &[u64], out: &mut [T]) {
    assert_eq!(words.len(), out.len() * T::WORDS, "word count mismatch");
    for (o, w) in out.iter_mut().zip(words.chunks_exact(T::WORDS)) {
        *o = T::from_words(w);
    }
}

/// Sums `size` equally long (`out.len()` elements) word blocks laid out back to back, strictly in
/// increasing block order: `((b0 + b1) + b2) + ...`, the first block being copied unchanged.
pub(crate) fn ordered_sum_words<T: Element>(gathered: &[u64], size: usize, out: &mut [T]) {
    let block = out.len() * T::WORDS;
    assert_eq!(gathered.len(), size * block, "gathered length mismatch");
    decode_into(&gathered[..block], out);
    for r in 1..size {
        let blk = &gathered[r * block..(r + 1) * block];
        for (o, w) in out.iter_mut().zip(blk.chunks_exact(T::WORDS)) {
            *o = o.add(T::from_words(w));
        }
    }
}
