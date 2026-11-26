//! Plain-old-data element trait for communicated buffers (data-structures §7.1).

use std::borrow::Cow;

use bytemuck::{cast_slice, cast_slice_mut};
use num_complex::Complex;

/// A plain-old-data element that can be communicated and summed.
///
/// Elements are transported as 64-bit words (`WORDS` per element). `f64`, `i64`, `u64` and
/// `Complex<f64>` are `bytemuck::Pod` with 8-byte alignment, so their buffers are reinterpreted as
/// `u64` slices without conversion (safe `bytemuck::cast_slice`, no `unsafe`). `usize` is converted
/// element by element (one temporary copy per call) so that it stays correct on every pointer width.
///
/// `add` is the operation of the ordered reductions. Floating-point types use ordinary IEEE
/// addition. Integer types use checked addition and **panic on overflow**, so an overflowing
/// reduction is never silently wrapped.
///
/// The word-level methods are implementation details of the communicators.
pub trait Element: Copy + Send + Sync + 'static {
    /// Number of 64-bit words per element.
    const WORDS: usize;
    /// Human-readable type name used in diagnostics.
    const NAME: &'static str;

    /// The reduction operation `self + rhs`.
    ///
    /// # Panics
    ///
    /// Integer types panic on overflow with a message naming the type.
    fn add(self, rhs: Self) -> Self;

    /// The elements as words, borrowed when no conversion is needed.
    #[doc(hidden)]
    fn as_words(xs: &[Self]) -> Cow<'_, [u64]>;

    /// Mutable word view of `xs` when no conversion is needed (`None` otherwise).
    #[doc(hidden)]
    fn as_words_mut(xs: &mut [Self]) -> Option<&mut [u64]>;

    /// Reads `words` (`out.len() * WORDS` of them) into `out`.
    #[doc(hidden)]
    fn from_words(words: &[u64], out: &mut [Self]);
}

macro_rules! pod_element {
    ($t:ty, $words:expr, $name:expr, |$a:ident, $b:ident| $add:expr) => {
        impl Element for $t {
            const WORDS: usize = $words;
            const NAME: &'static str = $name;
            fn add(self, rhs: Self) -> Self {
                let ($a, $b) = (self, rhs);
                $add
            }
            fn as_words(xs: &[Self]) -> Cow<'_, [u64]> {
                Cow::Borrowed(cast_slice(xs))
            }
            fn as_words_mut(xs: &mut [Self]) -> Option<&mut [u64]> {
                Some(cast_slice_mut(xs))
            }
            fn from_words(words: &[u64], out: &mut [Self]) {
                assert_eq!(words.len(), out.len() * Self::WORDS, "word count mismatch");
                cast_slice_mut::<Self, u64>(out).copy_from_slice(words);
            }
        }
    };
}

pod_element!(f64, 1, "f64", |a, b| a + b);
pod_element!(Complex<f64>, 2, "Complex<f64>", |a, b| a + b);
pod_element!(u64, 1, "u64", |a, b| a
    .checked_add(b)
    .expect("u64 overflow in ordered reduction; reduce in a wider type"));
pod_element!(i64, 1, "i64", |a, b| a
    .checked_add(b)
    .expect("i64 overflow in ordered reduction; reduce in a wider type"));

impl Element for usize {
    const WORDS: usize = 1;
    const NAME: &'static str = "usize";
    fn add(self, rhs: Self) -> Self {
        self.checked_add(rhs)
            .expect("usize overflow in ordered reduction; reduce in a wider type")
    }
    fn as_words(xs: &[Self]) -> Cow<'_, [u64]> {
        Cow::Owned(xs.iter().map(|&x| x as u64).collect())
    }
    fn as_words_mut(_xs: &mut [Self]) -> Option<&mut [u64]> {
        None
    }
    fn from_words(words: &[u64], out: &mut [Self]) {
        assert_eq!(words.len(), out.len(), "word count mismatch");
        for (o, &w) in out.iter_mut().zip(words) {
            // INVARIANT: the words were produced from `usize` values by `as_words`.
            *o = w as usize;
        }
    }
}

/// Flattens a slice of elements to owned words (one copy).
pub(crate) fn encode<T: Element>(xs: &[T]) -> Vec<u64> {
    T::as_words(xs).into_owned()
}

/// Reads words into a slice of elements.
pub(crate) fn decode_into<T: Element>(words: &[u64], out: &mut [T]) {
    T::from_words(words, out);
}

/// Sums `size` equally long (`out.len()` elements) word blocks laid out back to back, strictly in
/// increasing block order: `((b0 + b1) + b2) + ...`, the first block being copied unchanged.
pub(crate) fn ordered_sum_words<T: Element>(gathered: &[u64], size: usize, out: &mut [T]) {
    let block = out.len() * T::WORDS;
    assert_eq!(gathered.len(), size * block, "gathered length mismatch");
    T::from_words(&gathered[..block], out);
    if size == 1 {
        return;
    }
    let mut scratch = out.to_vec();
    for r in 1..size {
        T::from_words(&gathered[r * block..(r + 1) * block], &mut scratch);
        for (o, &x) in out.iter_mut().zip(&scratch) {
            *o = o.add(x);
        }
    }
}
