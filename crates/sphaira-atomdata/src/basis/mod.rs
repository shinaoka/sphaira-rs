//! Basis-specification strings such as `C6.0-s2p2d1`.
//!
//! Implements `pao-basis.md`, "Basis specification" (string format, interpretation, selection of
//! radial functions) and "Ordering of orbitals on an atom".

use crate::error::BasisSpecError;

/// Shell letters for `l = 0..=4`.
pub const SHELL_LETTERS: [char; 5] = ['s', 'p', 'd', 'f', 'g'];

/// Index of a stored radial function: angular momentum `l` and multiplicity `zeta` (0-based).
///
/// The radial functions of a basis are the first `N_l` of each `l` in the file
/// (`pao-basis.md`, "Radial functions selected from the file").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RadialIndex {
    /// Angular momentum.
    pub l: usize,
    /// Multiplicity index `zeta`, counted from 0.
    pub zeta: usize,
}

/// One orbital on an atom: radial function and component index of the real spherical harmonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrbitalLabel {
    /// Angular momentum.
    pub l: usize,
    /// Multiplicity index `zeta`.
    pub zeta: usize,
    /// Component index `0..2l+1` in the book's ordering of real spherical harmonics.
    pub component: usize,
}

/// Parsed basis specification.
///
/// Implements `pao-basis.md`, "String format" and "Interpretation of the string".
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::BasisSpec;
/// let b = BasisSpec::parse("C6.0-s2p2d1", false).unwrap();
/// assert_eq!(b.pao_file_name(), "C6.0.pao");
/// assert_eq!((b.count(0), b.count(1), b.count(2)), (2, 2, 1));
/// assert_eq!(b.num_basis_functions(), 13);
/// assert_eq!(b.l_max(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisSpec {
    spec: String,
    name: String,
    counts: [usize; 5],
    primitives: [usize; 5],
}

impl BasisSpec {
    /// Parses a specification.
    ///
    /// The part before the last `-` is the PAO file name without `.pao`. After it come shell
    /// letters `s p d f g` (lowercase, each at most once, any order) followed by a count; `a>b`
    /// denotes `a` original and `b` contracted radial functions and is accepted when
    /// `orbital_optimization` is true or `a == b`. Unwritten shells have count 0 (zero counts may
    /// also be written explicitly).
    ///
    /// # Errors
    ///
    /// A [`BasisSpecError`] for a missing `-`, empty name or shell list, unknown characters, a
    /// missing or duplicated count, a count above 99, a contraction without optimization or with
    /// `b == 0` or `b > a`, or when no count is nonzero.
    pub fn parse(spec: &str, orbital_optimization: bool) -> Result<Self, BasisSpecError> {
        let spec = spec.trim();
        let err_spec = || spec.to_string();
        let dash = spec
            .rfind('-')
            .ok_or_else(|| BasisSpecError::MissingSeparator { spec: err_spec() })?;
        let (name, body) = (&spec[..dash], &spec[dash + 1..]);
        if name.is_empty() || name.chars().any(char::is_whitespace) {
            return Err(BasisSpecError::BadName { spec: err_spec() });
        }
        if body.is_empty() {
            return Err(BasisSpecError::EmptyShells { spec: err_spec() });
        }
        let mut counts = [0usize; 5];
        let mut primitives = [0usize; 5];
        let mut seen = [false; 5];
        let bytes = body.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            let ch = bytes[pos] as char;
            let Some(l) = SHELL_LETTERS.iter().position(|&c| c == ch) else {
                return Err(BasisSpecError::UnexpectedCharacter {
                    spec: err_spec(),
                    ch: body[pos..].chars().next().unwrap_or(ch),
                    position: dash + 1 + pos,
                });
            };
            if seen[l] {
                return Err(BasisSpecError::DuplicateShell {
                    spec: err_spec(),
                    letter: ch,
                });
            }
            seen[l] = true;
            pos += 1;
            let original = read_count(body, &mut pos, spec, ch)?.ok_or_else(|| {
                BasisSpecError::MissingCount {
                    spec: err_spec(),
                    letter: ch,
                }
            })?;
            let mut contracted = original;
            if pos < bytes.len() && bytes[pos] == b'>' {
                pos += 1;
                contracted = read_count(body, &mut pos, spec, ch)?.ok_or_else(|| {
                    BasisSpecError::MissingCount {
                        spec: err_spec(),
                        letter: ch,
                    }
                })?;
                if contracted == 0 || contracted > original {
                    return Err(BasisSpecError::InvalidContraction {
                        spec: err_spec(),
                        letter: ch,
                        original,
                        contracted,
                    });
                }
                if contracted != original && !orbital_optimization {
                    return Err(BasisSpecError::ContractionWithoutOptimization {
                        spec: err_spec(),
                        letter: ch,
                        original,
                        contracted,
                    });
                }
            }
            counts[l] = contracted;
            primitives[l] = original;
        }
        if counts.iter().all(|&n| n == 0) {
            return Err(BasisSpecError::NoOrbitals { spec: err_spec() });
        }
        Ok(Self {
            spec: spec.to_string(),
            name: name.to_string(),
            counts,
            primitives,
        })
    }

    /// The specification text.
    pub fn as_str(&self) -> &str {
        &self.spec
    }

    /// PAO file name without extension (`C6.0`).
    pub fn species_file(&self) -> &str {
        &self.name
    }

    /// PAO file name with extension (`C6.0.pao`).
    pub fn pao_file_name(&self) -> String {
        format!("{}.pao", self.name)
    }

    /// `N_l`: number of radial functions (after contraction) for angular momentum `l`; 0 for `l > 4`.
    pub fn count(&self, l: usize) -> usize {
        self.counts.get(l).copied().unwrap_or(0)
    }

    /// Number of original (primitive) radial functions of `l` (equal to `count` without `>`).
    pub fn primitive_count(&self, l: usize) -> usize {
        self.primitives.get(l).copied().unwrap_or(0)
    }

    /// True if any shell is written `a>b` with `a != b`.
    pub fn is_contracted(&self) -> bool {
        self.counts != self.primitives
    }

    /// Largest `l` with `N_l > 0` (`pao-basis.md`, "Interpretation of the string").
    pub fn l_max(&self) -> usize {
        // INVARIANT: `parse` rejects specifications with all counts zero.
        self.counts.iter().rposition(|&n| n > 0).unwrap_or(0)
    }

    /// `sum_l (2l+1) N_l`.
    pub fn num_basis_functions(&self) -> usize {
        self.counts
            .iter()
            .enumerate()
            .map(|(l, &n)| (2 * l + 1) * n)
            .sum()
    }

    /// Selected radial functions: for each `l` ascending, the first `N_l` (`zeta = 0..N_l`).
    pub fn radial_indices(&self) -> Vec<RadialIndex> {
        (0..self.counts.len())
            .flat_map(|l| (0..self.counts[l]).map(move |zeta| RadialIndex { l, zeta }))
            .collect()
    }

    /// Orbitals on an atom in the book's order: `l`, then `zeta`, then component
    /// (`pao-basis.md`, "Ordering of orbitals on an atom").
    pub fn orbitals(&self) -> Vec<OrbitalLabel> {
        self.radial_indices()
            .into_iter()
            .flat_map(|RadialIndex { l, zeta }| {
                (0..2 * l + 1).map(move |component| OrbitalLabel { l, zeta, component })
            })
            .collect()
    }
}

/// Reads a run of ASCII digits at `*pos`; `Ok(None)` if there is none.
fn read_count(
    body: &str,
    pos: &mut usize,
    spec: &str,
    letter: char,
) -> Result<Option<usize>, BasisSpecError> {
    let start = *pos;
    while body.as_bytes().get(*pos).is_some_and(u8::is_ascii_digit) {
        *pos += 1;
    }
    if *pos == start {
        return Ok(None);
    }
    match body[start..*pos].parse::<usize>() {
        Ok(n) if n <= 99 => Ok(Some(n)),
        _ => Err(BasisSpecError::CountTooLarge {
            spec: spec.to_string(),
            letter,
        }),
    }
}

#[cfg(test)]
mod tests;
