//! Typed errors of the crate.

use thiserror::Error;

/// Errors of the basis-specification string parser.
///
/// Implements the string format of the PAO basis chapter, "Basis specification"
/// (`pao-basis.md`, "String format" and "Interpretation of the string").
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BasisSpecError {
    /// The string has no `-` between the PAO file name and the shell list.
    #[error(
        "basis specification `{spec}` has no `-`; write it as `<pao-name>-<shells>`, e.g. `C5.0-s1p1`"
    )]
    MissingSeparator {
        /// The offending specification.
        spec: String,
    },
    /// The PAO file name before `-` is empty or contains whitespace.
    #[error(
        "basis specification `{spec}` has an empty or malformed PAO file name; start it with e.g. `C5.0-`"
    )]
    BadName {
        /// The offending specification.
        spec: String,
    },
    /// Nothing follows the `-`.
    #[error("basis specification `{spec}` has no shell list after `-`; append e.g. `s1p1`")]
    EmptyShells {
        /// The offending specification.
        spec: String,
    },
    /// A character that is neither a shell letter, a digit nor `>`.
    #[error(
        "basis specification `{spec}`: unexpected `{ch}` at byte {position}; shells are written with the letters s, p, d, f, g followed by a count"
    )]
    UnexpectedCharacter {
        /// The offending specification.
        spec: String,
        /// The unexpected character.
        ch: char,
        /// Byte offset in the specification.
        position: usize,
    },
    /// A shell letter without a following count.
    #[error("basis specification `{spec}`: shell `{letter}` has no count; write e.g. `{letter}1`")]
    MissingCount {
        /// The offending specification.
        spec: String,
        /// The shell letter.
        letter: char,
    },
    /// The same shell letter appears twice.
    #[error(
        "basis specification `{spec}`: shell `{letter}` appears more than once; merge the counts"
    )]
    DuplicateShell {
        /// The offending specification.
        spec: String,
        /// The shell letter.
        letter: char,
    },
    /// A count that does not fit in the supported range.
    #[error("basis specification `{spec}`: count of shell `{letter}` is too large; use at most 99")]
    CountTooLarge {
        /// The offending specification.
        spec: String,
        /// The shell letter.
        letter: char,
    },
    /// `a>b` with `a != b` although orbital optimization is not in use.
    #[error(
        "basis specification `{spec}`: contraction `{letter}{original}>{contracted}` requires orbital optimization; remove `>{contracted}` or enable orbital optimization"
    )]
    ContractionWithoutOptimization {
        /// The offending specification.
        spec: String,
        /// The shell letter.
        letter: char,
        /// Number of original radial functions.
        original: usize,
        /// Number after contraction.
        contracted: usize,
    },
    /// `a>b` with `b == 0` or `b > a`.
    #[error(
        "basis specification `{spec}`: contraction `{letter}{original}>{contracted}` is invalid; the contracted count must satisfy 1 <= count <= original"
    )]
    InvalidContraction {
        /// The offending specification.
        spec: String,
        /// The shell letter.
        letter: char,
        /// Number of original radial functions.
        original: usize,
        /// Number after contraction.
        contracted: usize,
    },
    /// All counts are zero.
    #[error(
        "basis specification `{spec}` contains no basis function; give at least one nonzero count"
    )]
    NoOrbitals {
        /// The offending specification.
        spec: String,
    },
}

/// Errors of the readers and tables of this crate.
#[derive(Debug, Error)]
pub enum AtomDataError {
    /// A file could not be read.
    #[error("cannot read `{path}`: {source}; check that the file exists and is readable")]
    Io {
        /// Path of the file.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A required keyword is absent.
    #[error(
        "{file_kind} file: required keyword `{keyword}` is missing; add a line `{keyword} <value>`"
    )]
    MissingKeyword {
        /// `"PAO"` or `"VPS"`.
        file_kind: &'static str,
        /// The keyword.
        keyword: String,
    },
    /// A required block is absent or not terminated.
    #[error(
        "{file_kind} file: required block `<{block}` ... `{block}>` is missing or not terminated; check the file is complete"
    )]
    MissingBlock {
        /// `"PAO"` or `"VPS"`.
        file_kind: &'static str,
        /// The block name.
        block: String,
    },
    /// A token could not be parsed.
    #[error("{file_kind} file, line {line}: cannot parse `{text}` as {expected}; fix the value")]
    Parse {
        /// `"PAO"` or `"VPS"`.
        file_kind: &'static str,
        /// 1-based line number.
        line: usize,
        /// Offending text.
        text: String,
        /// What was expected.
        expected: &'static str,
    },
    /// A table row has the wrong number of columns.
    #[error(
        "{file_kind} file, line {line}: expected {expected} columns, found {found}; the file does not match the documented layout"
    )]
    Columns {
        /// `"PAO"` or `"VPS"`.
        file_kind: &'static str,
        /// 1-based line number.
        line: usize,
        /// Expected number of columns.
        expected: usize,
        /// Found number of columns.
        found: usize,
    },
    /// A table has a different number of rows than announced.
    #[error(
        "{file_kind} file: block `{block}` has {found} rows but {expected} were announced; fix `grid.num.output` or the table"
    )]
    RowCount {
        /// `"PAO"` or `"VPS"`.
        file_kind: &'static str,
        /// The block name.
        block: String,
        /// Expected number of rows.
        expected: usize,
        /// Found number of rows.
        found: usize,
    },
    /// The radial grid is not uniform in ln r, or r != exp(x), or the grid is too short.
    #[error(
        "invalid radial grid: {detail}; the table must be equally spaced in x = ln r with at least 4 points"
    )]
    Grid {
        /// Description of the violation.
        detail: String,
    },
    /// Two tables expected on the same grid are on different grids.
    #[error("radial tables live on different grids ({detail}); use tables from the same file")]
    GridMismatch {
        /// Description of the mismatch.
        detail: String,
    },
    /// Basis specification error.
    #[error(transparent)]
    BasisSpec(#[from] BasisSpecError),
    /// The basis asks for more than the PAO file stores.
    #[error(
        "basis `{spec}` needs {what} but the PAO file stores {available}; choose a smaller basis or another PAO file"
    )]
    BasisExceedsFile {
        /// Basis specification text.
        spec: String,
        /// What is requested.
        what: String,
        /// What the file offers.
        available: String,
    },
    /// `AtomSpecies` differs between PAO and VPS files.
    #[error(
        "AtomSpecies differs between the PAO file ({pao}) and the VPS file ({vps}); use files of the same element"
    )]
    AtomSpeciesMismatch {
        /// Value in the PAO file.
        pao: f64,
        /// Value in the VPS file.
        vps: f64,
    },
    /// An old-format VPS file declares j-dependent data.
    #[error(
        "VPS file is in the old format (no `<project.energies` block) but `j.dependent.pseudo.potentials` is on; use a new-format file"
    )]
    OldFormatJDependent,
    /// Other inconsistency in file content.
    #[error("{file_kind} file: {detail}")]
    Inconsistent {
        /// `"PAO"` or `"VPS"`.
        file_kind: &'static str,
        /// Description.
        detail: String,
    },
}
