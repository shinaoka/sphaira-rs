//! PAO file reader.
//!
//! Implements `pao-basis.md`, "PAO file format" (items read, table of radial functions,
//! contraction-coefficient block) and "Values outside the grid".

use std::path::Path;

use crate::basis::{BasisSpec, RadialIndex};
use crate::error::AtomDataError;
use crate::radial::{InnerBehavior, RadialTable};
use crate::text::{Document, parse_floats, parse_rows, table_to_columns};

const KIND: &str = "PAO";

/// One row of a `<Contraction.coefficients...` block: `Atom= L= Mul= p= coefficient`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContractionRow {
    /// Atom index as written.
    pub atom: usize,
    /// Angular momentum.
    pub l: usize,
    /// Contracted function index.
    pub mul: usize,
    /// Index of the original PAO.
    pub p: usize,
    /// Coefficient.
    pub coefficient: f64,
}

/// A contraction-coefficient block (a record of how the file was made; not used to read the basis).
///
/// Implements `pao-basis.md`, "Contraction-coefficient block".
#[derive(Clone, Debug, PartialEq)]
pub struct ContractionBlock {
    /// Block name as written, e.g. `contraction.coefficients1` (lowercased).
    pub name: String,
    /// All rows in file order.
    pub rows: Vec<ContractionRow>,
}

/// A PAO file: all stored radial functions and the data of `pao-basis.md`, "Items read".
///
/// Radial functions are `R(r)` itself (not `r R(r)`). `radial(l, zeta)` tables carry the cutoff
/// radius, so they vanish for `r >= r_c`; below the first grid point `l = 0` tables return the
/// first value and `l >= 1` tables return 0; above the last grid point everything is 0
/// ("Values outside the grid").
#[derive(Clone, Debug)]
pub struct PaoFile {
    /// `AtomSpecies`: atomic number (fractional for virtual atoms).
    pub atom_species: f64,
    /// `PAO.Lmax`: maximum angular momentum of the stored radial functions.
    pub lmax: usize,
    /// `PAO.Mul`: number of stored radial functions per `l`.
    pub mul: usize,
    /// `radial.cutoff.pao`: cutoff radius `r_c` in bohr (default 5.0 if absent).
    pub cutoff_radius: f64,
    /// `grid.num.output`: number of radial grid points.
    pub grid_num: usize,
    /// `number.optpao` if present.
    pub number_optpao: Option<usize>,
    /// Atomic valence density `n^a(r)` of `<valence.charge.density` (bohr^-3), as read
    /// (not renormalized), if the block is present.
    pub valence_density: Option<RadialTable>,
    /// All stored radial functions, indexed `[l][zeta]`, `l = 0..=lmax`, `zeta = 0..mul`.
    pub radial: Vec<Vec<RadialTable>>,
    /// Contraction-coefficient blocks in file order (empty if none).
    pub contraction_blocks: Vec<ContractionBlock>,
}

impl PaoFile {
    /// Reads a PAO file from disk.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::Io`] if the file cannot be read, otherwise as [`PaoFile::parse`].
    pub fn read(path: impl AsRef<Path>) -> Result<Self, AtomDataError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| AtomDataError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Parses PAO file text.
    ///
    /// # Errors
    ///
    /// A typed [`AtomDataError`] if `AtomSpecies`, `PAO.Lmax`, `PAO.Mul` or `grid.num.output` is
    /// missing or unparsable, a `<pseudo.atomic.orbitals.L=l` block for some `l <= PAO.Lmax` is
    /// missing, a table row has not exactly `PAO.Mul + 2` columns (3 for the density), a table
    /// does not have `grid.num.output` rows, the grid is not uniform in `ln r` or `r != exp(x)`, or
    /// the tables of different blocks are on different grids.
    pub fn parse(text: &str) -> Result<Self, AtomDataError> {
        let doc = Document::parse(text);
        let atom_species: f64 = doc.required_value(KIND, "AtomSpecies", "a number")?;
        let lmax: usize = doc.required_value(KIND, "PAO.Lmax", "a non-negative integer")?;
        let mul: usize = doc.required_value(KIND, "PAO.Mul", "a positive integer")?;
        let cutoff_radius: f64 = doc
            .optional_value(KIND, "radial.cutoff.pao", "a number")?
            .unwrap_or(5.0);
        let grid_num: usize = doc.required_value(KIND, "grid.num.output", "a positive integer")?;
        let number_optpao = doc.optional_value(KIND, "number.optpao", "a non-negative integer")?;

        let mut ref_xs: Option<Vec<f64>> = None;
        let mut check_grid = |xs: &[f64], block: &str| -> Result<(), AtomDataError> {
            match &ref_xs {
                None => ref_xs = Some(xs.to_vec()),
                Some(r) => {
                    if r.iter().zip(xs).any(|(a, b)| (a - b).abs() > 1e-8) {
                        return Err(AtomDataError::GridMismatch {
                            detail: format!("block `{block}` differs from the first table"),
                        });
                    }
                }
            }
            Ok(())
        };

        let mut radial = Vec::with_capacity(lmax + 1);
        for l in 0..=lmax {
            let name = format!("pseudo.atomic.orbitals.L={l}");
            let block = doc.required_block(KIND, &name)?;
            let rows = parse_rows(KIND, block, mul + 2)?;
            check_rows(&rows, grid_num, &name)?;
            let (xs, rs, cols) = table_to_columns(&rows);
            check_grid(&xs, &name)?;
            let tables = cols
                .into_iter()
                .map(|c| {
                    Ok(
                        RadialTable::from_columns(&xs, &rs, c, InnerBehavior::pao(l))?
                            .with_cutoff(cutoff_radius),
                    )
                })
                .collect::<Result<Vec<_>, AtomDataError>>()?;
            radial.push(tables);
        }

        let valence_density = match doc.block("valence.charge.density") {
            None => None,
            Some(block) => {
                let rows = parse_rows(KIND, block, 3)?;
                check_rows(&rows, grid_num, "valence.charge.density")?;
                let (xs, rs, mut cols) = table_to_columns(&rows);
                check_grid(&xs, "valence.charge.density")?;
                Some(RadialTable::from_columns(
                    &xs,
                    &rs,
                    cols.remove(0),
                    InnerBehavior::Constant,
                )?)
            }
        };

        let contraction_blocks = doc
            .blocks_with_prefix("contraction.coefficients")
            .map(parse_contraction)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            atom_species,
            lmax,
            mul,
            cutoff_radius,
            grid_num,
            number_optpao,
            valence_density,
            radial,
            contraction_blocks,
        })
    }

    /// Stored radial function `R_{l zeta}`, or `None` if out of range.
    pub fn radial_function(&self, l: usize, zeta: usize) -> Option<&RadialTable> {
        self.radial.get(l)?.get(zeta)
    }

    /// Radial functions selected by a basis specification: the first `N_l` of each `l`, in the
    /// order of [`BasisSpec::radial_indices`] (`pao-basis.md`, "Radial functions selected from the
    /// file"). No reordering is done.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::BasisExceedsFile`] if `l_max` of the basis exceeds `PAO.Lmax` or some
    /// `N_l` (or primitive count) exceeds `PAO.Mul`.
    pub fn select_basis(&self, spec: &BasisSpec) -> Result<Vec<RadialIndex>, AtomDataError> {
        if spec.l_max() > self.lmax {
            return Err(AtomDataError::BasisExceedsFile {
                spec: spec.as_str().into(),
                what: format!("l_max = {}", spec.l_max()),
                available: format!("PAO.Lmax = {}", self.lmax),
            });
        }
        for l in 0..=spec.l_max() {
            let n = spec.count(l).max(spec.primitive_count(l));
            if n > self.mul {
                return Err(AtomDataError::BasisExceedsFile {
                    spec: spec.as_str().into(),
                    what: format!("{n} radial functions for l = {l}"),
                    available: format!("PAO.Mul = {}", self.mul),
                });
            }
        }
        Ok(spec.radial_indices())
    }
}

fn check_rows(rows: &[Vec<f64>], expected: usize, block: &str) -> Result<(), AtomDataError> {
    if rows.len() != expected {
        return Err(AtomDataError::RowCount {
            file_kind: KIND,
            block: block.into(),
            expected,
            found: rows.len(),
        });
    }
    Ok(())
}

/// Parses a contraction block: a count line followed by `Atom= a L= l Mul= m p= p coeff` rows.
fn parse_contraction(block: &crate::text::Block<'_>) -> Result<ContractionBlock, AtomDataError> {
    let mut lines = block.lines.iter();
    let &(cl, count_line) = lines.next().ok_or_else(|| AtomDataError::Inconsistent {
        file_kind: KIND,
        detail: format!(
            "block `{}` is empty; it must start with a row count",
            block.name
        ),
    })?;
    let announced = parse_floats(KIND, cl, count_line)?;
    if announced.len() != 1 {
        return Err(AtomDataError::Inconsistent {
            file_kind: KIND,
            detail: format!(
                "line {cl}: the first line of `{}` must be the row count",
                block.name
            ),
        });
    }
    let mut rows = Vec::new();
    for &(no, line) in lines {
        let spaced = line.replace('=', "= ");
        let t: Vec<&str> = spaced.split_whitespace().collect();
        let bad = |text: &str| AtomDataError::Parse {
            file_kind: KIND,
            line: no,
            text: text.to_string(),
            expected: "`Atom= <n> L= <l> Mul= <m> p= <p> <coefficient>`",
        };
        if t.len() != 9 || t[0] != "Atom=" || t[2] != "L=" || t[4] != "Mul=" || t[6] != "p=" {
            return Err(bad(line));
        }
        let int = |s: &str| s.parse::<usize>().map_err(|_| bad(s));
        rows.push(ContractionRow {
            atom: int(t[1])?,
            l: int(t[3])?,
            mul: int(t[5])?,
            p: int(t[7])?,
            coefficient: t[8].parse::<f64>().map_err(|_| bad(t[8]))?,
        });
    }
    if rows.len() as f64 != announced[0] {
        return Err(AtomDataError::RowCount {
            file_kind: KIND,
            block: block.name.clone(),
            expected: announced[0] as usize,
            found: rows.len(),
        });
    }
    Ok(ContractionBlock {
        name: block.name.clone(),
        rows,
    })
}

#[cfg(test)]
mod tests;
