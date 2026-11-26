//! VPS (pseudopotential) file reader and the scalar-relativistic j-averaging rule.
//!
//! Implements `nonlocal-pseudopotential.md` §1.3 (representation), §2 (VPS file format, new and
//! old) and §4 (treatment of j-dependent data in the scalar-relativistic case).

use std::path::Path;

use crate::error::AtomDataError;
use crate::radial::{InnerBehavior, OuterBehavior, RadialTable};
use crate::text::{Document, parse_floats, parse_rows, table_to_columns};

const KIND: &str = "VPS";

/// Sanity bound on `l` in `<project.energies` rows (the book states none).
const NEW_FORMAT_L_MAX: usize = 99;
/// Sanity bound on the projector count in `<project.energies`.
const MAX_PROJECTORS: usize = 1 << 20;
/// Largest `l` accepted in `<pseudo.NandL` rows of the old format (§2.6).
const OLD_FORMAT_L_MAX: usize = 29;

/// Which layout the file uses (`nonlocal-pseudopotential.md` §2.3 vs §2.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VpsFormat {
    /// Has a `<project.energies` block.
    New,
    /// No such block (`<projection.energies` and `<pseudo.NandL` are used).
    Old,
}

/// The `j = l - 1/2` part of a j-dependent projector.
#[derive(Clone, Debug)]
pub struct JMinusPart {
    /// Projector energy `epsilon_{p-}` (Hartree).
    pub energy: f64,
    /// Radial function `b_{p-}(r)`.
    pub radial: RadialTable,
}

/// A projector `p` of `nonlocal-pseudopotential.md` §1.3.
///
/// For j-dependent files `energy`/`radial` hold the `j = l + 1/2` part (`epsilon_{p+}`, `b_{p+}`)
/// and `j_minus` the `j = l - 1/2` part; otherwise `j_minus` is `None`. Values are stored as read.
#[derive(Clone, Debug)]
pub struct VpsProjector {
    /// Angular momentum `l_p`.
    pub l: usize,
    /// Energy `epsilon_p` (or `epsilon_{p+}`), Hartree.
    pub energy: f64,
    /// Radial function `b_p(r)` (or `b_{p+}`), Hartree^1/2 bohr^-3/2 for `epsilon = +-1`.
    pub radial: RadialTable,
    /// The `j = l - 1/2` part for j-dependent data.
    pub j_minus: Option<JMinusPart>,
}

/// A scalar (j-independent) projector: energy and radial function.
#[derive(Clone, Debug)]
pub struct ScalarProjector {
    /// Angular momentum.
    pub l: usize,
    /// Energy (Hartree).
    pub energy: f64,
    /// Radial function.
    pub radial: RadialTable,
}

/// How j-dependent projectors are reduced to scalar ones (`nonlocal-pseudopotential.md` §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JAveraging {
    /// Average projector radial functions and energies separately (§4.2, the default).
    #[default]
    Projector,
    /// Average the operator: two scalar projectors `w_+ eps_+, b_+` and `w_- eps_-, b_-` (§4.2 "operator average").
    Operator,
}

/// A VPS file.
///
/// Out-of-grid rules (§2.7): projectors are zero beyond the last point; inside the first point
/// `l = 0` projectors are extrapolated linearly from the first interval and `l >= 1` projectors
/// are 0. The local potential is held constant inside the first point (its continuation belongs to
/// the neutral-atom chapter). The PCC density is continued inside the first point by an even
/// quadratic `a r^2 + b` ([`InnerBehavior::EvenQuadratic`], matched in value and slope at the
/// first grid radius: an implementation choice, because the book does not fix the matching
/// point) and keeps its last tabulated value beyond the last point ([`OuterBehavior::Constant`]).
#[derive(Clone, Debug)]
pub struct VpsFile {
    /// New or old layout.
    pub format: VpsFormat,
    /// `AtomSpecies` (fractional for virtual atoms).
    pub atom_species: f64,
    /// `total.electron` (defaults to `AtomSpecies`).
    pub total_electron: f64,
    /// `valence.electron` (defaults to 0).
    pub valence_electron: f64,
    /// `j.dependent.pseudo.potentials`.
    pub j_dependent: bool,
    /// `grid.num.output`.
    pub grid_num: usize,
    /// Local potential `V_loc(r)` in Hartree.
    pub local_potential: RadialTable,
    /// Projectors in file order (the order defines the projector index).
    pub projectors: Vec<VpsProjector>,
    /// Partial-core-correction density (bohr^-3) when `charge.pcc.calc` is on.
    pub pcc_density: Option<RadialTable>,
}

impl VpsFile {
    /// Effective ionic charge `Z_v = valence.electron + AtomSpecies - total.electron` (§2.2).
    pub fn valence_charge(&self) -> f64 {
        self.valence_electron + self.atom_species - self.total_electron
    }

    /// Reads a VPS file from disk.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::Io`] if the file cannot be read, otherwise as [`VpsFile::parse`].
    pub fn read(path: impl AsRef<Path>) -> Result<Self, AtomDataError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| AtomDataError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Parses VPS file text (new or old format).
    ///
    /// # Errors
    ///
    /// A typed [`AtomDataError`] if `AtomSpecies` or `grid.num.output` is missing, a required block
    /// is missing (old format: `number.vps`, `local.part.vps`, `<pseudo.NandL`, and
    /// `<projection.energies` when a projector remains; [`AtomDataError::OldFormatNoLines`] for
    /// `number.vps` = 0; [`AtomDataError::TokenCount`] for missing or extra tokens in
    /// `<projection.energies`; an old-format `l` outside 0..=29), rows have the wrong number of columns (`3 + N_p`, or `3 + 2 N_p` for j-dependent
    /// data), row counts differ from `grid.num.output`, the PCC grid differs from the potential
    /// grid, an old-format file is j-dependent ([`AtomDataError::OldFormatJDependent`]), or the
    /// projector description is inconsistent (non-integer `l`, wrong count).
    pub fn parse(text: &str) -> Result<Self, AtomDataError> {
        let doc = Document::parse(text);
        let atom_species: f64 = doc.required_value(KIND, "AtomSpecies", "a number")?;
        let total_electron = doc
            .optional_value(KIND, "total.electron", "a number")?
            .unwrap_or(atom_species);
        let valence_electron = doc
            .optional_value(KIND, "valence.electron", "a number")?
            .unwrap_or(0.0);
        let grid_num: usize = doc.required_value(KIND, "grid.num.output", "a positive integer")?;
        let j_dependent = doc
            .optional_bool(KIND, "j.dependent.pseudo.potentials")?
            .unwrap_or(false);
        let pcc_on = doc.optional_bool(KIND, "charge.pcc.calc")?.unwrap_or(false);

        let (format, descr) = if let Some(block) = doc.block("project.energies") {
            (VpsFormat::New, parse_new_energies(block, j_dependent)?)
        } else {
            if j_dependent {
                return Err(AtomDataError::OldFormatJDependent);
            }
            (VpsFormat::Old, parse_old_description(&doc)?)
        };
        let np = descr.len();

        let ncols = 3 + if j_dependent { 2 * np } else { np };
        let pot_block = doc.required_block(KIND, "Pseudo.Potentials")?;
        let rows = parse_rows(KIND, pot_block, ncols)?;
        check_rows(&rows, grid_num, "Pseudo.Potentials")?;
        let (xs, rs, cols) = table_to_columns(&rows);
        let mut cols = cols.into_iter();
        // INVARIANT: `ncols >= 3` so at least the local-potential column exists.
        let local_potential = RadialTable::from_columns(
            &xs,
            &rs,
            cols.next().unwrap_or_default(),
            InnerBehavior::Constant,
        )?;
        let mut projectors = Vec::with_capacity(np);
        for (l, energies) in descr {
            let mut take = |energy: f64| -> Result<(f64, RadialTable), AtomDataError> {
                let col = cols.next().unwrap_or_default();
                Ok((
                    energy,
                    RadialTable::from_columns(&xs, &rs, col, InnerBehavior::projector(l))?,
                ))
            };
            let (energy, radial) = take(energies.0)?;
            let j_minus = match energies.1 {
                Some(em) => {
                    let (energy, radial) = take(em)?;
                    Some(JMinusPart { energy, radial })
                }
                None => None,
            };
            projectors.push(VpsProjector {
                l,
                energy,
                radial,
                j_minus,
            });
        }

        let pcc_density = if pcc_on {
            let block = doc.required_block(KIND, "density.PCC")?;
            let prow = parse_rows(KIND, block, 3)?;
            check_rows(&prow, grid_num, "density.PCC")?;
            let (pxs, _, mut pcols) = table_to_columns(&prow);
            if pxs.iter().zip(&xs).any(|(a, b)| (a - b).abs() > 1e-8) {
                return Err(AtomDataError::GridMismatch {
                    detail: "`density.PCC` grid differs from `Pseudo.Potentials`".into(),
                });
            }
            Some(
                RadialTable::new(
                    *local_potential.grid(),
                    pcols.remove(0),
                    InnerBehavior::EvenQuadratic,
                )?
                .with_outer(OuterBehavior::Constant),
            )
        } else {
            None
        };

        Ok(Self {
            format,
            atom_species,
            total_electron,
            valence_electron,
            j_dependent,
            grid_num,
            local_potential,
            projectors,
            pcc_density,
        })
    }

    /// Scalar projectors for a scalar-relativistic calculation (`nonlocal-pseudopotential.md` §4).
    ///
    /// Projectors without a `j_minus` part are returned unchanged.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::GridMismatch`] if the `+` and `-` radial functions are on different grids
    /// (cannot happen for tables read from one file).
    pub fn scalar_projectors(
        &self,
        mode: JAveraging,
    ) -> Result<Vec<ScalarProjector>, AtomDataError> {
        average_j_dependent(&self.projectors, mode)
    }
}

/// Weights `(w_+, w_-)` of `j = l +- 1/2`: `w_+ = (l+1)/(2l+1)`, `w_- = l/(2l+1)` (§4.1).
///
/// ```
/// use sphaira_atomdata::j_weights;
/// let (wp, wm) = j_weights(1);
/// assert!((wp - 2.0 / 3.0).abs() < 1e-15 && (wm - 1.0 / 3.0).abs() < 1e-15);
/// ```
pub fn j_weights(l: usize) -> (f64, f64) {
    let d = (2 * l + 1) as f64;
    ((l + 1) as f64 / d, l as f64 / d)
}

/// Applies the scalar-relativistic j-averaging rule of `nonlocal-pseudopotential.md` §4.2.
///
/// With [`JAveraging::Projector`] each j-dependent projector gives one scalar projector with
/// `b = w_+ b_+ + w_- b_-` and `eps = w_+ eps_+ + w_- eps_-`. With [`JAveraging::Operator`] it gives
/// two projectors `(w_+ eps_+, b_+)` and `(w_- eps_-, b_-)`, whose sum reproduces the spin average
/// of the j-dependent operator. Projectors without a `j_minus` part pass through.
///
/// # Errors
///
/// [`AtomDataError::GridMismatch`] if `+` and `-` tables are on different grids.
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::{average_j_dependent, JAveraging, JMinusPart, LogGrid, InnerBehavior,
///     RadialTable, VpsProjector};
/// let g = LogGrid::new(-3.0, 0.1, 20).unwrap();
/// let t = |c: f64| RadialTable::new(g, vec![c; 20], InnerBehavior::Zero).unwrap();
/// let p = VpsProjector { l: 1, energy: 1.0, radial: t(3.0),
///     j_minus: Some(JMinusPart { energy: -1.0, radial: t(6.0) }) };
/// let s = average_j_dependent(&[p], JAveraging::Projector).unwrap();
/// assert!((s[0].energy - 1.0 / 3.0).abs() < 1e-15); // 1/(2l+1) for opposite signs
/// assert!((s[0].radial.values()[0] - 4.0).abs() < 1e-14); // 2/3*3 + 1/3*6
/// ```
pub fn average_j_dependent(
    projectors: &[VpsProjector],
    mode: JAveraging,
) -> Result<Vec<ScalarProjector>, AtomDataError> {
    let mut out = Vec::with_capacity(projectors.len());
    for p in projectors {
        let Some(m) = &p.j_minus else {
            out.push(ScalarProjector {
                l: p.l,
                energy: p.energy,
                radial: p.radial.clone(),
            });
            continue;
        };
        let (wp, wm) = j_weights(p.l);
        match mode {
            JAveraging::Projector => out.push(ScalarProjector {
                l: p.l,
                energy: wp * p.energy + wm * m.energy,
                radial: RadialTable::combine(wp, &p.radial, wm, &m.radial)?,
            }),
            JAveraging::Operator => {
                out.push(ScalarProjector {
                    l: p.l,
                    energy: wp * p.energy,
                    radial: p.radial.clone(),
                });
                out.push(ScalarProjector {
                    l: p.l,
                    energy: wm * m.energy,
                    radial: m.radial.clone(),
                });
            }
        }
    }
    Ok(out)
}

/// (`l`, (`epsilon` or `epsilon_+`, optional `epsilon_-`)) per projector.
type Description = Vec<(usize, (f64, Option<f64>))>;

fn to_l(line: usize, v: f64, l_max: usize) -> Result<usize, AtomDataError> {
    if v >= 0.0 && v.fract() == 0.0 && v <= l_max as f64 {
        Ok(v as usize)
    } else {
        Err(AtomDataError::Parse {
            file_kind: KIND,
            line,
            text: v.to_string(),
            expected: "an angular momentum (non-negative integer)",
        })
    }
}

/// `<project.energies`: count line, then `l eps` or `l eps+ eps-` rows (§2.3).
fn parse_new_energies(
    block: &crate::text::Block<'_>,
    j_dependent: bool,
) -> Result<Description, AtomDataError> {
    let mut lines = block.lines.iter();
    let &(cl, cline) = lines.next().ok_or_else(|| AtomDataError::Inconsistent {
        file_kind: KIND,
        detail: "`<project.energies` is empty; the first line must be the number of projectors"
            .into(),
    })?;
    let first = parse_floats(KIND, cl, cline)?;
    let np = to_l(cl, first.first().copied().unwrap_or(-1.0), MAX_PROJECTORS)?;
    let cols = if j_dependent { 3 } else { 2 };
    let mut out = Vec::with_capacity(np);
    for &(no, line) in lines {
        let row = parse_floats(KIND, no, line)?;
        if row.len() != cols {
            return Err(AtomDataError::Columns {
                file_kind: KIND,
                line: no,
                expected: cols,
                found: row.len(),
            });
        }
        out.push((
            to_l(no, row[0], NEW_FORMAT_L_MAX)?,
            (row[1], row.get(2).copied()),
        ));
    }
    if out.len() != np {
        return Err(AtomDataError::RowCount {
            file_kind: KIND,
            block: "project.energies".into(),
            expected: np,
            found: out.len(),
        });
    }
    Ok(out)
}

/// Old format (§2.6): `number.vps` and `local.part.vps` (both required), `<pseudo.NandL` rows
/// (four or five numbers, only `l` used), `<projection.energies` (one `index eps` pair per
/// projector, no count line; required only if a projector remains).
fn parse_old_description(doc: &Document<'_>) -> Result<Description, AtomDataError> {
    let nvps: usize = doc.required_value(KIND, "number.vps", "a positive integer")?;
    if nvps == 0 {
        return Err(AtomDataError::OldFormatNoLines);
    }
    let local: usize = doc.required_value(KIND, "local.part.vps", "a non-negative integer")?;
    let nl = doc.required_block(KIND, "pseudo.NandL")?;
    if nl.lines.len() != nvps {
        return Err(AtomDataError::RowCount {
            file_kind: KIND,
            block: "pseudo.NandL".into(),
            expected: nvps,
            found: nl.lines.len(),
        });
    }
    let mut ls = Vec::new();
    for (i, &(no, line)) in nl.lines.iter().enumerate() {
        let row = parse_floats(KIND, no, line)?;
        if !(4..=5).contains(&row.len()) {
            return Err(AtomDataError::Columns {
                file_kind: KIND,
                line: no,
                expected: 4,
                found: row.len(),
            });
        }
        // Only the `l` of projector lines is used (and range-checked); the local-part line is not.
        if i != local {
            ls.push(to_l(no, row[2], OLD_FORMAT_L_MAX)?);
        }
    }
    let mut tokens = Vec::new();
    if let Some(eb) = doc.block("projection.energies") {
        for &(no, line) in &eb.lines {
            tokens.extend(parse_floats(KIND, no, line)?);
        }
    } else if !ls.is_empty() {
        return Err(AtomDataError::MissingBlock {
            file_kind: KIND,
            block: "projection.energies".into(),
        });
    }
    // INVARIANT: the pairs are read as one token stream; the index of each pair is ignored.
    if tokens.len() != 2 * ls.len() {
        return Err(AtomDataError::TokenCount {
            file_kind: KIND,
            block: "projection.energies".into(),
            expected: 2 * ls.len(),
            found: tokens.len(),
        });
    }
    Ok(ls
        .into_iter()
        .zip(tokens.chunks_exact(2))
        .map(|(l, pair)| (l, (pair[1], None)))
        .collect())
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

#[cfg(test)]
mod tests;
