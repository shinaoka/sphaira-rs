//! Per-species data bundle and global tables.
//!
//! Implements the data inventory of `neutral-atom.md` §1.2-1.4 (what is kept from the PAO and VPS
//! files), `pao-basis.md` (selection of radial functions) and the neutral-atom expansion maximum
//! angular momentum of `neutral-atom.md` §3 (`L_max = max l of basis + scf.BufferL.VNA`, default 6).

use std::path::Path;

use crate::basis::{BasisSpec, OrbitalLabel, RadialIndex};
use crate::bessel::BesselParams;
use crate::error::AtomDataError;
use crate::pao::PaoFile;
use crate::radial::RadialTable;
use crate::vps::{JAveraging, ScalarProjector, VpsFile};

/// Default `scf.BufferL.VNA` (`neutral-atom.md` §3 parameter table).
pub const DEFAULT_BUFFER_L: usize = 6;

/// Immutable data of one species: basis specification, the full PAO and VPS files, and the
/// derived scalar-relativistic projectors.
///
/// All stored radial functions of the PAO file are kept (the neutral-atom projector expansion is
/// seeded from all of them, `neutral-atom.md` §3.2); [`SpeciesData::basis_radial`] gives the
/// selected ones.
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::SpeciesData;
/// let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/DFT_DATA19");
/// let s = SpeciesData::from_files("C5.0-s1p1", false, &data.join("PAO"), &data.join("VPS/C_PBE19.vps"))
///     .unwrap();
/// assert_eq!(s.num_basis_functions(), 4);
/// assert_eq!(s.valence_charge(), 4.0);
/// assert_eq!(s.cutoff_radius(), 5.0);
/// ```
#[derive(Clone, Debug)]
pub struct SpeciesData {
    basis: BasisSpec,
    pao: PaoFile,
    vps: VpsFile,
    selected: Vec<RadialIndex>,
    scalar_projectors: Vec<ScalarProjector>,
}

impl SpeciesData {
    /// Bundles parsed data, validating the basis against the PAO file and `AtomSpecies` between
    /// the two files (not checked if either is 0, an empty atom). The scalar projectors use
    /// [`JAveraging::Projector`] (the book's default, `nonlocal-pseudopotential.md` §4.2).
    ///
    /// # Errors
    ///
    /// [`AtomDataError::BasisExceedsFile`] if the basis needs more than the PAO file stores,
    /// [`AtomDataError::AtomSpeciesMismatch`] if `AtomSpecies` differs by more than `1e-8`.
    pub fn new(basis: BasisSpec, pao: PaoFile, vps: VpsFile) -> Result<Self, AtomDataError> {
        let selected = pao.select_basis(&basis)?;
        let (a, b) = (pao.atom_species, vps.atom_species);
        if a != 0.0 && b != 0.0 && (a - b).abs() > 1e-8 {
            return Err(AtomDataError::AtomSpeciesMismatch { pao: a, vps: b });
        }
        let scalar_projectors = vps.scalar_projectors(JAveraging::Projector)?;
        Ok(Self {
            basis,
            pao,
            vps,
            selected,
            scalar_projectors,
        })
    }

    /// Reads `<pao_dir>/<name>.pao` and `vps_path` for the basis string `spec`.
    ///
    /// # Errors
    ///
    /// Any error of [`BasisSpec::parse`], [`PaoFile::read`], [`VpsFile::read`] or
    /// [`SpeciesData::new`].
    pub fn from_files(
        spec: &str,
        orbital_optimization: bool,
        pao_dir: &Path,
        vps_path: &Path,
    ) -> Result<Self, AtomDataError> {
        let basis = BasisSpec::parse(spec, orbital_optimization)?;
        let pao = PaoFile::read(pao_dir.join(basis.pao_file_name()))?;
        let vps = VpsFile::read(vps_path)?;
        Self::new(basis, pao, vps)
    }

    /// The basis specification.
    pub fn basis(&self) -> &BasisSpec {
        &self.basis
    }

    /// The PAO file (all stored radial functions, density, contraction blocks).
    pub fn pao(&self) -> &PaoFile {
        &self.pao
    }

    /// The VPS file as read (including j-dependent data).
    pub fn vps(&self) -> &VpsFile {
        &self.vps
    }

    /// PAO cutoff radius `r_c` (bohr), not the number in the file name.
    pub fn cutoff_radius(&self) -> f64 {
        self.pao.cutoff_radius
    }

    /// Upper radius `r_c + 0.5` bohr of the radial transform integral
    /// (`two-center-overlap-kinetic.md`, "Numerical parameters", (a)).
    pub fn transform_radius(&self) -> f64 {
        self.pao.cutoff_radius + 0.5
    }

    /// Effective ionic charge `Z_v` of the pseudopotential.
    pub fn valence_charge(&self) -> f64 {
        self.vps.valence_charge()
    }

    /// Largest `l` of the basis.
    pub fn l_max_basis(&self) -> usize {
        self.basis.l_max()
    }

    /// Number of basis functions on an atom of this species.
    pub fn num_basis_functions(&self) -> usize {
        self.basis.num_basis_functions()
    }

    /// Orbitals in the book's intra-atom order.
    pub fn orbitals(&self) -> Vec<OrbitalLabel> {
        self.basis.orbitals()
    }

    /// Selected radial functions (`l`, `zeta`) in basis order.
    pub fn selected_radial(&self) -> &[RadialIndex] {
        &self.selected
    }

    /// The `i`-th selected radial function (`i` indexes [`SpeciesData::selected_radial`]).
    ///
    /// # Panics
    ///
    /// If `i` is out of range.
    pub fn basis_radial(&self, i: usize) -> &RadialTable {
        let RadialIndex { l, zeta } = self.selected[i];
        &self.pao.radial[l][zeta]
    }

    /// Scalar-relativistic projectors (j-averaged with [`JAveraging::Projector`]).
    pub fn scalar_projectors(&self) -> &[ScalarProjector] {
        &self.scalar_projectors
    }
}

/// Tables shared by all species.
///
/// Holds the maximum angular momentum of the neutral-atom projector expansion
/// `L_max = (largest basis l over all species) + buffer` (`neutral-atom.md` §3, parameter table,
/// `scf.BufferL.VNA` = 6 by default) and the radial-transform parameters.
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::{GlobalTables, SpeciesData};
/// let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/DFT_DATA19");
/// let h = SpeciesData::from_files("H5.0-s1", false, &data.join("PAO"), &data.join("VPS/H_PBE19.vps")).unwrap();
/// let c = SpeciesData::from_files("C5.0-s1p1d1", false, &data.join("PAO"), &data.join("VPS/C_PBE19.vps")).unwrap();
/// let g = GlobalTables::new(&[h, c]);
/// assert_eq!(g.neutral_atom_l_max(), 2 + 6);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalTables {
    l_max_basis: usize,
    buffer_l: usize,
    bessel: BesselParams,
}

impl GlobalTables {
    /// Tables with the default buffer ([`DEFAULT_BUFFER_L`]) and default [`BesselParams`].
    pub fn new(species: &[SpeciesData]) -> Self {
        Self::with_parameters(species, DEFAULT_BUFFER_L, BesselParams::default())
    }

    /// Tables with explicit `scf.BufferL.VNA` and transform parameters. An empty species list
    /// gives `l_max_basis = 0`.
    pub fn with_parameters(species: &[SpeciesData], buffer_l: usize, bessel: BesselParams) -> Self {
        Self {
            l_max_basis: species
                .iter()
                .map(SpeciesData::l_max_basis)
                .max()
                .unwrap_or(0),
            buffer_l,
            bessel,
        }
    }

    /// Largest basis `l` over all species.
    pub fn l_max_basis(&self) -> usize {
        self.l_max_basis
    }

    /// Maximum `l` of the neutral-atom expansion: largest basis `l` plus the buffer.
    pub fn neutral_atom_l_max(&self) -> usize {
        self.l_max_basis + self.buffer_l
    }

    /// Radial-transform parameters.
    pub fn bessel_params(&self) -> &BesselParams {
        &self.bessel
    }
}

#[cfg(test)]
mod tests;
