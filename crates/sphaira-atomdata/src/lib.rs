//! PAO/VPS readers, real spherical harmonics, radial grids and spherical Bessel transforms.
//!
//! Built against the specification book commit recorded in `SPEC_VERSION`; each public item names
//! the chapter section it implements. Units are Hartree atomic units (`notation.md`, "Units").

pub mod basis;
pub mod bessel;
pub mod error;
pub mod pao;
pub mod radial;
pub mod species;
pub(crate) mod text;
pub mod vps;
pub mod ylm;

pub use basis::{BasisSpec, OrbitalLabel, RadialIndex, SHELL_LETTERS};
pub use bessel::{BesselParams, WaveNumberTable, radial_to_wavenumber, spherical_bessel_j};
pub use error::{AtomDataError, BasisSpecError};
pub use pao::{ContractionBlock, ContractionRow, PaoFile};
pub use radial::{InnerBehavior, LogGrid, OuterBehavior, RadialTable};
pub use species::{DEFAULT_BUFFER_L, GlobalTables, SpeciesData};
pub use vps::{
    JAveraging, JMinusPart, ScalarProjector, VpsFile, VpsFormat, VpsProjector, average_j_dependent,
    j_weights,
};
pub use ylm::{
    YlmComponent, YlmKind, component_order, real_ylm, real_ylm_into, real_ylm_with_gradient,
};
