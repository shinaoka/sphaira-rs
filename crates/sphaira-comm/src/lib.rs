//! Communicator trait with single-rank, in-process simulated and MPI implementations.
//!
//! Implements data-structures §2 (principle 4), §3 (index spaces, module [`ids`]) and §7
//! (communication: [`Comm`], [`SelfComm`], [`SimComm`], `MpiComm`), with the deterministic
//! reductions of §7.3.

mod comm;
mod element;
pub mod ids;
mod self_comm;
mod sim;

#[cfg(feature = "mpi")]
mod mpi_comm;

pub use comm::Comm;
pub use element::Element;
pub use ids::{AtomId, Cell, GridIndex, KIndex, KTask, OrbId, PairId, SpeciesId, Spin};
#[cfg(feature = "mpi")]
pub use mpi_comm::MpiComm;
pub use self_comm::SelfComm;
pub use sim::SimComm;
