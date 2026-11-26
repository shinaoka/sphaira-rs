# sphaira-rs

A Rust implementation of an electronic-structure method in the style of OpenMX: numerical pseudo-atomic orbitals,
norm-conserving pseudopotentials and real-space grids, designed for distributed memory from the start.

sphaira-rs is **not** OpenMX and not an official successor of OpenMX. It is implemented from the specifications in
[ab-initio-common-rust-substrate](https://github.com/shinaoka/ab-initio-common-rust-substrate) under a
[clean-room procedure](https://github.com/shinaoka/ab-initio-common-rust-substrate/blob/main/src/process/clean-room.md).

## Status

Planning. No code yet. The plan is tracked in the issues.

First goal: a self-consistent field (SCF) calculation (Cluster and Band, non-spin-polarized and collinear spin,
LDA and GGA-PBE) whose total energy agrees with the reference implementation on small systems.

## Specification version

The specification version this repository is built against is recorded in [`SPEC_VERSION`](SPEC_VERSION)
(a commit of the specification book).

Key design chapters:

- [Data structures for a parallel SCF](https://github.com/shinaoka/ab-initio-common-rust-substrate/blob/main/src/methods/lcao-openmx/design/data-structures.md)
- [A visual guide to the data structures](https://github.com/shinaoka/ab-initio-common-rust-substrate/blob/main/src/methods/lcao-openmx/design/data-structures-guide.md)

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
