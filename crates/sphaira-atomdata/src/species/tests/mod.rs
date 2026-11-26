use std::path::PathBuf;

use crate::{AtomDataError, BasisSpec, GlobalTables, PaoFile, SpeciesData, VpsFile};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/DFT_DATA19")
}

fn species(spec: &str, vps: &str) -> SpeciesData {
    SpeciesData::from_files(
        spec,
        false,
        &data_dir().join("PAO"),
        &data_dir().join("VPS").join(vps),
    )
    .unwrap()
}

#[test]
fn bundle_contents() {
    let c = species("C5.0-s2p2d1", "C_PBE19.vps");
    assert_eq!(c.num_basis_functions(), 13);
    assert_eq!(c.selected_radial().len(), 5);
    assert_eq!(c.l_max_basis(), 2);
    assert_eq!(c.cutoff_radius(), 5.0);
    assert_eq!(c.transform_radius(), 5.5);
    assert_eq!(c.valence_charge(), 4.0);
    // all stored radial functions are kept, not only the selected ones
    assert_eq!(c.pao().radial.iter().map(Vec::len).sum::<usize>(), 4 * 15);
    assert_eq!(c.scalar_projectors().len(), 12);
    assert_eq!(c.orbitals().len(), 13);
    // basis_radial(2) is (l=1, zeta=0)
    assert_eq!(c.basis_radial(2).values(), c.pao().radial[1][0].values());
}

#[test]
fn mismatched_species_is_an_error() {
    let basis = BasisSpec::parse("C5.0-s1", false).unwrap();
    let pao = PaoFile::read(data_dir().join("PAO/C5.0.pao")).unwrap();
    let vps = VpsFile::read(data_dir().join("VPS/O_PBE19.vps")).unwrap();
    let e = SpeciesData::new(basis, pao, vps).unwrap_err();
    assert!(matches!(e, AtomDataError::AtomSpeciesMismatch { .. }));
}

#[test]
fn io_error_names_the_path() {
    let e = SpeciesData::from_files("X9.9-s1", false, &data_dir().join("PAO"), &data_dir())
        .unwrap_err();
    assert!(e.to_string().contains("X9.9.pao"));
}

#[test]
fn global_l_max_is_largest_plus_six() {
    let h = species("H5.0-s2", "H_PBE19.vps");
    let o = species("O6.0-s2p2d1f1", "O_PBE19.vps");
    assert_eq!(
        GlobalTables::new(std::slice::from_ref(&h)).neutral_atom_l_max(),
        6
    );
    let g = GlobalTables::new(&[h, o]);
    assert_eq!(g.l_max_basis(), 3);
    assert_eq!(g.neutral_atom_l_max(), 9);
    assert_eq!(g.bessel_params().num_grid_k, 900);
    assert_eq!(GlobalTables::new(&[]).neutral_atom_l_max(), 6);
}
