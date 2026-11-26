use crate::{BasisSpec, BasisSpecError, OrbitalLabel, RadialIndex};

#[test]
fn c6_s2p2d1() {
    let b = BasisSpec::parse("C6.0-s2p2d1", false).unwrap();
    assert_eq!(b.species_file(), "C6.0");
    assert_eq!(b.pao_file_name(), "C6.0.pao");
    assert_eq!(
        [b.count(0), b.count(1), b.count(2), b.count(3)],
        [2, 2, 1, 0]
    );
    assert_eq!(b.num_basis_functions(), 13);
    assert_eq!(b.l_max(), 2);
    let orbs = b.orbitals();
    assert_eq!(orbs.len(), 13);
    let o = |l, zeta, component| OrbitalLabel { l, zeta, component };
    assert_eq!(orbs[0], o(0, 0, 0));
    assert_eq!(orbs[1], o(0, 1, 0));
    assert_eq!(orbs[2], o(1, 0, 0));
    assert_eq!(orbs[4], o(1, 0, 2));
    assert_eq!(orbs[5], o(1, 1, 0));
    assert_eq!(orbs[8], o(2, 0, 0));
    assert_eq!(orbs[12], o(2, 0, 4));
}

#[test]
fn missing_intermediate_shell() {
    let b = BasisSpec::parse("H5.0-s2d1", false).unwrap();
    assert_eq!(b.count(1), 0);
    assert_eq!(b.l_max(), 2);
    assert_eq!(b.num_basis_functions(), 7);
    assert_eq!(
        b.radial_indices(),
        vec![
            RadialIndex { l: 0, zeta: 0 },
            RadialIndex { l: 0, zeta: 1 },
            RadialIndex { l: 2, zeta: 0 }
        ]
    );
}

#[test]
fn contraction_rules() {
    let err = BasisSpec::parse("C5.0-s2>1", false).unwrap_err();
    assert!(matches!(
        err,
        BasisSpecError::ContractionWithoutOptimization { .. }
    ));
    let same = BasisSpec::parse("C5.0-s1>1p2", false).unwrap();
    assert_eq!(
        same,
        BasisSpec::parse("C5.0-s1p2", false)
            .unwrap()
            .with_spec_of(&same)
    );
    let c = BasisSpec::parse("C5.0-s3>2p1", true).unwrap();
    assert_eq!((c.count(0), c.primitive_count(0)), (2, 3));
    assert!(c.is_contracted());
    assert!(matches!(
        BasisSpec::parse("C5.0-s2>3", true).unwrap_err(),
        BasisSpecError::InvalidContraction { .. }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s2>0", true).unwrap_err(),
        BasisSpecError::InvalidContraction { .. }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s2>", true).unwrap_err(),
        BasisSpecError::MissingCount { .. }
    ));
}

#[test]
fn syntax_errors() {
    use BasisSpecError as E;
    let p = |s| BasisSpec::parse(s, true).unwrap_err();
    assert!(matches!(p("C5.0"), E::MissingSeparator { .. }));
    assert!(matches!(p("-s1"), E::BadName { .. }));
    assert!(matches!(p("C5.0-"), E::EmptyShells { .. }));
    assert!(matches!(
        p("C5.0-s1h1"),
        E::UnexpectedCharacter { ch: 'h', .. }
    ));
    assert!(matches!(
        p("C5.0-S1"),
        E::UnexpectedCharacter { ch: 'S', .. }
    ));
    assert!(matches!(p("C5.0-sp1"), E::MissingCount { letter: 's', .. }));
    assert!(matches!(
        p("C5.0-s1s2"),
        E::DuplicateShell { letter: 's', .. }
    ));
    assert!(matches!(p("C5.0-s100"), E::CountTooLarge { .. }));
    assert!(matches!(p("C5.0-s0p0"), E::NoOrbitals { .. }));
    assert!(matches!(
        p("C5.0-s1 p1"),
        E::UnexpectedCharacter { ch: ' ', .. }
    ));
    // error messages say what to fix
    assert!(p("C5.0").to_string().contains("write it as"));
}

#[test]
fn name_with_dash_and_order() {
    let b = BasisSpec::parse("  Fe-x6.0S-d1s2g1 ", false).unwrap();
    assert_eq!(b.species_file(), "Fe-x6.0S");
    assert_eq!(b.l_max(), 4);
    assert_eq!(b.num_basis_functions(), 2 + 5 + 9);
}

impl BasisSpec {
    fn with_spec_of(mut self, other: &Self) -> Self {
        self.spec = other.spec.clone();
        self
    }
}
