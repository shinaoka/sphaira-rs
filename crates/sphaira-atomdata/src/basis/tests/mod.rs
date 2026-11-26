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
fn contraction_with_optimization() {
    let c = BasisSpec::parse("C5.0-s3>2p1>1", true).unwrap();
    assert_eq!((c.count(0), c.primitive_count(0)), (2, 3));
    assert_eq!((c.count(1), c.primitive_count(1)), (1, 1));
    assert!(c.is_contracted());
    assert_eq!(c.num_basis_functions(), 2 + 3);
    // b == a is accepted, `s0` (zero entry) needs no `>`
    let same = BasisSpec::parse("C5.0-s2>2p0d1>1", true).unwrap();
    assert!(!same.is_contracted());
    assert_eq!((same.count(1), same.l_max()), (0, 2));
    // b <= a is the only condition on b: a contraction to zero is accepted, and l_max follows
    // the original count
    let z = BasisSpec::parse("C5.0-s1>1d2>0", true).unwrap();
    assert_eq!((z.count(2), z.primitive_count(2), z.l_max()), (0, 2, 2));
    assert_eq!(z.num_basis_functions(), 1);
}

#[test]
fn contraction_errors() {
    use BasisSpecError as E;
    // without optimization `>` is always an error, even for a == b
    for s in ["C5.0-s2>1", "C5.0-s1>1p2", "C5.0-s>2", "C5.0-s2>"] {
        assert!(
            matches!(
                BasisSpec::parse(s, false).unwrap_err(),
                E::ContractionWithoutOptimization { letter: 's', .. }
            ),
            "{s}"
        );
    }
    // with optimization every nonzero entry must be `a>b`
    assert!(matches!(
        BasisSpec::parse("C5.0-s3", true).unwrap_err(),
        E::ExpectedContraction {
            letter: 's',
            count: 3,
            ..
        }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s1>1p2", true).unwrap_err(),
        E::ExpectedContraction {
            letter: 'p',
            count: 2,
            ..
        }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s2>3", true).unwrap_err(),
        E::InvalidContraction {
            original: 2,
            contracted: 3,
            ..
        }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s2>", true).unwrap_err(),
        E::MissingCount { .. }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s>2", true).unwrap_err(),
        E::MissingCount { .. }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-s2>0", true).unwrap_err(),
        E::NoOrbitals { .. }
    ));
}

#[test]
fn syntax_errors() {
    use BasisSpecError as E;
    let p = |s| BasisSpec::parse(s, false).unwrap_err();
    assert!(matches!(p("C5.0"), E::MissingSeparator { .. }));
    assert!(matches!(p("-s1"), E::BadName { .. }));
    assert!(matches!(p("C5.0-"), E::EmptyShells { .. }));
    assert!(matches!(
        p("C5.0-s1h1"),
        E::UnexpectedCharacter { ch: 'h', .. }
    ));
    // only lowercase letters
    assert!(matches!(
        p("C5.0-S1"),
        E::UnexpectedCharacter { ch: 'S', .. }
    ));
    assert!(matches!(
        BasisSpec::parse("C5.0-P1", true).unwrap_err(),
        E::UnexpectedCharacter { ch: 'P', .. }
    ));
    assert!(matches!(
        p("C5.0-s1 p1"),
        E::UnexpectedCharacter { ch: ' ', .. }
    ));
    assert!(matches!(p("C5.0-s0p0"), E::NoOrbitals { .. }));
    assert!(matches!(p("C5.0-sp"), E::NoOrbitals { .. }));
    // error messages say what to fix
    assert!(p("C5.0").to_string().contains("write it as"));
}

#[test]
fn digit_limit_without_optimization() {
    let ok = BasisSpec::parse("C5.0-s999", false).unwrap();
    assert_eq!(ok.count(0), 999);
    for s in ["C5.0-s1000", "C5.0-s0001", "C5.0-p1s0000"] {
        assert!(
            matches!(
                BasisSpec::parse(s, false).unwrap_err(),
                BasisSpecError::CountTooLarge { .. }
            ),
            "{s}"
        );
    }
    // with optimization the limit does not apply (only overflow is an error)
    assert!(BasisSpec::parse("C5.0-s1000>1000", true).is_ok());
    assert!(matches!(
        BasisSpec::parse("C5.0-s99999999999999999999999>1", true).unwrap_err(),
        BasisSpecError::CountTooLarge { .. }
    ));
}

#[test]
fn name_is_split_at_first_dash() {
    let b = BasisSpec::parse("Fe-s2", false).unwrap();
    assert_eq!(b.species_file(), "Fe");
    // everything after the first `-` is the shell list, so a second `-` is a character error
    assert!(matches!(
        BasisSpec::parse("Fe-x6.0S-d1s2g1", false).unwrap_err(),
        BasisSpecError::UnexpectedCharacter { ch: 'x', .. }
    ));
    assert!(matches!(
        BasisSpec::parse("C6.0-s2-p1", false).unwrap_err(),
        BasisSpecError::UnexpectedCharacter {
            ch: '-',
            position: 7,
            ..
        }
    ));
    assert_eq!(
        BasisSpec::parse("  Fe6.0-d1s2g1 ", false)
            .unwrap()
            .species_file(),
        "Fe6.0"
    );
}

#[test]
fn letters_in_any_order_and_repeated() {
    let b = BasisSpec::parse("C5.0-d1s2g1", false).unwrap();
    assert_eq!(b.l_max(), 4);
    assert_eq!(b.num_basis_functions(), 2 + 5 + 9);
    // a repeated letter with digits replaces the earlier count
    let r = BasisSpec::parse("C5.0-s1p2s3", false).unwrap();
    assert_eq!((r.count(0), r.count(1)), (3, 2));
    // `s0` replaces too
    let z = BasisSpec::parse("C5.0-s2p1s0", false).unwrap();
    assert_eq!((z.count(0), z.count(1)), (0, 1));
    // a letter without digits leaves the count unchanged
    let u = BasisSpec::parse("C5.0-s2ps", false).unwrap();
    assert_eq!((u.count(0), u.count(1)), (2, 0));
    let v = BasisSpec::parse("C5.0-s2p1ps", false).unwrap();
    assert_eq!((v.count(0), v.count(1)), (2, 1));
    // same under optimization: the replacement is by the later `a>b`
    let o = BasisSpec::parse("C5.0-s3>2p1>1s1>1s", true).unwrap();
    assert_eq!((o.count(0), o.primitive_count(0), o.count(1)), (1, 1, 1));
}
