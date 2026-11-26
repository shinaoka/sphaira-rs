use std::path::PathBuf;

use crate::{AtomDataError, BasisSpec, PaoFile};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/DFT_DATA19")
}

fn read(name: &str) -> PaoFile {
    PaoFile::read(data_dir().join("PAO").join(name)).unwrap()
}

#[test]
fn every_distributed_pao_file_parses() {
    let mut n = 0;
    for entry in std::fs::read_dir(data_dir().join("PAO")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "pao") {
            let f = PaoFile::read(&path).unwrap();
            assert_eq!(f.radial.len(), f.lmax + 1);
            assert!(f.radial.iter().all(|v| v.len() == f.mul));
            assert!(f.valence_density.is_some());
            n += 1;
        }
    }
    assert_eq!(n, 3);
}

#[test]
fn header_items() {
    for (name, z, rc) in [
        ("H5.0.pao", 1.0, 5.0),
        ("C5.0.pao", 6.0, 5.0),
        ("O6.0.pao", 8.0, 6.0),
    ] {
        let f = read(name);
        assert_eq!(f.atom_species, z, "{name}");
        assert_eq!((f.lmax, f.mul, f.grid_num), (3, 15, 500), "{name}");
        assert_eq!(f.cutoff_radius, rc, "{name}");
        let g = f.radial[0][0].grid();
        assert_eq!(g.len(), 500);
        assert!(g.r_min() > 4e-4 && g.r_min() < 6e-4);
        let outside = g.r_last() - rc;
        assert!(outside > 0.5 && outside < 1.3, "{name}: {outside}");
        assert_eq!(
            f.contraction_blocks.len(),
            f.number_optpao.unwrap(),
            "{name}"
        );
        assert!(f.contraction_blocks.iter().all(|b| !b.rows.is_empty()));
    }
}

#[test]
fn contraction_rows_are_kept() {
    let f = read("C5.0.pao");
    assert_eq!(f.contraction_blocks[0].name, "contraction.coefficients1");
    assert_eq!(f.contraction_blocks[0].rows.len(), 75);
    let r = f.contraction_blocks[1].rows[0];
    assert_eq!((r.atom, r.l, r.mul, r.p), (1, 0, 0, 0));
    assert!((r.coefficient - 0.994624417313332).abs() < 1e-15);
    assert_eq!(f.contraction_blocks[1].rows.len(), 30);
}

#[test]
fn grid_is_log_uniform_and_values_vanish_beyond_cutoff() {
    let f = read("C5.0.pao");
    let g = *f.radial[1][0].grid();
    for i in [0, 1, 250, 499] {
        let x = g.x_min() + g.dx() * i as f64;
        assert!((g.r(i) - x.exp()).abs() <= 1e-11 * x.exp());
    }
    for l in 0..=f.lmax {
        for t in &f.radial[l] {
            for (i, &v) in t.values().iter().enumerate() {
                if g.r(i) > f.cutoff_radius {
                    assert_eq!(v, 0.0);
                }
            }
        }
    }
}

#[test]
fn out_of_grid_values() {
    let f = read("C5.0.pao");
    let s = &f.radial[0][0];
    let p = &f.radial[1][0];
    let r_min = s.grid().r_min();
    assert_eq!(s.eval(0.1 * r_min), s.values()[0]);
    assert_eq!(p.eval(0.1 * r_min), 0.0);
    assert_eq!(s.eval(s.grid().r_last() * 1.01), 0.0);
    assert_eq!(s.eval(f.cutoff_radius), 0.0);
    // interpolation reproduces the grid values
    let r10 = s.grid().r(10);
    assert!((s.eval(r10) - s.values()[10]).abs() < 1e-12 * s.values()[10].abs().max(1.0));
}

#[test]
fn leading_radial_functions_are_orthonormal() {
    for name in ["H5.0.pao", "C5.0.pao", "O6.0.pao"] {
        let f = read(name);
        for (l, funcs) in f.radial.iter().enumerate().take(f.lmax + 1) {
            for a in 0..4 {
                for b in 0..=a {
                    let ov = funcs[a].trapezoid_r2_product(&funcs[b]).unwrap();
                    let expect = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (ov - expect).abs() < 1e-3,
                        "{name} l={l} ({a},{b}): overlap {ov}"
                    );
                }
            }
        }
    }
}

#[test]
fn valence_density_is_on_the_same_grid() {
    let f = read("C5.0.pao");
    let d = f.valence_density.as_ref().unwrap();
    assert!(d.grid().same_as(f.radial[0][0].grid()));
    // the book notes that the raw integral of C5.0/C6.0 is not Z (about 4.5 for Z = 4)
    let q = d.trapezoid_4pi_r2();
    assert!(q > 3.5 && q < 5.5, "{q}");
}

#[test]
fn select_basis_checks_file_limits() {
    let f = read("C5.0.pao");
    let ok = BasisSpec::parse("C5.0-s2p2d1", false).unwrap();
    assert_eq!(f.select_basis(&ok).unwrap().len(), 5);
    let too_high = BasisSpec::parse("C5.0-g1", false).unwrap();
    assert!(matches!(
        f.select_basis(&too_high),
        Err(AtomDataError::BasisExceedsFile { .. })
    ));
    let too_many = BasisSpec::parse("C5.0-s16", false).unwrap();
    assert!(matches!(
        f.select_basis(&too_many),
        Err(AtomDataError::BasisExceedsFile { .. })
    ));
    assert!(f.radial_function(0, 14).is_some());
    assert!(f.radial_function(0, 15).is_none());
}

fn synthetic(extra: &str) -> String {
    let mut s = String::from("AtomSpecies 1\nPAO.Lmax 0\nPAO.Mul 1\ngrid.num.output 4\n");
    s.push_str(extra);
    s
}

fn block(name: &str, rows: &[[f64; 3]]) -> String {
    let mut s = format!("<{name}\n");
    for r in rows {
        s.push_str(&format!("{} {} {}\n", r[0], r[0].exp(), r[2]));
    }
    s.push_str(&format!("{name}>\n"));
    s
}

#[test]
fn synthetic_defaults_and_errors() {
    let rows = [
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.9],
        [0.2, 0.0, 0.8],
        [0.3, 0.0, 0.7],
    ];
    let ok = synthetic(&block("pseudo.atomic.orbitals.L=0", &rows));
    let f = PaoFile::parse(&ok).unwrap();
    assert_eq!(f.cutoff_radius, 5.0); // default
    assert!(f.valence_density.is_none());
    assert!(f.contraction_blocks.is_empty());

    assert!(matches!(
        PaoFile::parse("PAO.Lmax 0\n"),
        Err(AtomDataError::MissingKeyword { .. })
    ));
    assert!(matches!(
        PaoFile::parse(&synthetic("")),
        Err(AtomDataError::MissingBlock { .. })
    ));
    let short = synthetic(&block("pseudo.atomic.orbitals.L=0", &rows[..3]));
    assert!(matches!(
        PaoFile::parse(&short),
        Err(AtomDataError::RowCount {
            expected: 4,
            found: 3,
            ..
        })
    ));
    let bad_grid = synthetic(
        "<pseudo.atomic.orbitals.L=0\n0 1 1\n0.1 1.1 1\n0.25 1.3 1\n0.3 1.35 1\npseudo.atomic.orbitals.L=0>\n",
    );
    assert!(matches!(
        PaoFile::parse(&bad_grid),
        Err(AtomDataError::Grid { .. })
    ));
    let msg = PaoFile::parse("PAO.Lmax 0\n").unwrap_err().to_string();
    assert!(msg.contains("AtomSpecies"));
}
