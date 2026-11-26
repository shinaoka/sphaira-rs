use std::path::PathBuf;

use crate::{
    AtomDataError, InnerBehavior, JAveraging, JMinusPart, LogGrid, RadialTable, VpsFile, VpsFormat,
    VpsProjector, average_j_dependent, j_weights,
};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/DFT_DATA19")
}

fn read(name: &str) -> VpsFile {
    VpsFile::read(data_dir().join("VPS").join(name)).unwrap()
}

#[test]
fn every_distributed_vps_file_parses() {
    let mut n = 0;
    for entry in std::fs::read_dir(data_dir().join("VPS")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "vps") {
            let f = VpsFile::read(&path).unwrap();
            assert_eq!(f.format, VpsFormat::New);
            assert!(f.j_dependent);
            assert!(f.projectors.iter().all(|p| p.j_minus.is_some()));
            n += 1;
        }
    }
    assert_eq!(n, 3);
}

#[test]
fn carbon_fields() {
    let f = read("C_PBE19.vps");
    assert_eq!(f.atom_species, 6.0);
    assert_eq!((f.total_electron, f.valence_electron), (6.0, 4.0));
    assert_eq!(f.valence_charge(), 4.0);
    assert_eq!(f.grid_num, 500);
    assert_eq!(f.projectors.len(), 12);
    let ls: Vec<usize> = f.projectors.iter().map(|p| p.l).collect();
    assert_eq!(ls, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
    let e: Vec<f64> = f.projectors.iter().map(|p| p.energy).collect();
    assert_eq!(e, [1., 1., -1., 1., -1., -1., 1., -1., -1., -1., -1., 1.]);
    let g = f.local_potential.grid();
    assert!((g.x_min() + 7.5).abs() < 1e-12);
    assert!((g.x(499) - 2.799).abs() < 1e-12);
    // r V_loc -> -Z_v at the last grid point
    let r = g.r(499);
    assert!((r * f.local_potential.values()[499] + 4.0).abs() < 1e-6);
    // first row of the table (checked against the file)
    assert!((f.local_potential.values()[0] + 5.71194539855942).abs() < 1e-13);
    assert!((f.projectors[0].radial.values()[0] - 6.10861347964058).abs() < 1e-13);
    let minus = f.projectors[2].j_minus.as_ref().unwrap();
    assert_eq!(minus.energy, -1.0);
    // PCC
    let pcc = f.pcc_density.as_ref().unwrap();
    assert!((pcc.values()[0] - 0.219922643717031).abs() < 1e-15);
    let q = pcc.trapezoid_4pi_r2();
    assert!(q > 0.2 && q < 0.3, "partial core charge {q}");
}

#[test]
fn hydrogen_and_oxygen_fields() {
    let h = read("H_PBE19.vps");
    assert_eq!((h.atom_species, h.valence_charge()), (1.0, 1.0));
    assert_eq!(h.projectors.len(), 6);
    assert!(h.pcc_density.is_none()); // charge.pcc.calc OFF
    let o = read("O_PBE19.vps");
    assert_eq!((o.atom_species, o.valence_charge()), (8.0, 6.0));
    assert_eq!(o.projectors.len(), 7);
    assert!(o.pcc_density.is_some());
    let l1: Vec<f64> = o
        .projectors
        .iter()
        .filter(|p| p.l == 1)
        .map(|p| p.energy)
        .collect();
    assert_eq!(l1, [-1.0, 1.0]);
}

#[test]
fn projector_inner_rules() {
    let c = read("C_PBE19.vps");
    assert_eq!(
        c.projectors[0].radial.inner(),
        InnerBehavior::LinearFromFirstInterval
    );
    assert_eq!(c.projectors[4].radial.inner(), InnerBehavior::Zero);
    let r_min = c.local_potential.grid().r_min();
    assert_eq!(c.projectors[4].radial.eval(0.5 * r_min), 0.0);
    assert_eq!(c.projectors[0].radial.eval(1e3), 0.0);
}

fn table(c: f64) -> RadialTable {
    RadialTable::new(
        LogGrid::new(-3.0, 0.1, 20).unwrap(),
        vec![c; 20],
        InnerBehavior::Zero,
    )
    .unwrap()
}

fn proj(l: usize, ep: f64, bp: f64, em: f64, bm: f64) -> VpsProjector {
    VpsProjector {
        l,
        energy: ep,
        radial: table(bp),
        j_minus: Some(JMinusPart {
            energy: em,
            radial: table(bm),
        }),
    }
}

#[test]
fn j_weights_sum_to_one() {
    for l in 0..6 {
        let (p, m) = j_weights(l);
        assert!((p + m - 1.0).abs() < 1e-15);
    }
    assert_eq!(j_weights(0), (1.0, 0.0));
}

#[test]
fn j_averaging_known_answers() {
    // l = 2: w+ = 3/5, w- = 2/5
    let ps = [
        proj(2, 1.0, 10.0, 1.0, 20.0),
        proj(1, 1.0, 3.0, -1.0, 6.0), // opposite signs: eps -> 1/(2l+1) = 1/3
        proj(0, 0.5, 7.0, -0.5, 100.0), // l = 0: w- = 0, equals the + part
    ];
    let s = average_j_dependent(&ps, JAveraging::Projector).unwrap();
    assert_eq!(s.len(), 3);
    assert!((s[0].energy - 1.0).abs() < 1e-15);
    assert!((s[0].radial.values()[5] - (0.6 * 10.0 + 0.4 * 20.0)).abs() < 1e-13);
    assert!((s[1].energy - 1.0 / 3.0).abs() < 1e-15);
    assert!((s[1].radial.values()[0] - 4.0).abs() < 1e-14);
    assert_eq!(s[2].energy, 0.5);
    assert_eq!(s[2].radial.values()[0], 7.0);

    let o = average_j_dependent(&ps[..2], JAveraging::Operator).unwrap();
    assert_eq!(o.len(), 4);
    assert!((o[0].energy - 0.6).abs() < 1e-15 && (o[1].energy - 0.4).abs() < 1e-15);
    assert_eq!(o[1].radial.values()[0], 20.0);
    assert!((o[2].energy - 2.0 / 3.0).abs() < 1e-15 && (o[3].energy + 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn scalar_projectors_pass_through_without_j_minus() {
    let p = VpsProjector {
        l: 1,
        energy: 0.3,
        radial: table(2.0),
        j_minus: None,
    };
    let s = average_j_dependent(&[p], JAveraging::Projector).unwrap();
    assert_eq!(
        (s[0].l, s[0].energy, s[0].radial.values()[3]),
        (1, 0.3, 2.0)
    );
}

#[test]
fn j_averaging_on_real_file() {
    let c = read("C_PBE19.vps");
    let s = c.scalar_projectors(JAveraging::Projector).unwrap();
    assert_eq!(s.len(), 12);
    // l = 1 projector #0 (index 4): eps+ = eps- = -1 -> -1; b = 2/3 b+ + 1/3 b-
    assert!((s[4].energy + 1.0).abs() < 1e-15);
    let p = &c.projectors[4];
    let m = p.j_minus.as_ref().unwrap();
    let i = 100;
    let expect = 2.0 / 3.0 * p.radial.values()[i] + 1.0 / 3.0 * m.radial.values()[i];
    assert!((s[4].radial.values()[i] - expect).abs() < 1e-14 * expect.abs().max(1.0));
}

fn synthetic_new(proj_energies: &str, ncols: usize, j: &str, pcc: &str) -> String {
    let mut s = format!(
        "AtomSpecies 3\ntotal.electron 3.0\nvalence.electron 1.0\ngrid.num.output 4\nj.dependent.pseudo.potentials {j}\ncharge.pcc.calc {pcc}\n<project.energies\n{proj_energies}\nproject.energies>\n<Pseudo.Potentials\n"
    );
    for i in 0..4 {
        let x = 0.1 * i as f64;
        s.push_str(&format!("{x} {}", x.exp()));
        for c in 0..ncols - 2 {
            s.push_str(&format!(" {}", (c + 1) as f64 * 10.0 + i as f64));
        }
        s.push('\n');
    }
    s.push_str("Pseudo.Potentials>\n");
    if pcc == "on" {
        s.push_str("<density.PCC\n");
        for i in 0..4 {
            let x = 0.1 * i as f64;
            s.push_str(&format!("{x} {} {}\n", x.exp(), 0.5 + i as f64));
        }
        s.push_str("density.PCC>\n");
    }
    s
}

#[test]
fn synthetic_new_format_and_zero_projectors() {
    // non-j-dependent, two projectors with non-unit energies
    let f = VpsFile::parse(&synthetic_new("2\n0 0.5\n1 -2.5", 5, "off", "off")).unwrap();
    assert_eq!(f.projectors.len(), 2);
    assert_eq!(f.projectors[1].energy, -2.5);
    assert!(f.projectors[0].j_minus.is_none());
    assert_eq!(f.projectors[1].radial.values()[0], 30.0);
    assert_eq!(f.valence_charge(), 1.0);
    // E.vps-like: no projectors, only V_loc column
    let e = VpsFile::parse(&synthetic_new("0", 3, "off", "off")).unwrap();
    assert!(e.projectors.is_empty());
    // j-dependent with PCC
    let j = VpsFile::parse(&synthetic_new("1\n1 1.0 -1.0", 5, "on", "on")).unwrap();
    let m = j.projectors[0].j_minus.as_ref().unwrap();
    assert_eq!((j.projectors[0].energy, m.energy), (1.0, -1.0));
    assert_eq!(j.projectors[0].radial.values()[0], 20.0);
    assert_eq!(m.radial.values()[0], 30.0);
    assert_eq!(j.pcc_density.unwrap().values()[3], 3.5);
}

#[test]
fn synthetic_defaults_and_errors() {
    // defaults: total.electron = AtomSpecies, valence.electron = 0
    let txt = "AtomSpecies 3\ngrid.num.output 4\n<project.energies\n0\nproject.energies>\n<Pseudo.Potentials\n0 1 5\n0.1 1.1051709180756477 5\n0.2 1.2214027581601699 5\n0.3 1.3498588075760032 5\nPseudo.Potentials>\n";
    let f = VpsFile::parse(txt).unwrap();
    assert_eq!(
        (f.total_electron, f.valence_electron, f.valence_charge()),
        (3.0, 0.0, 0.0)
    );
    // wrong column count
    let e = VpsFile::parse(&synthetic_new("1\n0 1.0", 5, "off", "off")).unwrap_err();
    assert!(matches!(
        e,
        AtomDataError::Columns {
            expected: 4,
            found: 5,
            ..
        }
    ));
    // wrong projector row width for j-dependent
    let e = VpsFile::parse(&synthetic_new("1\n0 1.0", 5, "on", "off")).unwrap_err();
    assert!(matches!(e, AtomDataError::Columns { .. }));
    // announced count mismatch
    let e = VpsFile::parse(&synthetic_new("2\n0 1.0", 4, "off", "off")).unwrap_err();
    assert!(matches!(e, AtomDataError::RowCount { .. }));
    // PCC on but block missing
    let no_pcc =
        synthetic_new("0", 3, "off", "off").replace("charge.pcc.calc off", "charge.pcc.calc on");
    assert!(matches!(
        VpsFile::parse(&no_pcc),
        Err(AtomDataError::MissingBlock { .. })
    ));
    // PCC grid mismatch
    let full = synthetic_new("0", 3, "off", "on");
    let (head, tail) = full.split_once("<density.PCC\n").unwrap();
    let bad = format!(
        "{head}<density.PCC\n{}",
        tail.replacen("0 1 0.5", "0.01 1 0.5", 1)
    );
    assert!(matches!(
        VpsFile::parse(&bad),
        Err(AtomDataError::GridMismatch { .. })
    ));
}

fn synthetic_old(local: &str, extra: &str) -> String {
    format!(
        "AtomSpecies 6\ntotal.electron 6.0\nvalence.electron 4.0\ngrid.num.output 4\nnumber.vps 3\n<pseudo.NandL\n 0 2 0 1.30 0.0\n 1 2 1 1.30 0.0\n 2 3 2 1.30 0.0\npseudo.NandL>\n{local}{extra}<projection.energies\n 0 0.7\n 1 -0.3\nprojection.energies>\n<Pseudo.Potentials\n0 1 5 11 12\n0.1 1.1051709180756477 5 11 12\n0.2 1.2214027581601699 5 11 12\n0.3 1.3498588075760032 5 11 12\nPseudo.Potentials>\n"
    )
}

#[test]
fn old_format() {
    // local part is line 1 (the p line): projectors are l = 0 and l = 2
    let f = VpsFile::parse(&synthetic_old("local.part.vps 1\n", "")).unwrap();
    assert_eq!(f.format, VpsFormat::Old);
    let ls: Vec<usize> = f.projectors.iter().map(|p| p.l).collect();
    assert_eq!(ls, [0, 2]);
    assert_eq!(
        (f.projectors[0].energy, f.projectors[1].energy),
        (0.7, -0.3)
    );
    assert_eq!(f.projectors[1].radial.values()[0], 12.0);
    // out-of-range local index: no line is excluded -> 3 projectors but only 2 energies
    let e = VpsFile::parse(&synthetic_old("local.part.vps 9\n", "")).unwrap_err();
    assert!(matches!(e, AtomDataError::RowCount { .. }));
    // j-dependent old format is an error
    let e = VpsFile::parse(&synthetic_old(
        "local.part.vps 1\n",
        "j.dependent.pseudo.potentials on\n",
    ))
    .unwrap_err();
    assert!(matches!(e, AtomDataError::OldFormatJDependent));
}
