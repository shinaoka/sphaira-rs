use std::f64::consts::PI;

use crate::{
    BesselParams, InnerBehavior, LogGrid, RadialTable, radial_to_wavenumber, spherical_bessel_j,
};

/// Table of `r^l exp(-a r^2)` on a log grid.
fn gaussian(l: usize, a: f64) -> RadialTable {
    let g = LogGrid::new(-7.5, 0.0186, 500).unwrap();
    let v = (0..g.len())
        .map(|i| g.r(i).powi(l as i32) * (-a * g.r(i).powi(2)).exp())
        .collect();
    RadialTable::new(g, v, InnerBehavior::pao(l)).unwrap()
}

/// int r^{l+2} j_l(kr) e^{-a r^2} dr = sqrt(pi) k^l / (2^{l+2} a^{l+3/2}) e^{-k^2/(4a)}.
fn analytic(l: usize, a: f64, k: f64) -> f64 {
    PI.sqrt() * k.powi(l as i32) / (2f64.powi(l as i32 + 2) * a.powf(l as f64 + 1.5))
        * (-k * k / (4.0 * a)).exp()
}

fn params() -> BesselParams {
    BesselParams {
        energy_cutoff_ry: 400.0,
        num_grid_k: 200,
        num_grid_r: 400,
    }
}

#[test]
fn bessel_functions_match_closed_forms() {
    for &x in &[0.5_f64, 0.7, 1.5, 4.0, 9.0, 25.0] {
        let (s, c) = x.sin_cos();
        let j0 = s / x;
        let j1 = s / (x * x) - c / x;
        let j2 = (3.0 / (x * x) - 1.0) * s / x - 3.0 * c / (x * x);
        let j3 = (15.0 / x.powi(3) - 6.0 / x) * s / x - (15.0 / (x * x) - 1.0) * c / x;
        for (l, exact) in [j0, j1, j2, j3].into_iter().enumerate() {
            let got = spherical_bessel_j(l, x);
            // closed forms lose accuracy at small x; compare with the series there
            let tol = if x < 0.5 {
                1e-9 * exact.abs().max(1e-30) + 1e-12 * x.powi(l as i32)
            } else {
                1e-13
            };
            assert!((got - exact).abs() <= tol, "l={l} x={x}: {got} vs {exact}");
        }
    }
    // small x: x^l/(2l+1)!! (1 - x^2/(2(2l+3)))
    for l in 0..4usize {
        let df: f64 = (1..=l).map(|k| (2 * k + 1) as f64).product();
        for &x in &[1e-6_f64, 0.01, 0.3] {
            let approx = x.powi(l as i32) / df * (1.0 - x * x / (2.0 * (2 * l + 3) as f64));
            let tol = (2.0 * x.powi(l as i32 + 4) + 1e-15 * x.powi(l as i32)) / df;
            assert!(
                (spherical_bessel_j(l, x) - approx).abs() <= tol,
                "l={l} x={x}"
            );
        }
    }
    assert!((spherical_bessel_j(3, 1e-3) / (1e-9 / 105.0) - 1.0).abs() < 1e-6);
}

#[test]
fn forward_transform_of_gaussians() {
    let p = params();
    for (l, a) in [(0, 1.0), (1, 1.3), (2, 0.8), (3, 2.0)] {
        let t = gaussian(l, a);
        let w = radial_to_wavenumber(&t, l, 9.0, &p);
        let mut worst: f64 = 0.0;
        for j in (0..p.num_grid_k).step_by(7) {
            let k = j as f64 * p.k_step();
            let err = (w.values()[j] - analytic(l, a, k)).abs();
            worst = worst.max(err);
        }
        assert!(worst < 5e-8, "l={l}: max abs error {worst:e}");
    }
}

#[test]
fn interpolation_in_k() {
    let p = params();
    let t = gaussian(0, 1.0);
    let w = radial_to_wavenumber(&t, 0, 9.0, &p);
    for &k in &[0.0, 0.123, 1.7, 3.3333, 9.9] {
        let err = (w.eval(k) - analytic(0, 1.0, k)).abs();
        assert!(err < 2e-6, "k={k}: {err:e}");
    }
    assert_eq!(w.eval(p.k_max() * 1.5), 0.0);
    assert!(w.eval(-1.0).is_nan());
}

#[test]
fn back_transform_recovers_gaussians() {
    // Back transform amplifies the noise of the forward table by k^2, so use a modest k_max.
    let p = BesselParams {
        energy_cutoff_ry: 100.0,
        num_grid_k: 100,
        num_grid_r: 400,
    };
    let radii = [0.0, 0.2, 0.9, 1.7, 2.5];
    for (l, a, tol) in [(0usize, 1.0, 1e-5), (1, 1.3, 2e-5), (2, 0.8, 2e-5)] {
        let t = gaussian(l, a);
        let w = radial_to_wavenumber(&t, l, 9.0, &p);
        let back = w.to_real_space(&radii);
        for (r, b) in radii.iter().zip(back) {
            let exact = r.powi(l as i32) * (-a * r * r).exp();
            assert!(
                (b - exact).abs() < tol,
                "l={l} r={r}: {b} vs {exact} err {:e}",
                (b - exact).abs()
            );
        }
    }
}

#[test]
fn default_parameters_follow_the_book() {
    let p = BesselParams::default();
    assert_eq!(
        (p.energy_cutoff_ry, p.num_grid_k, p.num_grid_r),
        (3600.0, 900, 900)
    );
    assert_eq!(p.k_max(), 60.0);
}
