use std::f64::consts::PI;

use crate::{YlmKind, component_order, real_ylm, real_ylm_with_gradient};

/// Gauss-Legendre nodes and weights on [-1, 1] by Newton iteration.
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp = 1.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / dp;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            (x, 2.0 / ((1.0 - x * x) * dp * dp))
        })
        .collect()
}

/// Product quadrature on the sphere: (unit vector, weight), weights sum to 4 pi.
fn sphere_quadrature(nz: usize, nphi: usize) -> Vec<([f64; 3], f64)> {
    let mut q = Vec::new();
    for (z, w) in gauss_legendre(nz) {
        let s = (1.0 - z * z).sqrt();
        for j in 0..nphi {
            let phi = 2.0 * PI * j as f64 / nphi as f64;
            q.push((
                [s * phi.cos(), s * phi.sin(), z],
                w * 2.0 * PI / nphi as f64,
            ));
        }
    }
    q
}

/// Deterministic pseudo-random unit vectors.
fn random_units(n: usize) -> Vec<[f64; 3]> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut v = Vec::new();
    while v.len() < n {
        let p = [next(), next(), next()];
        let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if r2 > 0.1 && r2 < 1.0 {
            let r = r2.sqrt();
            v.push([p[0] / r, p[1] / r, p[2] / r]);
        }
    }
    v
}

#[test]
fn quadrature_integrates_constant() {
    let q = sphere_quadrature(12, 24);
    let total: f64 = q.iter().map(|(_, w)| w).sum();
    assert!((total - 4.0 * PI).abs() < 1e-13);
}

#[test]
fn orthonormality_up_to_l5() {
    let q = sphere_quadrature(12, 24);
    let lmax = 5;
    // flatten all (l, mu)
    let vals: Vec<Vec<f64>> = q
        .iter()
        .map(|(u, _)| (0..=lmax).flat_map(|l| real_ylm(l, *u)).collect())
        .collect();
    let n = (lmax + 1) * (lmax + 1);
    let mut worst: f64 = 0.0;
    for a in 0..n {
        for b in 0..=a {
            let s: f64 = q.iter().zip(&vals).map(|((_, w), v)| w * v[a] * v[b]).sum();
            let expect = if a == b { 1.0 } else { 0.0 };
            worst = worst.max((s - expect).abs());
        }
    }
    assert!(worst < 1e-13, "max deviation {worst:e}");
}

#[test]
fn addition_theorem() {
    let pts = random_units(8);
    for l in 0..=6 {
        for a in &pts {
            for b in &pts {
                let sa = real_ylm(l, *a);
                let sb = real_ylm(l, *b);
                let lhs: f64 = sa.iter().zip(&sb).map(|(x, y)| x * y).sum();
                // P_l(cos gamma) by recurrence
                let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                let (mut p0, mut p1) = (1.0, c);
                let pl = match l {
                    0 => 1.0,
                    _ => {
                        for k in 2..=l {
                            let p2 = ((2 * k - 1) as f64 * c * p1 - (k - 1) as f64 * p0) / k as f64;
                            p0 = p1;
                            p1 = p2;
                        }
                        p1
                    }
                };
                let rhs = (2 * l + 1) as f64 / (4.0 * PI) * pl;
                assert!((lhs - rhs).abs() < 1e-13, "l={l}: {lhs} vs {rhs}");
            }
        }
    }
}

#[test]
fn ordering_table_up_to_l3() {
    // polynomials of the chapter's table, as functions of the unit vector
    type Poly = fn([f64; 3]) -> f64;
    let polys: [Vec<Poly>; 4] = [
        vec![|_| 1.0],
        vec![|u| u[0], |u| u[1], |u| u[2]],
        vec![
            |u| 3.0 * u[2] * u[2] - 1.0,
            |u| u[0] * u[0] - u[1] * u[1],
            |u| u[0] * u[1],
            |u| u[0] * u[2],
            |u| u[1] * u[2],
        ],
        vec![
            |u| u[2] * (5.0 * u[2] * u[2] - 3.0),
            |u| u[0] * (5.0 * u[2] * u[2] - 1.0),
            |u| u[1] * (5.0 * u[2] * u[2] - 1.0),
            |u| u[2] * (u[0] * u[0] - u[1] * u[1]),
            |u| u[0] * u[1] * u[2],
            |u| u[0] * (u[0] * u[0] - 3.0 * u[1] * u[1]),
            |u| u[1] * (3.0 * u[0] * u[0] - u[1] * u[1]),
        ],
    ];
    let pts = random_units(10);
    for (l, ps) in polys.iter().enumerate() {
        let s0 = real_ylm(l, pts[0]);
        for (mu, p) in ps.iter().enumerate() {
            let c = s0[mu] / p(pts[0]);
            assert!(c > 0.0, "l={l} mu={mu}: coefficient must be positive");
            for u in &pts {
                let s = real_ylm(l, *u)[mu];
                assert!((s - c * p(*u)).abs() < 1e-13, "l={l} mu={mu}");
            }
        }
    }
}

#[test]
fn explicit_coefficients_p_and_d() {
    let u = [0.3, -0.5, 0.8124038404635961]; // unit vector
    let sp = real_ylm(1, u);
    let c1 = (3.0 / (4.0 * PI)).sqrt();
    for i in 0..3 {
        assert!((sp[i] - c1 * u[i]).abs() < 1e-15);
    }
    let sd = real_ylm(2, u);
    let c = [
        (5.0 / (16.0 * PI)).sqrt(),
        (15.0 / (16.0 * PI)).sqrt(),
        (15.0 / (4.0 * PI)).sqrt(),
        (15.0 / (4.0 * PI)).sqrt(),
        (15.0 / (4.0 * PI)).sqrt(),
    ];
    let poly = [
        3.0 * u[2] * u[2] - 1.0,
        u[0] * u[0] - u[1] * u[1],
        u[0] * u[1],
        u[0] * u[2],
        u[1] * u[2],
    ];
    for i in 0..5 {
        assert!((sd[i] - c[i] * poly[i]).abs() < 1e-14, "d component {i}");
    }
    assert!((real_ylm(0, u)[0] - 1.0 / (4.0 * PI).sqrt()).abs() < 1e-16);
}

#[test]
fn l4_follows_general_rule() {
    let order = component_order(4);
    assert_eq!(order.len(), 9);
    assert_eq!((order[0].m, order[0].kind), (0, YlmKind::Zero));
    assert_eq!((order[1].m, order[1].kind), (1, YlmKind::Cos));
    assert_eq!((order[8].m, order[8].kind), (4, YlmKind::Sin));
    // cos 4 phi component is positive-proportional to x^4 - 6 x^2 y^2 + y^4 at z = 0
    let u = [0.6, 0.8, 0.0];
    let s = real_ylm(4, u)[7];
    let p = u[0].powi(4) - 6.0 * u[0] * u[0] * u[1] * u[1] + u[1].powi(4);
    let n = (9.0 / (2.0 * PI) / 40320.0).sqrt() * 105.0;
    assert!((s - n * p).abs() < 1e-13);
    assert!(s / p > 0.0);
}

#[test]
fn gradient_matches_finite_differences() {
    let mut pts = random_units(4);
    pts.push([1e-9, 0.0, 1.0]); // near the pole
    pts.push([0.0, 0.0, 1.0]);
    pts.push([0.0, 0.0, -1.0]);
    for l in 0..=5 {
        for u in &pts {
            let r = [1.3 * u[0], 1.3 * u[1], 1.3 * u[2]];
            let (_, g) = real_ylm_with_gradient(l, r);
            let h = 1e-6;
            for axis in 0..3 {
                let mut rp = r;
                let mut rm = r;
                rp[axis] += h;
                rm[axis] -= h;
                let (sp, sm) = (real_ylm(l, rp), real_ylm(l, rm));
                for mu in 0..2 * l + 1 {
                    let fd = (sp[mu] - sm[mu]) / (2.0 * h);
                    assert!(
                        (fd - g[mu][axis]).abs() < 1e-8,
                        "l={l} mu={mu} axis={axis}: fd {fd} vs {}",
                        g[mu][axis]
                    );
                }
            }
        }
    }
}

#[test]
fn zero_vector_is_handled() {
    assert!((real_ylm(0, [0.0; 3])[0] - 1.0 / (4.0 * PI).sqrt()).abs() < 1e-16);
    assert_eq!(real_ylm(2, [0.0; 3]), vec![0.0; 5]);
    let (_, g) = real_ylm_with_gradient(1, [0.0; 3]);
    assert!(g.iter().all(|v| *v == [0.0; 3]));
}
