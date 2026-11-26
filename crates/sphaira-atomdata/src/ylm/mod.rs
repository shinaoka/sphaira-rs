//! Real spherical harmonics `S_{lm}` in the book's ordering and sign convention, with gradients.
//!
//! Implements `pao-basis.md`, "Real spherical harmonics (Convention)": orthonormal, all
//! polynomial coefficients positive, component ordering
//!
//! | `l` | order |
//! |---|---|
//! | 0 | `1` |
//! | 1 | `x, y, z` (i.e. `cos phi`, `sin phi`, `m=0`) |
//! | 2 | `3z^2-1, x^2-y^2, xy, xz, yz` (`m=0`, `cos 2phi`, `sin 2phi`, `cos phi`, `sin phi`) |
//! | >= 3 | `m=0, cos phi, sin phi, cos 2phi, sin 2phi, ..., cos l phi, sin l phi` |

use std::f64::consts::PI;

/// Kind of a real spherical-harmonic component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YlmKind {
    /// `m = 0`.
    Zero,
    /// Proportional to `cos(m phi)`.
    Cos,
    /// Proportional to `sin(m phi)`.
    Sin,
}

/// `(|m|, kind)` of one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YlmComponent {
    /// `|m|`.
    pub m: usize,
    /// Cosine, sine or `m = 0`.
    pub kind: YlmKind,
}

/// Component ordering for angular momentum `l` (`2l+1` entries), as tabulated in `pao-basis.md`.
///
/// ```
/// use sphaira_atomdata::{component_order, YlmKind};
/// let d = component_order(2);
/// assert_eq!((d[1].m, d[1].kind), (2, YlmKind::Cos)); // x^2 - y^2 comes right after 3z^2-1
/// assert_eq!((d[3].m, d[3].kind), (1, YlmKind::Cos)); // xz
/// ```
pub fn component_order(l: usize) -> Vec<YlmComponent> {
    let c = |m, kind| YlmComponent { m, kind };
    match l {
        0 => vec![c(0, YlmKind::Zero)],
        1 => vec![c(1, YlmKind::Cos), c(1, YlmKind::Sin), c(0, YlmKind::Zero)],
        2 => vec![
            c(0, YlmKind::Zero),
            c(2, YlmKind::Cos),
            c(2, YlmKind::Sin),
            c(1, YlmKind::Cos),
            c(1, YlmKind::Sin),
        ],
        _ => {
            let mut v = vec![c(0, YlmKind::Zero)];
            for m in 1..=l {
                v.push(c(m, YlmKind::Cos));
                v.push(c(m, YlmKind::Sin));
            }
            v
        }
    }
}

/// `Q_l^m(z) = d^m P_l / dz^m` for `m <= l` (0 for `m > l`).
fn legendre_deriv(l: usize, m: usize, z: f64) -> f64 {
    if m > l {
        return 0.0;
    }
    let mut q_mm = 1.0;
    for k in 1..=m {
        q_mm *= (2 * k - 1) as f64;
    }
    if l == m {
        return q_mm;
    }
    let mut prev = q_mm;
    let mut cur = (2 * m + 1) as f64 * z * q_mm;
    for ll in m + 2..=l {
        let next = ((2 * ll - 1) as f64 * z * cur - (ll + m - 1) as f64 * prev) / (ll - m) as f64;
        prev = cur;
        cur = next;
    }
    cur
}

/// Normalization `N_{lm}` such that `S_{lm} = N_{lm} Q_l^m(z) {1, Re, Im}(x+iy)^m` is orthonormal.
fn norm(l: usize, m: usize) -> f64 {
    if m == 0 {
        return ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
    }
    let mut ratio = 1.0; // (l-m)!/(l+m)!
    for k in l - m + 1..=l + m {
        ratio /= k as f64;
    }
    ((2 * l + 1) as f64 / (2.0 * PI) * ratio).sqrt()
}

/// Values (and gradients with respect to the unit vector `u`, treating `S` as the polynomial in
/// `(x, y, z)` written above) of all components.
fn core(l: usize, u: [f64; 3], want_grad: bool) -> (Vec<f64>, Vec<[f64; 3]>) {
    let [x, y, z] = u;
    let mut c = vec![0.0; l + 1];
    let mut s = vec![0.0; l + 1];
    c[0] = 1.0;
    for m in 1..=l {
        c[m] = x * c[m - 1] - y * s[m - 1];
        s[m] = x * s[m - 1] + y * c[m - 1];
    }
    let comps = component_order(l);
    let mut vals = Vec::with_capacity(comps.len());
    let mut grads = Vec::with_capacity(if want_grad { comps.len() } else { 0 });
    for YlmComponent { m, kind } in comps {
        let n = norm(l, m);
        let q = legendre_deriv(l, m, z);
        let (cs, dcs_dx, dcs_dy) = match kind {
            YlmKind::Zero => (1.0, 0.0, 0.0),
            YlmKind::Cos => (c[m], m as f64 * c[m - 1], -(m as f64) * s[m - 1]),
            YlmKind::Sin => (s[m], m as f64 * s[m - 1], m as f64 * c[m - 1]),
        };
        vals.push(n * q * cs);
        if want_grad {
            let dq = legendre_deriv(l, m + 1, z);
            grads.push([n * q * dcs_dx, n * q * dcs_dy, n * dq * cs]);
        }
    }
    (vals, grads)
}

/// Writes `S_{l mu}(r_hat)`, `mu = 0..2l+1` in the book's order, into `out`.
///
/// `dir` need not be normalized; a zero vector gives `1/sqrt(4 pi)` for `l = 0` and 0 otherwise.
///
/// # Panics
///
/// If `out.len() != 2l + 1`.
pub fn real_ylm_into(l: usize, dir: [f64; 3], out: &mut [f64]) {
    assert_eq!(out.len(), 2 * l + 1, "output must have 2l+1 entries");
    let r = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if r == 0.0 {
        out.fill(0.0);
        if l == 0 {
            out[0] = 1.0 / (4.0 * PI).sqrt();
        }
        return;
    }
    let (v, _) = core(l, [dir[0] / r, dir[1] / r, dir[2] / r], false);
    out.copy_from_slice(&v);
}

/// `S_{l mu}(r_hat)` for `mu = 0..2l+1` in the book's order.
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::real_ylm;
/// let p = real_ylm(1, [0.0, 0.0, 2.0]); // x, y, z order
/// assert!(p[0].abs() < 1e-15 && p[1].abs() < 1e-15);
/// assert!((p[2] - (3.0 / (4.0 * std::f64::consts::PI)).sqrt()).abs() < 1e-15);
/// ```
pub fn real_ylm(l: usize, dir: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0; 2 * l + 1];
    real_ylm_into(l, dir, &mut out);
    out
}

/// Values and gradients of `S_{l mu}(r / |r|)` with respect to the Cartesian vector `r`
/// (bohr^-1 units: the gradient is tangential and scales as `1/|r|`).
///
/// The gradient is analytic and regular at the poles (no `1/sin(theta)` terms). At `r = 0` the
/// values follow [`real_ylm_into`] and all gradients are 0.
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::real_ylm_with_gradient;
/// // p_x = c x/r; d p_x/dx at (0,0,1) is c.
/// let (_, g) = real_ylm_with_gradient(1, [0.0, 0.0, 1.0]);
/// let c = (3.0 / (4.0 * std::f64::consts::PI)).sqrt();
/// assert!((g[0][0] - c).abs() < 1e-15);
/// ```
pub fn real_ylm_with_gradient(l: usize, r: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if len == 0.0 {
        return (real_ylm(l, r), vec![[0.0; 3]; 2 * l + 1]);
    }
    let u = [r[0] / len, r[1] / len, r[2] / len];
    let (vals, grads) = core(l, u, true);
    let grads = grads
        .into_iter()
        .map(|g| {
            let radial = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];
            [
                (g[0] - radial * u[0]) / len,
                (g[1] - radial * u[1]) / len,
                (g[2] - radial * u[2]) / len,
            ]
        })
        .collect();
    (vals, grads)
}

#[cfg(test)]
mod tests;
