//! Spherical Bessel functions and the radial transform onto the wave-number grid.
//!
//! Implements `two-center-overlap-kinetic.md`, "Method" §1 (transformation of radial functions) and
//! "Numerical parameters of the wave-number integration" (a), and the transform pair of
//! `notation.md` ("Fourier transform convention"):
//!
//! ```text
//! R~_l(k) = int_0^inf dr r^2 j_l(kr) R_l(r),    R_l(r) = (2/pi) int_0^inf dk k^2 j_l(kr) R~_l(k)
//! ```

use std::f64::consts::PI;

use crate::radial::RadialTable;

/// Spherical Bessel function `j_l(x)` for `x >= 0`.
///
/// Uses the power series for `x < max(l, 0.5)` and upward recurrence otherwise (stable there).
///
/// ```
/// use sphaira_atomdata::spherical_bessel_j;
/// assert!((spherical_bessel_j(0, 1.0) - 1.0f64.sin()).abs() < 1e-15);
/// assert_eq!(spherical_bessel_j(2, 0.0), 0.0);
/// assert_eq!(spherical_bessel_j(0, 0.0), 1.0);
/// ```
pub fn spherical_bessel_j(l: usize, x: f64) -> f64 {
    if x == 0.0 {
        return if l == 0 { 1.0 } else { 0.0 };
    }
    if x < (l as f64).max(0.5) {
        // j_l(x) = x^l/(2l+1)!! sum_k (-x^2/2)^k / (k! (2l+3)(2l+5)...(2l+2k+1))
        let mut pref = 1.0;
        for k in 1..=l {
            pref *= x / (2 * k + 1) as f64;
        }
        let mut term = 1.0;
        let mut sum = 1.0;
        let h = -0.5 * x * x;
        for k in 1..200 {
            term *= h / (k as f64 * (2 * l + 2 * k + 1) as f64);
            sum += term;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        return pref * sum;
    }
    let (s, c) = x.sin_cos();
    let j0 = s / x;
    if l == 0 {
        return j0;
    }
    let j1 = (s / x - c) / x;
    let (mut a, mut b) = (j0, j1);
    for k in 1..l {
        let next = (2 * k + 1) as f64 / x * b - a;
        a = b;
        b = next;
    }
    b
}

/// Parameters of the radial transform (`two-center-overlap-kinetic.md`, "Input keywords").
///
/// Defaults: `1DFFT.EnergyCutoff = 3600` Ry, `1DFFT.NumGridK = 900`, `1DFFT.NumGridR = 900`.
///
/// ```
/// use sphaira_atomdata::BesselParams;
/// let p = BesselParams::default();
/// assert_eq!(p.k_max(), 60.0);
/// assert!((p.k_step() - 60.0 / 900.0).abs() < 1e-15);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BesselParams {
    /// `1DFFT.EnergyCutoff` in Ry; `k_max = sqrt(E)` bohr^-1 (`notation.md`, "Units").
    pub energy_cutoff_ry: f64,
    /// `1DFFT.NumGridK`: number of stored wave numbers.
    pub num_grid_k: usize,
    /// `1DFFT.NumGridR`: number of real-space intervals of the transform integral.
    pub num_grid_r: usize,
}

impl Default for BesselParams {
    fn default() -> Self {
        Self {
            energy_cutoff_ry: 3600.0,
            num_grid_k: 900,
            num_grid_r: 900,
        }
    }
}

impl BesselParams {
    /// `k_max = sqrt(E_cut[Ry])` in bohr^-1.
    pub fn k_max(&self) -> f64 {
        self.energy_cutoff_ry.sqrt()
    }

    /// Spacing `k_max / N_K` of the stored wave-number grid `k_j = j k_max / N_K`.
    pub fn k_step(&self) -> f64 {
        self.k_max() / self.num_grid_k as f64
    }
}

/// A radial function in wave-number space on `k_j = j dk`, `j = 0..N_K`.
///
/// `eval` interpolates with 4-point Lagrange (cubic) interpolation and returns 0 beyond the last
/// stored point (`two-center-overlap-kinetic.md`, "Outside storage").
#[derive(Clone, Debug, PartialEq)]
pub struct WaveNumberTable {
    l: usize,
    dk: f64,
    values: Vec<f64>,
}

impl WaveNumberTable {
    /// Angular momentum of the transform.
    pub fn l(&self) -> usize {
        self.l
    }

    /// Grid spacing `dk`.
    pub fn dk(&self) -> f64 {
        self.dk
    }

    /// Stored values `R~(k_j)`.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// `R~(k)` by cubic interpolation; 0 for `k` beyond the last point; NaN for negative `k`.
    pub fn eval(&self, k: f64) -> f64 {
        if k < 0.0 || k.is_nan() {
            return f64::NAN;
        }
        let n = self.values.len();
        let s = k / self.dk;
        if s > (n - 1) as f64 {
            return 0.0;
        }
        // INVARIANT: n >= 4 is enforced by the constructors, so the 4-point stencil fits.
        let i = (s.floor() as usize).clamp(1, n - 3) - 1;
        let t = s - i as f64; // in [0, 3]; the interval of interest is [1, 2]
        let v = &self.values[i..i + 4];
        let (a, b, c, d) = (t, t - 1.0, t - 2.0, t - 3.0);
        -b * c * d / 6.0 * v[0] + a * c * d / 2.0 * v[1] - a * b * d / 2.0 * v[2]
            + a * b * c / 6.0 * v[3]
    }

    /// Back transform `R(r) = (2/pi) int dk k^2 j_l(kr) R~(k)` at the given radii, by the
    /// trapezoidal rule on the stored grid (last point with half weight).
    ///
    /// Accuracy is limited by the end correction of the trapezoidal rule for `l >= 1`
    /// (`O(dk^4)`); it is exponentially accurate for `l = 0` smooth, decaying `R~`.
    pub fn to_real_space(&self, radii: &[f64]) -> Vec<f64> {
        let n = self.values.len();
        radii
            .iter()
            .map(|&r| {
                let sum: f64 = (1..n)
                    .map(|j| {
                        let k = j as f64 * self.dk;
                        let w = if j == n - 1 { 0.5 } else { 1.0 };
                        w * k * k * spherical_bessel_j(self.l, k * r) * self.values[j]
                    })
                    .sum();
                2.0 / PI * self.dk * sum
            })
            .collect()
    }
}

/// Forward spherical Bessel transform `R~(k_j) = int dr r^2 j_l(k_j r) R(r)` of a radial table.
///
/// The integral runs over `r_min <= r <= r_upper` (the book uses `r_c + 0.5` bohr; `r_upper` is
/// clipped to the last grid point) with `N_R` equal intervals in `r` and the trapezoidal rule;
/// `R(r)` is interpolated from the table. The result is stored on `k_j = j k_max / N_K`,
/// `j = 0..N_K` (`two-center-overlap-kinetic.md`, "Numerical parameters", (a)). No prefactor.
///
/// # Panics
///
/// If `params.num_grid_k < 4` or `params.num_grid_r == 0`.
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::{radial_to_wavenumber, BesselParams, InnerBehavior, LogGrid, RadialTable};
/// let g = LogGrid::new(-7.5, 0.02, 500).unwrap();
/// let v = (0..500).map(|i| (-g.r(i).powi(2)).exp()).collect();
/// let t = RadialTable::new(g, v, InnerBehavior::Constant).unwrap();
/// let p = BesselParams { energy_cutoff_ry: 100.0, num_grid_k: 100, num_grid_r: 400 };
/// let w = radial_to_wavenumber(&t, 0, 8.0, &p);
/// // int r^2 e^{-r^2} dr = sqrt(pi)/4 at k = 0
/// assert!((w.values()[0] - std::f64::consts::PI.sqrt() / 4.0).abs() < 1e-6);
/// ```
pub fn radial_to_wavenumber(
    table: &RadialTable,
    l: usize,
    r_upper: f64,
    params: &BesselParams,
) -> WaveNumberTable {
    assert!(params.num_grid_k >= 4, "num_grid_k must be at least 4");
    assert!(params.num_grid_r >= 1, "num_grid_r must be at least 1");
    let r_min = table.grid().r_min();
    let r_hi = r_upper.min(table.grid().r_last());
    let nr = params.num_grid_r;
    let h = (r_hi - r_min) / nr as f64;
    // weight * r^2 * R(r) at each quadrature point
    let g: Vec<(f64, f64)> = (0..=nr)
        .map(|j| {
            let r = r_min + h * j as f64;
            let w = if j == 0 || j == nr { 0.5 } else { 1.0 };
            (r, w * h * r * r * table.eval(r))
        })
        .collect();
    let dk = params.k_step();
    let values = (0..params.num_grid_k)
        .map(|j| {
            let k = j as f64 * dk;
            g.iter()
                .map(|&(r, wg)| wg * spherical_bessel_j(l, k * r))
                .sum()
        })
        .collect();
    WaveNumberTable { l, dk, values }
}

#[cfg(test)]
mod tests;
