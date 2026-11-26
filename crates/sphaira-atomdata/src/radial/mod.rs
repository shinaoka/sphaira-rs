//! Radial grids equally spaced in `ln r` and interpolated radial tables.
//!
//! Implements the radial-table rules of `pao-basis.md` ("Table of radial functions", "Values outside
//! the grid") and `nonlocal-pseudopotential.md` §2.7. The book leaves the interpolation scheme free
//! ((Design)); this crate uses a clamped cubic spline in `x = ln r` (see [`RadialTable`]).

use crate::error::AtomDataError;

/// Relative tolerance for `r == exp(x)` when reading tables (files carry 12-15 digits).
const R_EXP_TOLERANCE: f64 = 1e-8;
/// Absolute tolerance for uniformity of `x` when reading tables.
const X_UNIFORM_TOLERANCE: f64 = 1e-8;

/// Radial grid `x_i = x_min + i dx`, `r_i = exp(x_i)`, `i = 0..len`.
///
/// Implements the grid description of `pao-basis.md`, "Table of radial functions".
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::LogGrid;
/// let g = LogGrid::new(-7.5, 0.1, 10).unwrap();
/// assert!((g.r(0) - (-7.5f64).exp()).abs() < 1e-18);
/// assert_eq!(g.len(), 10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogGrid {
    x_min: f64,
    dx: f64,
    len: usize,
}

impl LogGrid {
    /// Creates a grid.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::Grid`] if `dx <= 0`, a value is not finite, or `len < 4`.
    pub fn new(x_min: f64, dx: f64, len: usize) -> Result<Self, AtomDataError> {
        if !(x_min.is_finite() && dx.is_finite() && dx > 0.0) || len < 4 {
            return Err(AtomDataError::Grid {
                detail: format!("x_min={x_min}, dx={dx}, len={len}"),
            });
        }
        Ok(Self { x_min, dx, len })
    }

    /// Builds the grid from the `x` column of a table and checks uniform spacing.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::Grid`] if fewer than 4 points are given, `x` is not increasing, or the
    /// spacing deviates from uniform by more than `1e-8`.
    pub fn from_x_values(xs: &[f64]) -> Result<Self, AtomDataError> {
        if xs.len() < 4 {
            return Err(AtomDataError::Grid {
                detail: format!("only {} points", xs.len()),
            });
        }
        let dx = (xs[xs.len() - 1] - xs[0]) / (xs.len() - 1) as f64;
        let grid = Self::new(xs[0], dx, xs.len())?;
        for (i, &x) in xs.iter().enumerate() {
            if (x - grid.x(i)).abs() > X_UNIFORM_TOLERANCE {
                return Err(AtomDataError::Grid {
                    detail: format!("x[{i}]={x} deviates from uniform spacing {dx}"),
                });
            }
        }
        Ok(grid)
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false` (a grid has at least 4 points); provided for API completeness.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// `x` of the first point.
    pub fn x_min(&self) -> f64 {
        self.x_min
    }

    /// Spacing in `x = ln r`.
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// `x_i = ln r_i`.
    pub fn x(&self, i: usize) -> f64 {
        self.x_min + self.dx * i as f64
    }

    /// `r_i` in bohr.
    pub fn r(&self, i: usize) -> f64 {
        self.x(i).exp()
    }

    /// First grid radius `r_min`.
    pub fn r_min(&self) -> f64 {
        self.r(0)
    }

    /// Last grid radius `r_last`.
    pub fn r_last(&self) -> f64 {
        self.r(self.len - 1)
    }

    /// True when both grids have the same size and `x` values to `1e-8`.
    pub fn same_as(&self, other: &Self) -> bool {
        self.len == other.len
            && (self.x_min - other.x_min).abs() < X_UNIFORM_TOLERANCE
            && (self.dx - other.dx).abs() < X_UNIFORM_TOLERANCE / self.len as f64
    }
}

/// Behaviour of a [`RadialTable`] inside the first grid point (`r < r_min`).
///
/// Implements the "Values outside the grid" table of `pao-basis.md` and `nonlocal-pseudopotential.md` §2.7.
/// Outside the last grid point a table is zero unless its [`OuterBehavior`] says otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerBehavior {
    /// The value at the first grid point (PAO `l = 0`, densities, local potential).
    Constant,
    /// Zero (PAO and projectors with `l >= 1`).
    Zero,
    /// Linear extrapolation in `r` from the first grid interval (VPS projectors with `l = 0`).
    LinearFromFirstInterval,
    /// Even quadratic `a r^2 + b` continuation (VPS partial-core-correction density,
    /// `nonlocal-pseudopotential.md` §2.7).
    ///
    /// **Implementation choice (the book does not fix the matching point):** the quadratic is
    /// matched at the first grid radius `r_min` in value and slope (C1), i.e.
    /// `a = f'(r_min) / (2 r_min)` and `b = f(r_min) - a r_min^2`, with `f'` the derivative of the
    /// interpolating spline at `r_min`. The result is even in `r`, so it is smooth at the origin.
    EvenQuadratic,
}

/// Behaviour of a [`RadialTable`] beyond the last grid point (`r > r_last`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OuterBehavior {
    /// Exactly zero (PAO, projectors, local potential).
    #[default]
    Zero,
    /// The last tabulated value (VPS partial-core-correction density,
    /// `nonlocal-pseudopotential.md` §2.7).
    Constant,
}

impl InnerBehavior {
    /// The PAO rule: constant for `l = 0`, zero for `l >= 1` (`pao-basis.md`, "Values outside the grid").
    pub fn pao(l: usize) -> Self {
        if l == 0 { Self::Constant } else { Self::Zero }
    }

    /// The VPS projector rule: linear extrapolation for `l = 0`, zero for `l >= 1`
    /// (`nonlocal-pseudopotential.md` §2.7).
    pub fn projector(l: usize) -> Self {
        if l == 0 {
            Self::LinearFromFirstInterval
        } else {
            Self::Zero
        }
    }
}

/// A function of `r` tabulated on a [`LogGrid`], with interpolation and out-of-grid rules.
///
/// **Interpolation.** A clamped cubic spline in `x = ln r` (C2; end slopes from the third-order
/// one-sided finite difference). The book does not prescribe a scheme (`pao-basis.md`, "Values
/// outside the grid", (Design)); the spline error is `O(dx^4)`, about `1e-7` relative for the
/// distributed files.
///
/// **Out-of-grid values.** `r > r_last` gives 0; `r < r_min` follows the [`InnerBehavior`]; if a
/// cutoff radius is set (see [`RadialTable::with_cutoff`]) every `r >= r_c` gives exactly 0
/// (allowed by `pao-basis.md`, "Values outside the grid", (Design)).
///
/// # Examples
///
/// ```
/// use sphaira_atomdata::{InnerBehavior, LogGrid, RadialTable};
/// let g = LogGrid::new(-5.0, 0.05, 200).unwrap();
/// let vals: Vec<f64> = (0..g.len()).map(|i| (-g.r(i).powi(2)).exp()).collect();
/// let t = RadialTable::new(g, vals, InnerBehavior::Constant).unwrap();
/// assert!((t.eval(0.7) - (-0.49f64).exp()).abs() < 1e-6);
/// assert_eq!(t.eval(1.0e3), 0.0); // outside the last grid point
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RadialTable {
    grid: LogGrid,
    values: Vec<f64>,
    d2: Vec<f64>,
    inner: InnerBehavior,
    outer: OuterBehavior,
    cutoff: Option<f64>,
}

impl RadialTable {
    /// Creates a table from values on `grid`.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::Grid`] if `values.len() != grid.len()`.
    pub fn new(
        grid: LogGrid,
        values: Vec<f64>,
        inner: InnerBehavior,
    ) -> Result<Self, AtomDataError> {
        if values.len() != grid.len() {
            return Err(AtomDataError::Grid {
                detail: format!("{} values for {} grid points", values.len(), grid.len()),
            });
        }
        let d2 = spline_second_derivatives(&values, grid.dx());
        Ok(Self {
            grid,
            values,
            d2,
            inner,
            outer: OuterBehavior::Zero,
            cutoff: None,
        })
    }

    /// Creates a table from the columns `x`, `r`, `value` of a file, validating `r = exp(x)` to a
    /// relative `1e-8` and uniform spacing in `x`.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::Grid`] on length mismatch, non-uniform `x`, or `r != exp(x)`.
    pub fn from_columns(
        xs: &[f64],
        rs: &[f64],
        values: Vec<f64>,
        inner: InnerBehavior,
    ) -> Result<Self, AtomDataError> {
        if xs.len() != rs.len() || xs.len() != values.len() {
            return Err(AtomDataError::Grid {
                detail: "columns have different lengths".into(),
            });
        }
        let grid = LogGrid::from_x_values(xs)?;
        for (i, (&x, &r)) in xs.iter().zip(rs).enumerate() {
            let e = x.exp();
            if (r - e).abs() > R_EXP_TOLERANCE * e {
                return Err(AtomDataError::Grid {
                    detail: format!("row {i}: r={r} but exp(x)={e}"),
                });
            }
        }
        Self::new(grid, values, inner)
    }

    /// Returns the table with the given rule beyond the last grid point.
    pub fn with_outer(mut self, outer: OuterBehavior) -> Self {
        self.outer = outer;
        self
    }

    /// The rule beyond the last grid point.
    pub fn outer(&self) -> OuterBehavior {
        self.outer
    }

    /// Returns the table with a hard cutoff: `eval` is exactly 0 for `r >= r_c`.
    pub fn with_cutoff(mut self, r_c: f64) -> Self {
        self.cutoff = Some(r_c);
        self
    }

    /// The hard cutoff radius, if any.
    pub fn cutoff(&self) -> Option<f64> {
        self.cutoff
    }

    /// The grid.
    pub fn grid(&self) -> &LogGrid {
        &self.grid
    }

    /// Stored values (as read, no cutoff applied).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The inner-region rule.
    pub fn inner(&self) -> InnerBehavior {
        self.inner
    }

    /// Interpolated value at `r` (bohr).
    pub fn eval(&self, r: f64) -> f64 {
        self.eval_with_derivative(r).0
    }

    /// Interpolated value and `d/dr` at `r` (bohr).
    pub fn eval_with_derivative(&self, r: f64) -> (f64, f64) {
        if r.is_nan() || self.cutoff.is_some_and(|rc| r >= rc) {
            return (0.0, 0.0);
        }
        if r > self.grid.r_last() {
            return match self.outer {
                OuterBehavior::Zero => (0.0, 0.0),
                OuterBehavior::Constant => (self.values[self.grid.len() - 1], 0.0),
            };
        }
        let r_min = self.grid.r_min();
        if r < r_min {
            return match self.inner {
                InnerBehavior::Constant => (self.values[0], 0.0),
                InnerBehavior::Zero => (0.0, 0.0),
                InnerBehavior::LinearFromFirstInterval => {
                    let slope = (self.values[1] - self.values[0]) / (self.grid.r(1) - r_min);
                    (self.values[0] + slope * (r - r_min), slope)
                }
                InnerBehavior::EvenQuadratic => {
                    let slope = self.spline(r_min).1;
                    let a = slope / (2.0 * r_min);
                    let b = self.values[0] - a * r_min * r_min;
                    (a * r * r + b, 2.0 * a * r)
                }
            };
        }
        self.spline(r)
    }

    /// Spline value and derivative at `r` with `r_min <= r <= r_last`.
    fn spline(&self, r: f64) -> (f64, f64) {
        let n = self.grid.len();
        let s = (r.ln() - self.grid.x_min()) / self.grid.dx();
        // INVARIANT: negative `s` saturates to 0 in the cast; `i <= n - 2` keeps `i + 1` in range.
        let i = (s.floor() as usize).min(n - 2);
        let t = (s - i as f64).clamp(0.0, 1.0);
        let h = self.grid.dx();
        let (y0, y1) = (self.values[i], self.values[i + 1]);
        let (m0, m1) = (self.d2[i], self.d2[i + 1]);
        let u = 1.0 - t;
        let val = u * y0 + t * y1 + h * h / 6.0 * ((u * u * u - u) * m0 + (t * t * t - t) * m1);
        let dx_deriv =
            (y1 - y0) / h + h / 6.0 * (-(3.0 * u * u - 1.0) * m0 + (3.0 * t * t - 1.0) * m1);
        (val, dx_deriv / r)
    }

    /// `integral r^2 f(r) g(r) dr` by the trapezoidal rule on the table points (as in the book's
    /// orthonormality check, `pao-basis.md` "Normalization and orthogonality").
    ///
    /// # Errors
    ///
    /// [`AtomDataError::GridMismatch`] if the two tables are on different grids.
    pub fn trapezoid_r2_product(&self, other: &Self) -> Result<f64, AtomDataError> {
        if !self.grid.same_as(&other.grid) {
            return Err(AtomDataError::GridMismatch {
                detail: "trapezoid_r2_product".into(),
            });
        }
        let f = |i: usize| {
            let r = self.grid.r(i);
            r * r * self.values[i] * other.values[i]
        };
        Ok((0..self.grid.len() - 1)
            .map(|i| 0.5 * (f(i) + f(i + 1)) * (self.grid.r(i + 1) - self.grid.r(i)))
            .sum())
    }

    /// `integral 4 pi r^2 f(r) dr` by the trapezoidal rule on the table points (for densities).
    pub fn trapezoid_4pi_r2(&self) -> f64 {
        let f = |i: usize| {
            let r = self.grid.r(i);
            4.0 * std::f64::consts::PI * r * r * self.values[i]
        };
        (0..self.grid.len() - 1)
            .map(|i| 0.5 * (f(i) + f(i + 1)) * (self.grid.r(i + 1) - self.grid.r(i)))
            .sum()
    }

    /// The table `ca * a + cb * b` on the common grid; inner behaviour and cutoff of `a` are kept.
    ///
    /// # Errors
    ///
    /// [`AtomDataError::GridMismatch`] if the grids differ.
    pub fn combine(ca: f64, a: &Self, cb: f64, b: &Self) -> Result<Self, AtomDataError> {
        if !a.grid.same_as(&b.grid) {
            return Err(AtomDataError::GridMismatch {
                detail: "combine".into(),
            });
        }
        let values = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(x, y)| ca * x + cb * y)
            .collect();
        let mut t = Self::new(a.grid, values, a.inner)?;
        t.cutoff = a.cutoff;
        Ok(t)
    }
}

/// Second derivatives (w.r.t. `x`) of the clamped cubic spline through `y` with spacing `h`.
fn spline_second_derivatives(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    // INVARIANT: `LogGrid` guarantees n >= 4, so the end-slope stencils below are in range.
    let d_first = (-11.0 * y[0] + 18.0 * y[1] - 9.0 * y[2] + 2.0 * y[3]) / (6.0 * h);
    let d_last = (11.0 * y[n - 1] - 18.0 * y[n - 2] + 9.0 * y[n - 3] - 2.0 * y[n - 4]) / (6.0 * h);
    let mut sub = vec![1.0; n];
    let mut diag = vec![4.0; n];
    let mut sup = vec![1.0; n];
    let mut rhs = vec![0.0; n];
    diag[0] = 2.0;
    rhs[0] = 6.0 / h * ((y[1] - y[0]) / h - d_first);
    for i in 1..n - 1 {
        rhs[i] = 6.0 / (h * h) * (y[i - 1] - 2.0 * y[i] + y[i + 1]);
    }
    diag[n - 1] = 2.0;
    rhs[n - 1] = 6.0 / h * (d_last - (y[n - 1] - y[n - 2]) / h);
    sub[0] = 0.0;
    sup[n - 1] = 0.0;
    // Thomas algorithm.
    for i in 1..n {
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    let mut m = vec![0.0; n];
    m[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        m[i] = (rhs[i] - sup[i] * m[i + 1]) / diag[i];
    }
    m
}

#[cfg(test)]
mod tests;
