use crate::{InnerBehavior, LogGrid, RadialTable};

fn gaussian_table(inner: InnerBehavior) -> RadialTable {
    let g = LogGrid::new(-7.5, 0.0186, 500).unwrap();
    let vals = (0..g.len()).map(|i| (-g.r(i).powi(2)).exp()).collect();
    RadialTable::new(g, vals, inner).unwrap()
}

#[test]
fn spline_reproduces_gaussian_and_derivative() {
    let t = gaussian_table(InnerBehavior::Constant);
    for &r in &[0.01, 0.3, 0.77, 1.5, 2.9] {
        let (v, d) = t.eval_with_derivative(r);
        let ev = (-r * r).exp();
        assert!((v - ev).abs() < 1e-8, "r={r}: value err {}", (v - ev).abs());
        let ed = -2.0 * r * ev;
        assert!((d - ed).abs() < 1e-6, "r={r}: deriv err {}", (d - ed).abs());
    }
}

#[test]
fn out_of_grid_rules() {
    let r_min = LogGrid::new(-7.5, 0.0186, 500).unwrap().r_min();
    let c = gaussian_table(InnerBehavior::Constant);
    assert_eq!(c.eval(0.5 * r_min), c.values()[0]);
    assert_eq!(c.eval(0.0), c.values()[0]);
    let z = gaussian_table(InnerBehavior::Zero);
    assert_eq!(z.eval(0.5 * r_min), 0.0);
    let l = gaussian_table(InnerBehavior::LinearFromFirstInterval);
    let v = l.eval(0.5 * r_min);
    assert!(v > 0.0 && (v - l.values()[0]).abs() < 1e-3);
    let last = c.grid().r_last();
    assert_eq!(c.eval(last * 1.000001), 0.0);
    assert!(c.eval(last) != 0.0 || c.values()[499] == 0.0);
    assert_eq!(c.clone().with_cutoff(1.0).eval(1.0), 0.0);
}

#[test]
fn grid_validation() {
    assert!(LogGrid::new(0.0, 0.0, 10).is_err());
    assert!(LogGrid::new(0.0, 0.1, 3).is_err());
    let xs = [0.0, 0.1, 0.2, 0.31];
    assert!(LogGrid::from_x_values(&xs).is_err());
    let xs = [0.0, 0.1, 0.2, 0.3];
    let rs: Vec<f64> = xs.iter().map(|x: &f64| x.exp()).collect();
    assert!(RadialTable::from_columns(&xs, &rs, vec![1.0; 4], InnerBehavior::Zero).is_ok());
    let mut bad = rs.clone();
    bad[2] *= 1.001;
    assert!(RadialTable::from_columns(&xs, &bad, vec![1.0; 4], InnerBehavior::Zero).is_err());
}

#[test]
fn trapezoid_integrals_and_combine() {
    let t = gaussian_table(InnerBehavior::Constant);
    // int 4 pi r^2 e^{-r^2} dr = pi^{3/2}
    let n = t.trapezoid_4pi_r2();
    // the trapezoid rule on the log grid has relative error ~6e-5 (the book quotes ~1e-4)
    let exact_n = std::f64::consts::PI.powf(1.5);
    assert!((n - exact_n).abs() < 2e-4 * exact_n);
    let sq = t.trapezoid_r2_product(&t).unwrap();
    // int r^2 e^{-2r^2} dr = sqrt(pi)/(4 * 2^{3/2})
    let exact = std::f64::consts::PI.sqrt() / (4.0 * 2f64.powf(1.5));
    assert!((sq - exact).abs() < 2e-4 * exact);
    let c = RadialTable::combine(2.0, &t, -1.0, &t).unwrap();
    assert!((c.eval(0.9) - t.eval(0.9)).abs() < 1e-14);
    let other = RadialTable::new(
        LogGrid::new(-7.0, 0.0186, 500).unwrap(),
        vec![0.0; 500],
        InnerBehavior::Zero,
    )
    .unwrap();
    assert!(t.trapezoid_r2_product(&other).is_err());
}
