//! Integrates two textbook ODEs with RK4 and Dormand-Prince, comparing against the
//! analytic solutions.
//!
//! Run with:
//! ```bash
//! cargo run --example ode_solvers
//! ```
//!
//! Systems:
//! - Exponential decay `dy/dt = -k*y`, analytic `y(t) = y0 * exp(-k*t)`.
//! - Undamped harmonic oscillator `y'' = -w^2 * y`, analytic
//!   `y(t) = y0*cos(w*t) + (v0/w)*sin(w*t)`.

use liquid_neural_network::prelude::*;

/// Exponential decay: `dy/dt = -rate * y`.
struct Decay {
    rate: f64,
}

impl OdeSystem for Decay {
    fn derivative(&self, _t: f64, state: &[f64], _input: &[f64]) -> Vec<f64> {
        vec![-self.rate * state[0]]
    }
    fn dimension(&self) -> usize {
        1
    }
}

/// Undamped harmonic oscillator with angular frequency `omega`:
/// `state = [position, velocity]`, `dx/dt = v`, `dv/dt = -omega^2 * x`.
struct Oscillator {
    omega: f64,
}

impl OdeSystem for Oscillator {
    fn derivative(&self, _t: f64, state: &[f64], _input: &[f64]) -> Vec<f64> {
        vec![state[1], -self.omega * self.omega * state[0]]
    }
    fn dimension(&self) -> usize {
        2
    }
}

fn main() -> Result<()> {
    // --- Exponential decay ---------------------------------------------------
    let rate = 1.5;
    let y0 = 2.0;
    let t1 = 3.0;
    let decay = Decay { rate };
    let analytic_decay = y0 * (-rate * t1).exp();

    // Fixed-step RK4 vs adaptive Dormand-Prince (dt is the initial step for DP).
    let rk4 = RungeKutta4;
    let dp = DormandPrince::default();
    let cfg = SolverConfig::new(0.01);

    let rk4_decay = rk4.integrate(&decay, vec![y0], &[], 0.0, t1, &cfg)?;
    let dp_decay = dp.integrate(&decay, vec![y0], &[], 0.0, t1, &cfg)?;

    println!("Exponential decay  dy/dt = -{rate}*y,  y(0) = {y0},  t = {t1}");
    println!("  analytic       = {analytic_decay:.8}");
    println!(
        "  RK4            = {:.8}  (err {:.2e})",
        rk4_decay.values[0],
        (rk4_decay.values[0] - analytic_decay).abs()
    );
    println!(
        "  Dormand-Prince = {:.8}  (err {:.2e})",
        dp_decay.values[0],
        (dp_decay.values[0] - analytic_decay).abs()
    );

    // --- Harmonic oscillator -------------------------------------------------
    let omega = 2.0;
    let x0 = 1.0;
    let v0 = 0.0;
    let t1 = 5.0;
    let osc = Oscillator { omega };
    let analytic_pos = x0 * (omega * t1).cos() + (v0 / omega) * (omega * t1).sin();

    let rk4_osc = rk4.integrate(&osc, vec![x0, v0], &[], 0.0, t1, &cfg)?;
    let dp_osc = dp.integrate(&osc, vec![x0, v0], &[], 0.0, t1, &cfg)?;

    println!("\nHarmonic oscillator  y'' = -{omega}^2*y,  y(0) = {x0},  y'(0) = {v0},  t = {t1}");
    println!("  analytic       = {analytic_pos:.8}");
    println!(
        "  RK4            = {:.8}  (err {:.2e})",
        rk4_osc.values[0],
        (rk4_osc.values[0] - analytic_pos).abs()
    );
    println!(
        "  Dormand-Prince = {:.8}  (err {:.2e})",
        dp_osc.values[0],
        (dp_osc.values[0] - analytic_pos).abs()
    );

    Ok(())
}
