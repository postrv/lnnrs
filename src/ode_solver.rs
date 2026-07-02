// src/ode_solver.rs
//! ODE (Ordinary Differential Equation) solvers for continuous-time neural dynamics.
//!
//! This module provides numerical integration methods for simulating the continuous-time
//! dynamics of Liquid Neural Networks. The primary solver is the 4th-order Runge-Kutta (RK4)
//! method, which provides a good balance between accuracy and computational cost.

use crate::error::{LnnError, Result};

/// State vector for ODE integration.
///
/// Represents the complete state of a dynamical system at a point in time.
#[derive(Debug, Clone)]
pub struct OdeState {
    /// The state variables
    pub values: Vec<f64>,
    /// Current time
    pub time: f64,
}

impl OdeState {
    /// Creates a new ODE state with the given values at time t=0.
    pub fn new(values: Vec<f64>) -> Self {
        OdeState { values, time: 0.0 }
    }

    /// Creates a new ODE state with specified time.
    pub fn with_time(values: Vec<f64>, time: f64) -> Self {
        OdeState { values, time }
    }

    /// Returns the dimension of the state vector.
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// Trait for systems of ordinary differential equations.
///
/// Implementors define the derivative function dy/dt = f(t, y) for the system.
pub trait OdeSystem {
    /// Computes the derivative of the state vector at time t.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `state` - Current state vector
    /// * `input` - External input to the system
    ///
    /// # Returns
    /// The derivative vector dy/dt
    fn derivative(&self, t: f64, state: &[f64], input: &[f64]) -> Vec<f64>;

    /// Returns the dimension of the state space.
    fn dimension(&self) -> usize;
}

/// Configuration for ODE solvers.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Time step for integration
    pub dt: f64,
    /// Maximum number of steps (for safety)
    pub max_steps: usize,
    /// Relative tolerance for adaptive methods
    pub rtol: f64,
    /// Absolute tolerance for adaptive methods
    pub atol: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            dt: 0.01,
            max_steps: 10000,
            rtol: 1e-6,
            atol: 1e-9,
        }
    }
}

impl SolverConfig {
    /// Creates a new solver config with the specified time step.
    pub fn new(dt: f64) -> Self {
        SolverConfig {
            dt,
            ..Default::default()
        }
    }

    /// Sets the maximum number of steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets the relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the absolute tolerance.
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }
}

/// Trait for ODE solvers.
pub trait OdeSolver {
    /// Advances the state by one time step.
    ///
    /// # Arguments
    /// * `system` - The ODE system to solve
    /// * `state` - Current state (modified in place)
    /// * `input` - External input
    /// * `dt` - Time step
    fn step<S: OdeSystem>(&self, system: &S, state: &mut OdeState, input: &[f64], dt: f64);

    /// Integrates the system from t0 to t1.
    ///
    /// # Arguments
    /// * `system` - The ODE system to solve
    /// * `initial_state` - Initial state vector
    /// * `input` - External input (assumed constant over integration)
    /// * `t0` - Start time
    /// * `t1` - End time
    /// * `config` - Solver configuration
    ///
    /// # Returns
    /// The final state, or [`LnnError::OdeSolver`] if `max_steps` is exhausted before
    /// reaching `t1`.
    ///
    /// # Errors
    /// Returns an error when the integration hits `config.max_steps` before the endpoint
    /// `t1` is reached, rather than silently returning a truncated result.
    fn integrate<S: OdeSystem>(
        &self,
        system: &S,
        initial_state: Vec<f64>,
        input: &[f64],
        t0: f64,
        t1: f64,
        config: &SolverConfig,
    ) -> Result<OdeState> {
        let mut state = OdeState::with_time(initial_state, t0);
        let mut steps = 0;
        let end_tol = endpoint_tolerance(t1);

        while t1 - state.time > end_tol {
            if steps >= config.max_steps {
                return Err(max_steps_error(config.max_steps, t1, state.time));
            }
            let dt = (t1 - state.time).min(config.dt);
            self.step(system, &mut state, input, dt);
            state.time += dt;
            steps += 1;
        }

        Ok(state)
    }

    /// Integrates and records the full trajectory.
    ///
    /// # Errors
    /// Returns [`LnnError::OdeSolver`] when `max_steps` is exhausted before reaching `t1`.
    fn integrate_with_trajectory<S: OdeSystem>(
        &self,
        system: &S,
        initial_state: Vec<f64>,
        input: &[f64],
        t0: f64,
        t1: f64,
        config: &SolverConfig,
    ) -> Result<Vec<OdeState>> {
        let mut state = OdeState::with_time(initial_state, t0);
        let mut trajectory = vec![state.clone()];
        let mut steps = 0;
        let end_tol = endpoint_tolerance(t1);

        while t1 - state.time > end_tol {
            if steps >= config.max_steps {
                return Err(max_steps_error(config.max_steps, t1, state.time));
            }
            let dt = (t1 - state.time).min(config.dt);
            self.step(system, &mut state, input, dt);
            state.time += dt;
            trajectory.push(state.clone());
            steps += 1;
        }

        Ok(trajectory)
    }
}

/// Floating-point-safe tolerance for detecting that integration has reached the endpoint.
///
/// Integration terminates once `t1 - t <= EPSILON * max(|t1|, 1.0)`, which avoids taking a
/// spurious extra sub-step (or looping forever) due to rounding when `t` lands just short
/// of `t1`.
fn endpoint_tolerance(t1: f64) -> f64 {
    f64::EPSILON * t1.abs().max(1.0)
}

/// Builds the error returned when a solver exhausts `max_steps` before reaching `t1`.
fn max_steps_error(max_steps: usize, t1: f64, reached: f64) -> LnnError {
    LnnError::ode_solver(format!(
        "max_steps ({max_steps}) exhausted before reaching t1 = {t1} (reached t = {reached})"
    ))
}

/// Euler method - first order accuracy.
///
/// Simple but less accurate. Use for debugging or when speed is critical.
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerSolver;

impl OdeSolver for EulerSolver {
    fn step<S: OdeSystem>(&self, system: &S, state: &mut OdeState, input: &[f64], dt: f64) {
        let dydt = system.derivative(state.time, &state.values, input);
        for (y, dy) in state.values.iter_mut().zip(dydt.iter()) {
            *y += dy * dt;
        }
    }
}

/// 4th-order Runge-Kutta method.
///
/// Standard choice for ODEs - good balance of accuracy and efficiency.
/// Local truncation error is O(dt^5), global error is O(dt^4).
#[derive(Debug, Clone, Copy, Default)]
pub struct RungeKutta4;

impl OdeSolver for RungeKutta4 {
    fn step<S: OdeSystem>(&self, system: &S, state: &mut OdeState, input: &[f64], dt: f64) {
        let n = state.values.len();
        let t = state.time;
        let y = &state.values;

        // k1 = f(t, y)
        let k1 = system.derivative(t, y, input);

        // k2 = f(t + dt/2, y + dt/2 * k1)
        let y2: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(yi, k1i)| yi + 0.5 * dt * k1i)
            .collect();
        let k2 = system.derivative(t + 0.5 * dt, &y2, input);

        // k3 = f(t + dt/2, y + dt/2 * k2)
        let y3: Vec<f64> = y
            .iter()
            .zip(k2.iter())
            .map(|(yi, k2i)| yi + 0.5 * dt * k2i)
            .collect();
        let k3 = system.derivative(t + 0.5 * dt, &y3, input);

        // k4 = f(t + dt, y + dt * k3)
        let y4: Vec<f64> = y
            .iter()
            .zip(k3.iter())
            .map(|(yi, k3i)| yi + dt * k3i)
            .collect();
        let k4 = system.derivative(t + dt, &y4, input);

        // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 0..n {
            state.values[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
}

/// Dormand-Prince method (RK45) with adaptive step size.
///
/// Uses embedded 4th and 5th order methods for error estimation and automatically adjusts
/// the step size to maintain accuracy. Adaptive stepping is First-Same-As-Last (FSAL): the
/// final stage of an accepted step (`k7 = f(t + dt, y_new)`) is reused as the first stage
/// (`k1`) of the next step, so a run of accepted steps costs ~6 derivative evaluations per
/// step rather than 7.
///
/// Tolerances (`rtol`/`atol`) are stored on the struct and used by [`OdeSolver::step`] and
/// [`DormandPrince::adaptive_step`]. [`OdeSolver::integrate`] instead uses the tolerances
/// carried by the supplied [`SolverConfig`], so a config overrides the struct defaults per
/// call.
#[derive(Debug, Clone)]
pub struct DormandPrince {
    /// Safety factor for step size control
    safety: f64,
    /// Minimum step size
    min_dt: f64,
    /// Maximum step size
    max_dt: f64,
    /// Relative error tolerance used by `step`/`adaptive_step`
    rtol: f64,
    /// Absolute error tolerance used by `step`/`adaptive_step`
    atol: f64,
}

impl Default for DormandPrince {
    fn default() -> Self {
        DormandPrince {
            safety: 0.9,
            min_dt: 1e-10,
            max_dt: 1.0,
            rtol: 1e-6,
            atol: 1e-9,
        }
    }
}

impl DormandPrince {
    /// Creates a new Dormand-Prince solver with custom step-size bounds.
    ///
    /// The error tolerances default to `rtol = 1e-6`, `atol = 1e-9`; override them with
    /// [`DormandPrince::with_tolerances`].
    pub fn new(safety: f64, min_dt: f64, max_dt: f64) -> Self {
        DormandPrince {
            safety,
            min_dt,
            max_dt,
            ..Default::default()
        }
    }

    /// Sets the relative and absolute error tolerances used by [`OdeSolver::step`] and
    /// [`DormandPrince::adaptive_step`].
    ///
    /// Note that [`OdeSolver::integrate`] uses the tolerances from its [`SolverConfig`]
    /// argument instead, which lets a config override these per call.
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    /// Computes one trial Dormand-Prince step of size `dt` from `(t, y)`.
    ///
    /// `k1 = f(t, y)` is supplied by the caller so it can be reused across rejected
    /// attempts and threaded between accepted steps (FSAL). Returns the 5th-order solution,
    /// its final-stage derivative `k7 = f(t + dt, y_new)` (reusable as the next step's
    /// `k1`), and the normalized RMS error estimate (`<= 1.0` means "accept").
    #[allow(clippy::too_many_arguments)]
    fn dp_step<S: OdeSystem>(
        &self,
        system: &S,
        t: f64,
        y: &[f64],
        input: &[f64],
        dt: f64,
        k1: &[f64],
        rtol: f64,
        atol: f64,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        // Dormand-Prince coefficients
        const A21: f64 = 1.0 / 5.0;
        const A31: f64 = 3.0 / 40.0;
        const A32: f64 = 9.0 / 40.0;
        const A41: f64 = 44.0 / 45.0;
        const A42: f64 = -56.0 / 15.0;
        const A43: f64 = 32.0 / 9.0;
        const A51: f64 = 19372.0 / 6561.0;
        const A52: f64 = -25360.0 / 2187.0;
        const A53: f64 = 64448.0 / 6561.0;
        const A54: f64 = -212.0 / 729.0;
        const A61: f64 = 9017.0 / 3168.0;
        const A62: f64 = -355.0 / 33.0;
        const A63: f64 = 46732.0 / 5247.0;
        const A64: f64 = 49.0 / 176.0;
        const A65: f64 = -5103.0 / 18656.0;
        const A71: f64 = 35.0 / 384.0;
        const A73: f64 = 500.0 / 1113.0;
        const A74: f64 = 125.0 / 192.0;
        const A75: f64 = -2187.0 / 6784.0;
        const A76: f64 = 11.0 / 84.0;

        // Error coefficients (difference between 5th and 4th order)
        const E1: f64 = 71.0 / 57600.0;
        const E3: f64 = -71.0 / 16695.0;
        const E4: f64 = 71.0 / 1920.0;
        const E5: f64 = -17253.0 / 339200.0;
        const E6: f64 = 22.0 / 525.0;
        const E7: f64 = -1.0 / 40.0;

        let n = y.len();

        // k1 is provided by the caller (FSAL); compute the remaining stages.
        let y2: Vec<f64> = (0..n).map(|i| y[i] + dt * A21 * k1[i]).collect();
        let k2 = system.derivative(t + dt / 5.0, &y2, input);

        let y3: Vec<f64> = (0..n)
            .map(|i| y[i] + dt * (A31 * k1[i] + A32 * k2[i]))
            .collect();
        let k3 = system.derivative(t + 3.0 * dt / 10.0, &y3, input);

        let y4: Vec<f64> = (0..n)
            .map(|i| y[i] + dt * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
            .collect();
        let k4 = system.derivative(t + 4.0 * dt / 5.0, &y4, input);

        let y5: Vec<f64> = (0..n)
            .map(|i| y[i] + dt * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
            .collect();
        let k5 = system.derivative(t + 8.0 * dt / 9.0, &y5, input);

        let y6: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + dt * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i])
            })
            .collect();
        let k6 = system.derivative(t + dt, &y6, input);

        // 5th order solution. The 7th-stage weights equal these b-coefficients, so
        // k7 = f(t + dt, y_new) and DP is FSAL.
        let y_new: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + dt * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i])
            })
            .collect();
        let k7 = system.derivative(t + dt, &y_new, input);

        // Normalized RMS error estimate (difference between 5th and 4th order solutions).
        let err: f64 = (0..n)
            .map(|i| {
                let err_i = dt
                    * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
                let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
                (err_i / scale).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / (n as f64).sqrt();

        (y_new, k7, err)
    }

    /// Performs one step with adaptive step size control.
    ///
    /// Repeatedly shrinks the step until the estimated error is within tolerance (or the
    /// minimum step size is reached), writes the accepted solution into `state`, and
    /// returns the actual step size taken. Standalone (unlike [`OdeSolver::integrate`]) it
    /// re-evaluates `k1` each call and does not carry FSAL state between calls.
    pub fn adaptive_step<S: OdeSystem>(
        &self,
        system: &S,
        state: &mut OdeState,
        input: &[f64],
        suggested_dt: f64,
        rtol: f64,
        atol: f64,
    ) -> f64 {
        let t = state.time;
        let y = state.values.clone();
        // k1 = f(t, y) is invariant across rejected attempts (t and y do not move).
        let k1 = system.derivative(t, &y, input);
        let mut dt = suggested_dt.clamp(self.min_dt, self.max_dt);

        loop {
            let (y_new, _k7, err) = self.dp_step(system, t, &y, input, dt, &k1, rtol, atol);

            if err <= 1.0 || dt <= self.min_dt {
                state.values = y_new;
                return dt;
            }

            // Reject step and reduce dt.
            let factor = (self.safety * error_factor(err)).clamp(0.1, 0.5);
            dt = (dt * factor).max(self.min_dt);
        }
    }
}

/// Standard embedded-RK step-size factor `err^(-1/5)` (guarding `err == 0`).
fn error_factor(err: f64) -> f64 {
    if err > 0.0 {
        err.powf(-0.2)
    } else {
        f64::INFINITY
    }
}

impl OdeSolver for DormandPrince {
    fn step<S: OdeSystem>(&self, system: &S, state: &mut OdeState, input: &[f64], dt: f64) {
        // The basic (fixed-size) step interface performs a single adaptive step using the
        // solver's own tolerances.
        self.adaptive_step(system, state, input, dt, self.rtol, self.atol);
    }

    fn integrate<S: OdeSystem>(
        &self,
        system: &S,
        initial_state: Vec<f64>,
        input: &[f64],
        t0: f64,
        t1: f64,
        config: &SolverConfig,
    ) -> Result<OdeState> {
        let mut state = OdeState::with_time(initial_state, t0);
        let mut dt = config.dt;
        let mut steps = 0;
        let end_tol = endpoint_tolerance(t1);
        // FSAL: k1 for the current step, reused from the previous accepted step's k7.
        let mut cached_k1: Option<Vec<f64>> = None;

        while t1 - state.time > end_tol {
            if steps >= config.max_steps {
                return Err(max_steps_error(config.max_steps, t1, state.time));
            }

            let t = state.time;
            let y = state.values.clone();
            let remaining = t1 - t;

            // k1 = f(t, y): reuse the previous step's k7 under FSAL, else evaluate it.
            let k1 = match cached_k1.take() {
                Some(k) => k,
                None => system.derivative(t, &y, input),
            };

            // Try a step from (t, y); on rejection, shrink and retry. k1 stays valid
            // across attempts because (t, y) does not move until a step is accepted.
            let mut trial_dt = dt.min(remaining).clamp(self.min_dt, self.max_dt);
            loop {
                let (y_new, k7, err) = self.dp_step(
                    system,
                    t,
                    &y,
                    input,
                    trial_dt,
                    &k1,
                    config.rtol,
                    config.atol,
                );

                if err <= 1.0 || trial_dt <= self.min_dt {
                    // Accept the step.
                    state.values = y_new;
                    state.time = t + trial_dt;
                    // FSAL: k7 = f(t + dt, y_new) becomes the next step's k1.
                    cached_k1 = Some(k7);
                    // Error-based growth of the next suggested step size.
                    let factor = (self.safety * error_factor(err)).clamp(0.2, 5.0);
                    dt = (trial_dt * factor).clamp(self.min_dt, self.max_dt);
                    break;
                }

                // Reject: shrink and retry from the same (t, y).
                let factor = (self.safety * error_factor(err)).clamp(0.1, 0.5);
                trial_dt = (trial_dt * factor).max(self.min_dt);
            }

            steps += 1;
        }

        Ok(state)
    }
}

/// A simple test ODE system for verification.
///
/// Implements dy/dt = -y (exponential decay)
#[derive(Debug)]
pub struct ExponentialDecay {
    decay_rate: f64,
}

impl ExponentialDecay {
    pub fn new(decay_rate: f64) -> Self {
        ExponentialDecay { decay_rate }
    }
}

impl OdeSystem for ExponentialDecay {
    fn derivative(&self, _t: f64, state: &[f64], _input: &[f64]) -> Vec<f64> {
        state.iter().map(|y| -self.decay_rate * y).collect()
    }

    fn dimension(&self) -> usize {
        1
    }
}

/// Harmonic oscillator for testing (dy/dt = v, dv/dt = -omega^2 * y)
#[derive(Debug)]
pub struct HarmonicOscillator {
    omega: f64,
}

impl HarmonicOscillator {
    pub fn new(omega: f64) -> Self {
        HarmonicOscillator { omega }
    }
}

impl OdeSystem for HarmonicOscillator {
    fn derivative(&self, _t: f64, state: &[f64], _input: &[f64]) -> Vec<f64> {
        // state[0] = position, state[1] = velocity
        vec![state[1], -self.omega * self.omega * state[0]]
    }

    fn dimension(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ode_state_creation() {
        let state = OdeState::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(state.dim(), 3);
        assert_eq!(state.time, 0.0);
    }

    #[test]
    fn test_solver_config_builder() {
        let config = SolverConfig::new(0.001)
            .with_max_steps(5000)
            .with_rtol(1e-8)
            .with_atol(1e-12);

        assert_eq!(config.dt, 0.001);
        assert_eq!(config.max_steps, 5000);
        assert_eq!(config.rtol, 1e-8);
        assert_eq!(config.atol, 1e-12);
    }

    #[test]
    fn test_euler_exponential_decay() {
        let system = ExponentialDecay::new(1.0);
        let solver = EulerSolver;
        let config = SolverConfig::new(0.001);

        let state = solver
            .integrate(&system, vec![1.0], &[], 0.0, 1.0, &config)
            .unwrap();

        // y(t) = e^(-t), so y(1) ~= 0.368
        assert_relative_eq!(state.values[0], (-1.0_f64).exp(), epsilon = 0.01);
    }

    #[test]
    fn test_rk4_exponential_decay() {
        let system = ExponentialDecay::new(1.0);
        let solver = RungeKutta4;
        let config = SolverConfig::new(0.01);

        let state = solver
            .integrate(&system, vec![1.0], &[], 0.0, 1.0, &config)
            .unwrap();

        // RK4 should be more accurate
        assert_relative_eq!(state.values[0], (-1.0_f64).exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_rk4_harmonic_oscillator() {
        let system = HarmonicOscillator::new(1.0);
        let solver = RungeKutta4;
        let config = SolverConfig::new(0.01);

        // Initial: y=1, v=0 -> Solution: y=cos(t), v=-sin(t)
        let state = solver
            .integrate(
                &system,
                vec![1.0, 0.0],
                &[],
                0.0,
                std::f64::consts::PI,
                &config,
            )
            .unwrap();

        // After half period: y ~= -1, v ~= 0
        assert_relative_eq!(state.values[0], -1.0, epsilon = 1e-4);
        assert_relative_eq!(state.values[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_dormand_prince_exponential_decay() {
        let system = ExponentialDecay::new(1.0);
        let solver = DormandPrince::default();
        let config = SolverConfig::new(0.1).with_rtol(1e-8).with_atol(1e-10);

        let state = solver
            .integrate(&system, vec![1.0], &[], 0.0, 1.0, &config)
            .unwrap();

        // Adaptive solver should be very accurate
        assert_relative_eq!(state.values[0], (-1.0_f64).exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_integrate_with_trajectory() {
        let system = ExponentialDecay::new(1.0);
        let solver = RungeKutta4;
        let config = SolverConfig::new(0.1);

        let trajectory = solver
            .integrate_with_trajectory(&system, vec![1.0], &[], 0.0, 1.0, &config)
            .unwrap();

        // Should have multiple points
        assert!(trajectory.len() > 5);

        // Values should be decreasing
        for window in trajectory.windows(2) {
            assert!(window[0].values[0] > window[1].values[0]);
        }
    }

    #[test]
    fn test_euler_single_step() {
        let system = ExponentialDecay::new(1.0);
        let solver = EulerSolver;
        let mut state = OdeState::new(vec![1.0]);

        solver.step(&system, &mut state, &[], 0.1);

        // After one Euler step: y_new = y + dt * (-y) = 1 - 0.1 = 0.9
        assert_relative_eq!(state.values[0], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_rk4_single_step() {
        let system = ExponentialDecay::new(1.0);
        let solver = RungeKutta4;
        let mut state = OdeState::new(vec![1.0]);

        solver.step(&system, &mut state, &[], 0.1);

        // RK4 approximation of e^(-0.1) ~= 0.9048374
        // RK4 has O(h^5) local error, so with h=0.1, error is ~1e-6
        assert_relative_eq!(state.values[0], (-0.1_f64).exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_max_steps_limit() {
        let system = ExponentialDecay::new(0.001); // Very slow decay
        let solver = EulerSolver;
        let config = SolverConfig::new(0.0001).with_max_steps(100);

        // Integration cannot reach t1 = 1000 within 100 steps of size 1e-4, so it must
        // now report an error rather than silently returning a truncated state.
        let result = solver.integrate(&system, vec![1.0], &[], 0.0, 1000.0, &config);
        assert!(matches!(result, Err(crate::error::LnnError::OdeSolver(_))));
    }

    #[test]
    fn test_variable_input() {
        // System where derivative depends on input
        struct InputDependentSystem;

        impl OdeSystem for InputDependentSystem {
            fn derivative(&self, _t: f64, _state: &[f64], input: &[f64]) -> Vec<f64> {
                vec![input.first().copied().unwrap_or(0.0)]
            }
            fn dimension(&self) -> usize {
                1
            }
        }

        let system = InputDependentSystem;
        let solver = RungeKutta4;
        let config = SolverConfig::new(0.01);

        // With constant input=1, dy/dt=1, so y(1)=1 (starting from 0)
        let state = solver
            .integrate(&system, vec![0.0], &[1.0], 0.0, 1.0, &config)
            .unwrap();

        assert_relative_eq!(state.values[0], 1.0, epsilon = 1e-6);
    }

    /// Wraps an [`OdeSystem`] and counts every derivative evaluation via a shared cell.
    struct CountingSystem<'a, S: OdeSystem> {
        inner: S,
        count: &'a std::cell::Cell<usize>,
    }

    impl<S: OdeSystem> OdeSystem for CountingSystem<'_, S> {
        fn derivative(&self, t: f64, state: &[f64], input: &[f64]) -> Vec<f64> {
            self.count.set(self.count.get() + 1);
            self.inner.derivative(t, state, input)
        }

        fn dimension(&self) -> usize {
            self.inner.dimension()
        }
    }

    #[test]
    fn test_dormand_prince_looser_tolerance_uses_fewer_evals() {
        let solver = DormandPrince::default();

        let loose_count = std::cell::Cell::new(0);
        let loose_system = CountingSystem {
            inner: ExponentialDecay::new(1.0),
            count: &loose_count,
        };
        let loose_cfg = SolverConfig::new(0.1).with_rtol(1e-2).with_atol(1e-2);
        solver
            .integrate(&loose_system, vec![1.0], &[], 0.0, 10.0, &loose_cfg)
            .unwrap();

        let tight_count = std::cell::Cell::new(0);
        let tight_system = CountingSystem {
            inner: ExponentialDecay::new(1.0),
            count: &tight_count,
        };
        let tight_cfg = SolverConfig::new(0.1).with_rtol(1e-10).with_atol(1e-12);
        solver
            .integrate(&tight_system, vec![1.0], &[], 0.0, 10.0, &tight_cfg)
            .unwrap();

        assert!(
            loose_count.get() < tight_count.get(),
            "loose tolerance ({}) should need fewer derivative evals than tight ({})",
            loose_count.get(),
            tight_count.get()
        );
    }

    #[test]
    fn test_dormand_prince_fsal_six_evals_per_step() {
        // Pin the step size (min_dt == max_dt == 0.25) so every step is accepted on the
        // first try and the endpoint (t1 = 1.0) is hit exactly after 4 steps. With FSAL,
        // the total derivative-evaluation count is 6 per step plus one extra for the very
        // first step's k1: 7 + 6*3 = 25, rather than 7*4 = 28 without FSAL.
        let solver = DormandPrince::new(0.9, 0.25, 0.25);
        let count = std::cell::Cell::new(0);
        let system = CountingSystem {
            inner: ExponentialDecay::new(1.0),
            count: &count,
        };
        let config = SolverConfig::new(0.25);

        let state = solver
            .integrate(&system, vec![1.0], &[], 0.0, 1.0, &config)
            .unwrap();

        assert_relative_eq!(state.time, 1.0, epsilon = 1e-12);
        let n_steps = 4;
        assert_eq!(
            count.get(),
            6 * n_steps + 1,
            "FSAL should yield 6 evals/step after the first (got {})",
            count.get()
        );
    }
}
