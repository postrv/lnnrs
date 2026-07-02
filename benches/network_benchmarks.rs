// benches/network_benchmarks.rs
//! Comprehensive benchmarks for Liquid Neural Network implementations.
//!
//! Compares performance of:
//! - Classic LNN (discrete-time reservoir computing)
//! - LTC (Liquid Time-Constant with ODE integration)
//! - CfC (Closed-form Continuous - analytical solution)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liquid_neural_network::prelude::*;
use std::hint::black_box;

/// Generates a sine wave sequence for benchmarking.
fn generate_sine_sequence(length: usize) -> Vec<Vec<f64>> {
    (0..length)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t.sin(), t.cos()]
        })
        .collect()
}

/// Benchmark LNN network creation with different sizes.
fn benchmark_lnn_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("LNN Creation");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("neurons", size), size, |b, &size| {
            b.iter(|| {
                let lnn = LnnBuilder::new(size)
                    .spectral_radius(0.9)
                    .connection_probability(0.8)
                    .build();
                black_box(lnn)
            });
        });
    }
    group.finish();
}

/// Benchmark LTC network creation with different sizes.
fn benchmark_ltc_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("LTC Creation");

    for size in [16, 32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::new("units", size), size, |b, &size| {
            b.iter(|| {
                let ltc = LtcNetwork::new(2, size, 1);
                black_box(ltc)
            });
        });
    }
    group.finish();
}

/// Benchmark CfC network creation with different sizes.
fn benchmark_cfc_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("CfC Creation");

    for size in [16, 32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::new("units", size), size, |b, &size| {
            b.iter(|| {
                let cfc = CfcBuilder::new(2, 1).hidden_units(size).build();
                black_box(cfc)
            });
        });
    }
    group.finish();
}

/// Benchmark LNN simulation with different sequence lengths.
fn benchmark_lnn_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("LNN Simulation");

    for length in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*length as u64));
        group.bench_with_input(BenchmarkId::new("steps", length), length, |b, &length| {
            let mut lnn = LnnBuilder::new(50)
                .spectral_radius(0.9)
                .connection_probability(0.8)
                .build();
            lnn.set_input_neurons(vec![0, 1]);
            lnn.set_output_neurons(vec![48, 49]);
            let inputs = generate_sine_sequence(length);

            b.iter(|| {
                let outputs = lnn.run_simulation(1, black_box(inputs.clone())).unwrap();
                black_box(outputs)
            });
        });
    }
    group.finish();
}

/// Benchmark LTC forward pass with different sequence lengths.
fn benchmark_ltc_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("LTC Forward");

    for length in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*length as u64));
        group.bench_with_input(BenchmarkId::new("steps", length), length, |b, &length| {
            let ltc = LtcNetwork::new(2, 32, 1);
            let inputs = generate_sine_sequence(length);

            b.iter(|| {
                let outputs = ltc.forward_sequence(black_box(&inputs));
                black_box(outputs)
            });
        });
    }
    group.finish();
}

/// Benchmark CfC forward pass with different sequence lengths.
fn benchmark_cfc_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("CfC Forward");

    for length in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*length as u64));
        group.bench_with_input(BenchmarkId::new("steps", length), length, |b, &length| {
            let cfc = CfcBuilder::new(2, 1).hidden_units(32).build();
            let inputs = generate_sine_sequence(length);

            b.iter(|| {
                let outputs = cfc.forward_sequence(black_box(&inputs));
                black_box(outputs)
            });
        });
    }
    group.finish();
}

/// Benchmark CfC no-gate mode (faster variant).
fn benchmark_cfc_no_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("CfC No-Gate");

    for length in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*length as u64));
        group.bench_with_input(BenchmarkId::new("steps", length), length, |b, &length| {
            let cfc = CfcBuilder::new(2, 1).hidden_units(32).no_gate(true).build();
            let inputs = generate_sine_sequence(length);

            b.iter(|| {
                let outputs = cfc.forward_sequence(black_box(&inputs));
                black_box(outputs)
            });
        });
    }
    group.finish();
}

/// Compare all three models at the same configuration.
fn benchmark_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model Comparison (32 units, 100 steps)");
    let inputs = generate_sine_sequence(100);

    group.throughput(Throughput::Elements(100));

    // LNN (reservoir computing style)
    group.bench_function("LNN", |b| {
        let mut lnn = LnnBuilder::new(50)
            .spectral_radius(0.9)
            .connection_probability(0.8)
            .build();
        lnn.set_input_neurons(vec![0, 1]);
        lnn.set_output_neurons(vec![48, 49]);

        b.iter(|| {
            let outputs = lnn.run_simulation(1, black_box(inputs.clone())).unwrap();
            black_box(outputs)
        });
    });

    // LTC (ODE-based)
    group.bench_function("LTC", |b| {
        let ltc = LtcNetwork::new(2, 32, 1);

        b.iter(|| {
            let outputs = ltc.forward_sequence(black_box(&inputs));
            black_box(outputs)
        });
    });

    // CfC (closed-form)
    group.bench_function("CfC", |b| {
        let cfc = CfcBuilder::new(2, 1).hidden_units(32).build();

        b.iter(|| {
            let outputs = cfc.forward_sequence(black_box(&inputs));
            black_box(outputs)
        });
    });

    // CfC No-Gate (fastest)
    group.bench_function("CfC-NoGate", |b| {
        let cfc = CfcBuilder::new(2, 1).hidden_units(32).no_gate(true).build();

        b.iter(|| {
            let outputs = cfc.forward_sequence(black_box(&inputs));
            black_box(outputs)
        });
    });

    group.finish();
}

/// Benchmark ODE solvers.
fn benchmark_ode_solvers(c: &mut Criterion) {
    use liquid_neural_network::ode_solver::*;

    // Simple exponential decay system for benchmarking
    struct DecaySystem;
    impl OdeSystem for DecaySystem {
        fn derivative(&self, _t: f64, state: &[f64], _input: &[f64]) -> Vec<f64> {
            state.iter().map(|y| -y).collect()
        }
        fn dimension(&self) -> usize {
            1
        }
    }

    let mut group = c.benchmark_group("ODE Solvers");
    let system = DecaySystem;
    let config = SolverConfig::new(0.01);

    group.bench_function("Euler", |b| {
        let solver = EulerSolver;
        b.iter(|| {
            let result = solver
                .integrate(&system, vec![1.0], &[], 0.0, 10.0, &config)
                .unwrap();
            black_box(result)
        });
    });

    group.bench_function("RK4", |b| {
        let solver = RungeKutta4;
        b.iter(|| {
            let result = solver
                .integrate(&system, vec![1.0], &[], 0.0, 10.0, &config)
                .unwrap();
            black_box(result)
        });
    });

    group.bench_function("Dormand-Prince", |b| {
        let solver = DormandPrince::default();
        let adaptive_config = SolverConfig::new(0.1).with_rtol(1e-6).with_atol(1e-9);
        b.iter(|| {
            let result = solver
                .integrate(&system, vec![1.0], &[], 0.0, 10.0, &adaptive_config)
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark network scaling with different sizes.
fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling (CfC)");
    let inputs = generate_sine_sequence(50);

    for units in [8, 16, 32, 64, 128].iter() {
        group.bench_with_input(BenchmarkId::new("units", units), units, |b, &units| {
            let cfc = CfcBuilder::new(2, 1).hidden_units(units).build();

            b.iter(|| {
                let outputs = cfc.forward_sequence(black_box(&inputs));
                black_box(outputs)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_lnn_creation,
    benchmark_ltc_creation,
    benchmark_cfc_creation,
    benchmark_lnn_simulation,
    benchmark_ltc_forward,
    benchmark_cfc_forward,
    benchmark_cfc_no_gate,
    benchmark_model_comparison,
    benchmark_ode_solvers,
    benchmark_scaling
);
criterion_main!(benches);
