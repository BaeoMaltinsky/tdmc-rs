use lazy_static::*;
use ord_subset::*;
use rand::distributions::Distribution;
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use std::ops::AddAssign;
use tdmc::*;

const PAPER_GAMMA: f64 = 0.4;
const PAPER_LAMBDA: f64 = 1.9;
const PAPER_EPSILON: f64 = 0.001;
const PAPER_NSTEPS: u32 = (2.0 / PAPER_EPSILON) as u32;
// Choose a number of replicates to perform.
const N_RUNS: u32 = 10_000;

// Define the number of particles and problem dimension.
// This code has been written so as to make adjusting
// this example to study the analogous rare event in
// an LJ 13 cluster in 3D a natural exercise.
const N_PARTICLES: usize = 7;
const DIMENSION: usize = 2;
const N_VARS: usize = N_PARTICLES * DIMENSION;

#[derive(Default, Clone, Copy)]
struct LjStateVec([f64; N_VARS]);

impl LjStateVec {
    fn new() -> LjStateVec {
        [0f64; N_VARS].into()
    }
}

impl std::ops::Deref for LjStateVec {
    type Target = [f64; N_VARS];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for LjStateVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<[f64; N_VARS]> for LjStateVec {
    #[inline]
    fn from(arr: [f64; N_VARS]) -> Self {
        Self(arr)
    }
}

impl std::fmt::Display for LjStateVec {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        for real in self.iter() {
            write!(fmt, "{}", real)?;
            write!(fmt, " ")?;
        }
        Ok(())
    }
}

struct LennardJones {}

impl TDMC for LennardJones {
    type State = LjStateVec;

    #[inline]
    fn propagate_sample(state: &mut Self::State, _: u32, rng: &mut XorShiftRng) {
        propagate_state(state, rng);
    }

    #[inline]
    fn chi(new: &Self::State, old: &Self::State, _: u32) -> f64 {
        dmc_chi(new, old)
    }
}

struct LennardJonesBF {}

impl TDMC for LennardJonesBF {
    type State = LjStateVec;

    #[inline]
    fn propagate_sample(state: &mut Self::State, _: u32, rng: &mut XorShiftRng) {
        propagate_state(state, rng);
    }

    #[inline]
    fn chi(_: &Self::State, _: &Self::State, _: u32) -> f64 {
        0.0
    }
}

/// Run the 2D LJ7 example using TDMC.
fn main() {
    println!("Reproducing the Hairer and Weare seven particle LJ cluster example.");
    println!(
        "using parameters γ = {}, λ = {}, and and ε = {}.",
        PAPER_GAMMA, PAPER_LAMBDA, PAPER_EPSILON
    );

    // Initialize 7 particles, 6 in a hexagon around
    // a single central particle.
    let mut initial_state: LjStateVec = [
        0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -0.5, 0.866_025, 0.5, 0.866_025, -0.5, -0.866_025, 0.5,
        -0.866_025,
    ]
    .into();

    // Scale so that the particles are close to the
    // sixfold symmetric energy minimum.
    initial_state.iter_mut().for_each(|state| {
        *state *= 1.11846;
    });

    // Print the initial state and the initial LJ forces on the particles.
    println!("Initial state: {}", initial_state);
    let force = calc_lj_force(&initial_state);
    println!("Initial forces: {}", &force);

    // Run TDMC and brute force simulation `N_RUNS` times each.
    let end_walker_data = LennardJones::run_tdmc(initial_state, PAPER_NSTEPS, N_RUNS);

    let end_bf_walker_data = LennardJonesBF::run_tdmc(initial_state, PAPER_NSTEPS, N_RUNS);

    // Calculate the estimators for TDMC.
    // Calculate means and variances of indicators
    // of existence and occupation of regions B and D.
    // Region B: a particle other than the initial central
    // particle is within .1σ of the cluster center.
    // Region D: a particle other than the initial central
    // particle is closest to the cluster center.
    // Region D contains Region B for all practical purposes.
    let mut i_b_estimator_sum: f64 = 0.0;
    let mut i_d_estimator_sum: f64 = 0.0;
    let mut one_estimator_sum: f64 = 0.0;
    let mut i_b_estimator_sumsq: f64 = 0.0;
    let mut i_d_estimator_sumsq: f64 = 0.0;
    let mut one_estimator_sumsq: f64 = 0.0;

    let mut curr_replicate: u32 = 0;
    let mut i_b_estimator: f64 = 0.0;
    let mut i_d_estimator: f64 = 0.0;
    let mut one_estimator: f64 = 0.0;

    for walker_datum in end_walker_data {
        // If we've run out of walkers for the current replicate,
        // gather the statistics for that replicate and move on
        // to the next.
        if walker_datum.1 != curr_replicate {
            i_b_estimator_sum += i_b_estimator;
            i_d_estimator_sum += i_d_estimator;
            one_estimator_sum += one_estimator;
            i_b_estimator_sumsq += i_b_estimator * i_b_estimator;
            i_d_estimator_sumsq += i_d_estimator * i_d_estimator;
            one_estimator_sumsq += one_estimator * one_estimator;
            curr_replicate = walker_datum.1;
            i_b_estimator = 0.0;
            i_d_estimator = 0.0;
            one_estimator = 0.0;
        }

        // Calculate the expectation of indicator functions by
        // weighting the walker by its final state bias as described
        // in the paper, then multiplying by the indicator of interest.
        // Sum over all the walkers for each replicate.
        let weight = dmc_chi(&walker_datum.0, &initial_state).exp();
        one_estimator += weight;

        if rearrangement_coord(&walker_datum.0) < 0.1 {
            i_b_estimator += weight;
        }

        if central_particle(&walker_datum.0) != 0 {
            i_d_estimator += weight;
        }
    }

    // Calculate the estimators for brute force simulation.
    // The expectation for existence is one in this case, and
    // an indicator value is equal to its square so don't bother
    // gathering both separately.
    let mut bf_i_b_estimator_sum: f64 = 0.0;
    let mut bf_i_d_estimator_sum: f64 = 0.0;
    for bf_walker_datum in end_bf_walker_data {
        // Calculate the indicators.
        if rearrangement_coord(&bf_walker_datum.0) < 0.1 {
            bf_i_b_estimator_sum += 1.0;
        }
        if central_particle(&bf_walker_datum.0) != 0 {
            bf_i_d_estimator_sum += 1.0;
        }
    }

    let n_runs = f64::from(N_RUNS);
    // Print the results
    println!(
        "Final estimate for E[1]: {}, variance {}",
        one_estimator_sum / n_runs,
        (one_estimator_sumsq / n_runs
            - (one_estimator_sum / n_runs) * (one_estimator_sum / n_runs))
    );

    println!(
        "Final estimate for E[I_B]: {}, variance {}",
        i_b_estimator_sum / n_runs,
        (i_b_estimator_sumsq / n_runs
            - (i_b_estimator_sum / n_runs) * (i_b_estimator_sum / n_runs))
    );

    println!(
        "Final estimate for E[bf_I_B]: {}, variance {}",
        bf_i_b_estimator_sum / n_runs,
        (bf_i_b_estimator_sum / n_runs
            - (bf_i_b_estimator_sum / n_runs) * (bf_i_b_estimator_sum / n_runs))
    );

    println!(
        "Final estimate for E[I_D]: {}, variance {}",
        i_d_estimator_sum / n_runs,
        (i_d_estimator_sumsq / n_runs
            - (i_d_estimator_sum / n_runs) * (i_d_estimator_sum / n_runs))
    );

    println!(
        "Final estimate for E[bf_I_D]: {}, variance {}",
        bf_i_d_estimator_sum / n_runs,
        (bf_i_d_estimator_sum / n_runs
            - (bf_i_d_estimator_sum / n_runs) * (bf_i_d_estimator_sum / n_runs))
    );
}

#[inline]
fn dmc_chi(x_new: &LjStateVec, x_old: &LjStateVec) -> f64 {
    paper_v(x_new) - paper_v(x_old)
}

#[inline]
fn propagate_state(x: &mut LjStateVec, rng: &mut XorShiftRng) {
    let conservative_force = calc_lj_force(x);
    let random_displacement = calc_random_displacement(rng);

    x.iter_mut()
        .zip(conservative_force.iter())
        .zip(random_displacement.iter())
        .for_each(|((state, conv), random)| {
            state.add_assign(conv * PAPER_EPSILON + random);
        });
}

/// Calculate the total forces experienced by each particle
/// in each direction as a sum of LJ pair forces (in LJ units).
#[inline]
fn calc_lj_force(x: &LjStateVec) -> LjStateVec {
    // Start from having zero forces.
    let mut force = LjStateVec::new();

    // Calculate the pair forces.
    particle_range().for_each(|i| {
        ((i + 1)..N_PARTICLES).for_each(|j| {
            let mut disp = [0f64; DIMENSION];

            // Calculate the pair displacement and squared distance.
            let mut dist_sq = 0.0;
            dimension_range().for_each(|k| {
                disp[k] = x[DIMENSION * i + k] - x[DIMENSION * j + k];
                dist_sq += disp[k].powi(2);
            });

            // Use the displacement and distance to calculate forces.
            dimension_range().for_each(|k| {
                force[DIMENSION * i + k] +=
                    (48.0 * dist_sq.powi(-7) - 24.0 * dist_sq.powi(-4)) * disp[k];
                force[DIMENSION * j + k] +=
                    (48.0 * dist_sq.powi(-7) - 24.0 * dist_sq.powi(-4)) * -disp[k];
            });
        })
    });
    force
}

/// Calculate a random displacement given the timestep
/// and temperature.
// TODO Why is there no timestamp dependence?
#[inline]
fn calc_random_displacement(rng: &mut XorShiftRng) -> LjStateVec {
    // Seed the RNG and set up a Gaussian number generator
    // prior to the first call of this function.
    // Static variables are used in lieu of a functor class
    // to make the code a little less indirect for educational
    // purposes.
    lazy_static! {
        static ref BROWNIAN_INCREMENT: Normal<f64> =
            Normal::new(0.0, (2.0 * PAPER_GAMMA * PAPER_EPSILON).sqrt()).unwrap();
    }

    // Start from zero.
    let mut disp = LjStateVec::new();

    for (x, inc) in disp.iter_mut().zip(BROWNIAN_INCREMENT.sample_iter(rng)) {
        x.add_assign(inc);
    }
    disp
}

/// Calculate the sampling-biasing function defined for this
/// system.
#[inline]
fn paper_v(x: &LjStateVec) -> f64 {
    (PAPER_LAMBDA / PAPER_GAMMA) * rearrangement_coord(x)
}

/// Calculate the minimum of the distances between the cluster
/// center of mass and all of the particles that did not begin
/// as the central particle of the cluster.
#[inline]
fn rearrangement_coord(x: &LjStateVec) -> f64 {
    let mut center_position = [0f64; DIMENSION];

    particle_range().for_each(|i| {
        dimension_range().for_each(|j| {
            center_position[j] += x[DIMENSION * i + j];
        })
    });

    dimension_range().for_each(|i| {
        center_position[i] /= N_PARTICLES as f64;
    });

    // Find the minimum squared distance from one of the
    // initially peripheral particles to the cluster center.
    // This differs from the paper only in that here the
    // particles are indexed from 0 rather than from 1.
    // TODO why does this comment seem to contradict the code?
    (1..N_PARTICLES)
        .map(|i| {
            dimension_range()
                .map(|j| (x[DIMENSION * i + j] - center_position[j]).powi(2))
                .sum()
        })
        .ord_subset_min()
        .unwrap_or(0.0)
        .sqrt()
}

/// Center a cluster state vector.
#[inline]
fn center_cluster(x: &mut LjStateVec) {
    // Calculate the center position of the cluster.
    let mut center_position = [0f64; DIMENSION];

    particle_range().for_each(|i| {
        dimension_range().for_each(|j| {
            center_position[j] += x[DIMENSION * i + j];
        })
    });

    dimension_range().for_each(|i| {
        center_position[i] /= N_PARTICLES as f64;
    });

    // Subtract that from every particle's position.
    particle_range().for_each(|i| {
        dimension_range().for_each(|j| {
            x[DIMENSION * i + j] -= center_position[j];
        })
    });
}

/// Return the index of the particle closest to a cluster center.
#[inline]
fn central_particle(x: &LjStateVec) -> usize {
    // Make a local copy.
    let mut y = *x;
    // Center the copied cluster.
    center_cluster(&mut y);

    particle_range()
        .map(|i| {
            // Calculate the squared distance from zero.
            let dist_sq: f64 = dimension_range()
                .map(|j| y[DIMENSION * i + j].powi(2))
                .sum();
            (dist_sq, i)
        })
        .ord_subset_min()
        .map(|(_, i)| i)
        .unwrap_or(0)
}

#[inline]
fn particle_range() -> std::ops::Range<usize> {
    0..N_PARTICLES
}

#[inline]
fn dimension_range() -> std::ops::Range<usize> {
    0..DIMENSION
}
