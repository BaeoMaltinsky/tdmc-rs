use lazy_static::*;
use rand::{distributions::Distribution, SeedableRng};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use tdmc::*;

const TIMESTEP: f64 = 0.0001;
const BETA: f64 = -1.25;

fn dmc_chi(x_new: f64, x_old: f64) -> f64 {
    return BETA * (x_new - x_old);
}

struct Drift {}

impl TDMC for Drift {
    type State = f64;

    fn propagate_sample(&mut self, x: &mut Self::State, _: i32) {
        *x += 0.1;
    }

    fn chi(x_new: &Self::State, x_old: &Self::State, _: i32) -> f64 {
        dmc_chi(*x_new, *x_old)
    }
}

struct Brownian {
    rng: XorShiftRng,
}

impl Brownian {
    fn new() -> Self {
        Brownian {
            rng: XorShiftRng::from_rng(rand::thread_rng()).unwrap(),
        }
    }
}

impl TDMC for Brownian {
    type State = f64;

    fn propagate_sample(&mut self, x: &mut Self::State, _: i32) {
        lazy_static! {
            static ref BROWNIAN_INCREMENT: Normal<f64> =
                Normal::new(0.0, (2.0 * TIMESTEP).sqrt()).unwrap();
        }
        *x += (*BROWNIAN_INCREMENT).sample(&mut self.rng);
    }

    fn chi(x_new: &Self::State, x_old: &Self::State, _: i32) -> f64 {
        dmc_chi(*x_new, *x_old)
    }
}

fn main() {
    let mut drift = Drift {};
    let end_walker_data = drift.run_tdmc(0.0, 1, 5);
    for walker_data in end_walker_data {
        print!("{} ", walker_data.0);
    }
    println!();

    let n_replicates = 1000;
    let n_timesteps = (1. / TIMESTEP) as i32;

    let mut brownian = Brownian::new();
    let end_walker_data = brownian.run_tdmc(0.0, n_timesteps, n_replicates);
    println!("Final walker number: {}", end_walker_data.len());
    println!(
        "Average final walkers per initial walker: {}",
        end_walker_data.len() as i32 / n_replicates
    );
    //for walker_data in end_walker_data {
    //    print!("{} ", walker_data.0);
    //}
    println!();
}
