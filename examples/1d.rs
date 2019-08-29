use lazy_static::*;
use rand::distributions::Distribution;
use rand_distr::Normal;
use tdmc::*;

const TIMESTEP: f64 = 0.0001;
const BETA: f64 = -1.25;

fn dmc_chi(x_new: f64, x_old: f64) -> f64 {
    return BETA * (x_new - x_old);
}

struct Drift {}

impl TDMC for Drift {
    type State = f64;

    #[inline]
    fn propagate_sample(x: &mut Self::State, _: u32) {
        *x += 0.1;
    }

    #[inline]
    fn chi(x_new: &Self::State, x_old: &Self::State, _: u32) -> f64 {
        dmc_chi(*x_new, *x_old)
    }
}

struct Brownian {}

impl TDMC for Brownian {
    type State = f64;

    #[inline]
    fn propagate_sample(state: &mut Self::State, _: u32) {
        lazy_static! {
            static ref BROWNIAN_INCREMENT: Normal<f64> =
                Normal::new(0.0, (2.0 * TIMESTEP).sqrt()).unwrap();
        }
        *state += (*BROWNIAN_INCREMENT).sample(&mut rand::thread_rng());
    }

    #[inline]
    fn chi(state_new: &Self::State, state_old: &Self::State, _: u32) -> f64 {
        dmc_chi(*state_new, *state_old)
    }
}

fn main() {
    let end_walker_data = Drift::run_tdmc(0.0, 1, 5);
    for walker_data in end_walker_data {
        print!("{} ", walker_data.0);
    }
    println!();

    let n_replicates = 10_000;
    let n_timesteps = (1. / TIMESTEP) as u32;

    let end_walker_data = Brownian::run_tdmc(0.0, n_timesteps, n_replicates);
    println!("Final walker number: {}", end_walker_data.len());
    println!(
        "Average final walkers per initial walker: {}",
        end_walker_data.len() as f64 / n_replicates as f64
    );
    //for walker_data in end_walker_data {
    //    print!("{} ", walker_data.0);
    //}
    println!();
}
