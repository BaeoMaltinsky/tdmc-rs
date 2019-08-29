use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::time::Instant;

pub fn get_rng() -> XorShiftRng {
    XorShiftRng::from_rng(rand::thread_rng()).unwrap()
}

struct Walkers<T: Clone + Send> {
    n_dynamics_evaluations: u64,
    walker_data: Vec<WalkerData<T>>,
}

pub trait TDMC: Sized + Send + Sync {
    type State: Clone + Send;

    fn propagate_sample(state: &mut Self::State, timestamp: i32);

    fn chi(new: &Self::State, old: &Self::State, timestamp: i32) -> f64;

    fn run_tdmc(
        initial_walker_state: Self::State,
        n_dmc_steps: i32,
        n_initial_walkers: i32,
    ) -> Vec<(Self::State, i32)> {
        println!("Beginning TDMC.");

        // Initialize clock.
        let start_clock = Instant::now();
        let mut n_dynamics_evaluations = 0u64;

        // Initialize RNG.
        let mut rng = get_rng();
        let uniform = Uniform::new_inclusive(0f64, 1f64);

        // Create a list of walkers with uniformly distributed tickets.
        let mut walker_list: Vec<Walkers<Self::State>> = (0..n_initial_walkers)
            .zip((&uniform).sample_iter(&mut rng))
            .map(|(replicate_id, ticket)| Walkers {
                n_dynamics_evaluations: 0,
                walker_data: vec![WalkerData {
                    state: initial_walker_state.clone(),
                    ticket,
                    replicate_id,
                }],
            })
            .collect();

        // Perform ticketed DMC sampling for n_dmc_steps steps.
        for i in 0..n_dmc_steps {
            // At every step, advance and adjust the copy number of each walker.
            // Iterate over all walkers starting from the first.
            walker_list.par_iter_mut().for_each(|replicates| {
                let mut rng = get_rng();

                let mut j = 0;
                while j < replicates.walker_data.len() {
                    // Calculate a new state, saving the old & new outside the
                    // list for just a moment.
                    let previous_state = replicates.walker_data[j].state.clone();
                    Self::propagate_sample(&mut replicates.walker_data[j].state, i);
                    replicates.n_dynamics_evaluations += 1;

                    // Calculate the generalized DMC weight for the proposed step.
                    let step_weight =
                        (-Self::chi(&replicates.walker_data[j].state, &previous_state, i)).exp();

                    // If the weight is lower than the walker's ticket, delete the walker
                    // and advance to the next.
                    if step_weight < replicates.walker_data[j].ticket {
                        // TODO this is unnecessary, should just be an option
                        replicates.walker_data.remove(j);
                    } else {
                        //                    dbg!(step_weight);
                        let replicate_id = replicates.walker_data[j].replicate_id;
                        let n_clones_needed =
                            1.max((step_weight + uniform.sample(&mut rng)) as i32);

                        replicates.walker_data[j].ticket /= step_weight;

                        for _ in 1..n_clones_needed {
                            let new_ticket_dist = Uniform::new_inclusive(1.0 / step_weight, 1.0);
                            let cloned_walker = WalkerData {
                                state: replicates.walker_data[j].state.clone(),
                                ticket: new_ticket_dist.sample(&mut rng),
                                replicate_id,
                            };

                            j += 1;
                            replicates.walker_data.push(cloned_walker);
                        }
                        j += 1
                    }
                }
            });
        }

        let res = walker_list
            .into_iter()
            .map(|replicates| {
                n_dynamics_evaluations += replicates.n_dynamics_evaluations;
                replicates.walker_data.into_iter().map(|walker| {
                    let WalkerData {
                        state,
                        replicate_id,
                        ..
                    } = walker;

                    (state, replicate_id)
                })
            })
            .flatten()
            .collect();
        println!(
            "Completed TDMC in {} milliseconds using {} evaluations of sampling dynamics.",
            (Instant::now() - start_clock).as_millis(),
            n_dynamics_evaluations
        );
        res
    }
}

#[derive(Clone, PartialEq)]
pub struct WalkerData<T: Clone + Send> {
    state: T,
    ticket: f64,
    replicate_id: i32,
}
