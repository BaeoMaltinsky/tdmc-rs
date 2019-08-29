use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use std::time::Instant;

#[inline]
pub fn get_rng() -> XorShiftRng {
    XorShiftRng::from_rng(rand::thread_rng()).unwrap()
}

struct Walkers<T: Clone + Send> {
    n_dynamics_evaluations: u64,
    rng: XorShiftRng,
    data: Vec<WalkerData<T>>,
}

impl<T: Clone + Send> Walkers<T> {
    #[inline]
    fn new(state: T, ticket: f64, replicate_id: u32) -> Self {
        let mut data = Vec::with_capacity(8);
        data.push(WalkerData::new(state, ticket, replicate_id));

        Walkers {
            data,
            n_dynamics_evaluations: 0,
            rng: get_rng(),
        }
    }
}

pub trait TDMC {
    type State: Clone + Send;

    fn propagate_sample(state: &mut Self::State, timestamp: u32);

    fn chi(new: &Self::State, old: &Self::State, timestamp: u32) -> f64;

    fn run_tdmc(
        initial_walker_state: Self::State,
        n_dmc_steps: u32,
        n_initial_walkers: u32,
    ) -> Vec<(Self::State, u32)> {
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
            .map(|(replicate_id, ticket)| {
                Walkers::new(initial_walker_state.clone(), ticket, replicate_id)
            })
            .collect();

        // Perform ticketed DMC sampling for n_dmc_steps steps.
        for i in 0..n_dmc_steps {
            // At every step, advance and adjust the copy number of each walker.
            // Iterate over all walkers starting from the first.
            walker_list.par_iter_mut().for_each(|replicates| {
                let rng = &mut replicates.rng;
                let mut pending = Vec::new();

                let mut j = 0;
                while j < replicates.data.len() {
                    // Calculate a new state, saving the old & new outside the
                    // list for just a moment.
                    let previous_state = replicates.data[j].state.clone();
                    Self::propagate_sample(&mut replicates.data[j].state, i);
                    replicates.n_dynamics_evaluations += 1;

                    // Calculate the generalized DMC weight for the proposed step.
                    let step_weight =
                        (-Self::chi(&replicates.data[j].state, &previous_state, i)).exp();

                    // If the weight is lower than the walker's ticket, delete the walker
                    // and advance to the next.
                    if step_weight < replicates.data[j].ticket {
                        // TODO this is unnecessary, should just be an option
                        replicates.data.remove(j);
                    } else {
                        let replicate_id = replicates.data[j].replicate_id;
                        let n_clones_needed = 1.max((step_weight + uniform.sample(rng)) as u32);

                        replicates.data[j].ticket /= step_weight;

                        for _ in 1..n_clones_needed {
                            let new_ticket_dist = Uniform::new_inclusive(1.0 / step_weight, 1.0);
                            let cloned_walker = WalkerData {
                                state: replicates.data[j].state.clone(),
                                ticket: new_ticket_dist.sample(rng),
                                replicate_id,
                            };

                            pending.push(cloned_walker);
                        }
                    }
                    j += 1;
                }
                replicates.data.extend(pending);
            });
        }

        let res = walker_list
            .into_iter()
            .map(|replicates| {
                // aggregate evaluation counts
                n_dynamics_evaluations += replicates.n_dynamics_evaluations;

                // convert
                replicates.data.into_iter().map(|walker| walker.into_pair())
            })
            .flatten()
            .collect();

        println!(
            "Completed TDMC in {} seconds using {} evaluations of sampling dynamics.",
            (Instant::now() - start_clock).as_micros() as f64 / 1_000_000.0,
            n_dynamics_evaluations
        );
        res
    }
}

#[derive(Clone, PartialEq)]
pub struct WalkerData<T: Clone + Send> {
    state: T,
    ticket: f64,
    replicate_id: u32,
}

impl<T: Clone + Send> WalkerData<T> {
    #[inline]
    fn new(state: T, ticket: f64, replicate_id: u32) -> Self {
        WalkerData {
            state,
            ticket,
            replicate_id,
        }
    }

    #[inline]
    fn into_pair(self) -> (T, u32) {
        let WalkerData {
            state,
            replicate_id,
            ..
        } = self;
        (state, replicate_id)
    }
}
