use im_rc::Vector as ImVector;
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_xorshift::XorShiftRng;
use std::time::Instant;

pub trait TDMC: Sized {
    type State: Clone;

    fn propagate_sample(&mut self, state: &mut Self::State, timestamp: i32);

    fn chi(new: &Self::State, old: &Self::State, timestamp: i32) -> f64;

    fn run_tdmc(
        &mut self,
        initial_walker_state: Self::State,
        n_dmc_steps: i32,
        n_initial_walkers: i32,
    ) -> ImVector<(Self::State, i32)> {
        println!("Beginning TDMC.");

        // Initialize clock.
        let start_clock = Instant::now();
        let mut n_dynamics_evaluations = 0;

        // Initialize RNG.
        let mut rng = XorShiftRng::from_rng(rand::thread_rng()).unwrap();
        let uniform = Uniform::new_inclusive(0f64, 1f64);

        // Create a list of walkers with uniformly distributed tickets.
        let mut walker_list: ImVector<WalkerData<Self::State>> = (0..n_initial_walkers)
            .zip((&uniform).sample_iter(&mut rng))
            .map(|(replicate_id, ticket)| WalkerData {
                state: initial_walker_state.clone(),
                ticket,
                replicate_id,
            })
            .collect();

        // Perform ticketed DMC sampling for n_dmc_steps steps.
        for i in 0..n_dmc_steps {
            // At every step, advance and adjust the copy number of each walker.
            // Iterate over all walkers starting from the first.
            let mut j = 0;
            while j < walker_list.len() {
                // Calculate a new state, saving the old & new outside the
                // list for just a moment.
                let previous_state = walker_list[j].state.clone();
                self.propagate_sample(&mut walker_list[j].state, i);
                n_dynamics_evaluations += 1;

                // Calculate the generalized DMC weight for the proposed step.
                let step_weight = (-Self::chi(&walker_list[j].state, &previous_state, i)).exp();

                // If the weight is lower than the walker's ticket, delete the walker
                // and advance to the next.
                if step_weight < walker_list[j].ticket {
                    // TODO this is unnecessary, should just be an option
                    walker_list.remove(j);
                } else {
                    //                    dbg!(step_weight);
                    let replicate_id = walker_list[j].replicate_id;
                    let n_clones_needed = 1.max((step_weight + uniform.sample(&mut rng)) as i32);

                    walker_list[j].ticket /= step_weight;

                    for _ in 1..n_clones_needed {
                        let new_ticket_dist = Uniform::new_inclusive(1.0 / step_weight, 1.0);
                        let cloned_walker = WalkerData {
                            state: walker_list[j].state.clone(),
                            ticket: new_ticket_dist.sample(&mut rng),
                            replicate_id,
                        };

                        j += 1;
                        walker_list.insert(j, cloned_walker);
                    }
                    j += 1
                }
            }
        }

        println!(
            "Completed TDMC in {} milliseconds using {} evaluations of sampling dynamics.",
            (Instant::now() - start_clock).as_millis(),
            n_dynamics_evaluations
        );
        walker_list
            .into_iter()
            .map(|walker| {
                let WalkerData {
                    state,
                    replicate_id,
                    ..
                } = walker;

                (state, replicate_id)
            })
            .collect()
    }
}

#[derive(Clone, PartialEq)]
pub struct WalkerData<T: Clone> {
    state: T,
    ticket: f64,
    replicate_id: i32,
}
