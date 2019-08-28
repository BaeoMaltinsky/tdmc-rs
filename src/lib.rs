use im_rc::Vector as ImVector;
use rand::distributions::{Distribution, Uniform};

pub trait TDMC: Sized {
    type State: Clone;

    fn propagate_sample(state: &mut Self::State, timestamp: i32);

    fn chi(x: &Self::State, y: &Self::State, timestamp: i32) -> f64;

    fn run_tdmc(
        initial_walker_state: Self::State,
        n_dmc_steps: i32,
        n_initial_walkers: i32,
    ) -> ImVector<(Self::State, i32)> {
        println!("Beginning TDMC.");

        let mut n_dynamics_evaluations = 0;

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new_inclusive(0f64, 1f64);

        let mut walker_list: ImVector<WalkerData<Self::State>> = (0..n_initial_walkers)
            .zip((&uniform).sample_iter(&mut rng))
            .map(|(replicate_id, ticket)| WalkerData {
                state: initial_walker_state.clone(),
                ticket,
                replicate_id,
            })
            .collect();

        for i in 0..n_dmc_steps {
            let mut j = 0;
            while j < walker_list.len() {
                let current_state = &mut walker_list[j].state;
                let previous_state = current_state.clone();

                Self::propagate_sample(current_state, i);
                n_dynamics_evaluations += 1;

                let step_weight = -Self::chi(&current_state, &previous_state, i).exp();

                if step_weight < walker_list[j].ticket {
                    // TODO this is unnecessary, should just be an option
                    walker_list.remove(0);
                } else {
                    let replicate_id = walker_list[j].replicate_id;
                    let n_clones_needed = 1.max((step_weight + uniform.sample(&mut rng)) as i32);

                    walker_list[j].ticket /= step_weight;

                    let new_ticket_dist = Uniform::new_inclusive(1.0 / step_weight, 1.0);

                    for _ in 0..n_clones_needed {
                        let cloned_walker = WalkerData {
                            state: walker_list[j].state.clone(),
                            ticket: new_ticket_dist.sample(&mut rng),
                            replicate_id,
                        };

                        j += 1;
                        walker_list.insert(j, cloned_walker);
                    }
                }
            }
        }

        println!("used {}", n_dynamics_evaluations);
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
