from lunar_lander.agent import Agent
from lunar_lander.utils.file_handling import load_pickle
from lunar_lander.utils.threading import MultiCoreController


def single_train(render: bool = False) -> None:
    """Train a single agent with the default hyperparameters."""
    agent = Agent(render=render)
    agent.train(
        n_episodes=10_000,
        learning_rate=0.15,
        gamma=0.99,
        min_epsilon=0.05,
        max_epsilon=1,
        epsilon_decay=0.005,
    )


def multicore_train_wrapper(
    n_episodes: int,
    learning_rate: float,
    gamma: float,
    min_epsilon: float,
    max_epsilon: float,
    epsilon_decay: float,
    render: bool = False,
) -> None:
    agent = Agent(render=render)
    agent.train(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        gamma=gamma,
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        epsilon_decay=epsilon_decay,
    )


def multicore_train(render: bool = False) -> None:
    """Run a small hyperparameter sweep across multiple processes."""
    params = [
        (
            10_000,  # n_episodes
            0.1,  # learning_rate
            0.99,  # gamma
            0.05,  # min_epsilon
            1,  # max_epsilon
            0.0001,  # epsilon_decay
        ),
        (
            10_000,  # n_episodes
            0.2,  # learning_rate
            0.99,  # gamma
            0.05,  # min_epsilon
            1,  # max_epsilon
            0.001,  # epsilon_decay
        ),
        # ...
    ]
    controller = MultiCoreController(n_cores=4)
    controller.run(function=multicore_train_wrapper, param_list=params)


def simple_run(run_id: str, render: bool = False):
    """Load a saved model and run it in the environment."""
    model = load_pickle(run=run_id)
    agent = Agent(render=render)
    agent.run(model, n_steps=1_000)
