import gymnasium as gym


def main() -> None:
    env = gym.make("LunarLander-v2", render_mode="human")
    _, _ = env.reset()

    for _ in range(1_000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            _, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
