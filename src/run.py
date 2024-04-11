import argparse

from lunar_lander import configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LunarLander agent controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_single = subparsers.add_parser(
        "train-single", help="Train a single agent")
    train_single.add_argument("--render", action="store_true",
                              help="Render the environment")

    train_multi = subparsers.add_parser(
        "train-multi", help="Train multiple agents in parallel")
    train_multi.add_argument("--render", action="store_true",
                             help="Render the environment")

    run = subparsers.add_parser("run", help="Run a saved model")
    run.add_argument("--run-id", required=True, help="Model run ID to load")
    run.add_argument("--render", action="store_true",
                     help="Render the environment")

    args = parser.parse_args()

    if args.command == "train-single":
        configs.single_train(render=args.render)
    elif args.command == "train-multi":
        configs.multicore_train(render=args.render)
    elif args.command == "run":
        configs.simple_run(run_id=args.run_id, render=args.render)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
