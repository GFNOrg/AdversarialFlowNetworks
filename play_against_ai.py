import argparse
import torch

from src.eval import play_game, HumanAgent


def main(args) -> None:
    ckpt = torch.load(args.ckpt_path)
    model = ckpt["model"]
    env = ckpt["env"]

    if args.human_first:
        play_game(env, HumanAgent(), model, verbose=True)
    else:
        play_game(env, model, HumanAgent(), verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ckpt_path", type=str, help="Path to the checkpoint (.pt) file."
    )
    parser.add_argument(
        "--human-first",
        action="store_true",
        help="Whether the Human plays first.",
        default=False,
    )

    args = parser.parse_args()

    main(args)
