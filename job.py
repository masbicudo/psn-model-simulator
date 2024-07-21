#!./.venv/bin/python
import os
import sys
import argparse

from libs.psn_model import do_experiment_simulation__variance_grows_with_window

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Job Sample')
    parser.add_argument('--md', type=int)
    parser.add_argument('--sdp', type=float)
    parser.add_argument('-r', type=float)
    parser.add_argument('-b', type=float)
    parser.add_argument('--rng_seed', type=int)
    parser.add_argument('--experiment_count', type=int, default=100)
    parser.add_argument('--time_count', type=int, default=150)
    args = parser.parse_args()
    print(args)

    os.makedirs("./result", exist_ok=True)

    args = (args.md, args.sdp, args.r, args.b,
            args.experiment_count, args.time_count,
            args.rng_seed)
    do_experiment_simulation__variance_grows_with_window(args)
