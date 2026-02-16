"""d20 dice roller. Prints a JSON result to stdout.

Usage:
    .venv/Scripts/python.exe scripts/d20.py
    .venv/Scripts/python.exe scripts/d20.py --n 3

Output:
    {"rolls": [17], "nat1": false, "nat20": false}
"""

import argparse
import json
import random
import time


def main():
    parser = argparse.ArgumentParser(description="Roll d20(s)")
    parser.add_argument("--n", type=int, default=1, help="Number of dice")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: time-based)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(int(time.time() * 1000) ^ id(args))

    rolls = [random.randint(1, 20) for _ in range(args.n)]

    print(json.dumps({
        "rolls": rolls,
        "nat1": 1 in rolls,
        "nat20": 20 in rolls,
    }))


if __name__ == "__main__":
    main()
