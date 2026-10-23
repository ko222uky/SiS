"""Command line: python -m sis {score,import,export,status} ..."""

import argparse

from sis import submission
from sis.score import N_MAX, grid_score, puzzle_score
from sis.store import DEFAULT_ROOT, SolutionStore
from sis.validate import problems


def cmd_score(args):
    sols = submission.read(args.file)
    invalid = {n: p for n, params in sols.items() if (p := problems(n, params))}
    for n, p in sorted(invalid.items()):
        print(f"n={n} invalid (scored as grid): {'; '.join(p)}")
    per_n = {
        n: puzzle_score(n, sols[n]) if n in sols and n not in invalid else grid_score(n)
        for n in range(1, N_MAX + 1)
    }
    gains = sorted(((grid_score(n) - s, n) for n, s in per_n.items()), reverse=True)
    print(f"puzzles in file: {len(sols)}, invalid: {len(invalid)}")
    print(f"total score: {sum(per_n.values()):.9f}  (grid total: "
          f"{sum(grid_score(n) for n in per_n):.9f})")
    if gains[0][0] > 0:
        print("largest gains over grid:", ", ".join(f"n={n} ({g:.5f})" for g, n in gains[:10]))


def cmd_import(args):
    store = SolutionStore(args.store)
    improved = [n for n, params in submission.read(args.file).items() if store.offer(n, params)]
    print(f"improved {len(improved)} puzzles" + (f": {improved[:20]}" if improved else ""))


def cmd_export(args):
    SolutionStore(args.store).export(args.file)
    print(f"wrote {args.file}")


def cmd_status(args):
    df = SolutionStore(args.store).summary()
    print(f"store total: {df['score'].sum():.9f}  grid total: {df['grid_score'].sum():.9f}  "
          f"improved: {(df['gain'] > 0).sum()}")
    print("worst remaining puzzles (highest score):")
    print(df.nlargest(args.top, "score").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(prog="sis")
    parser.add_argument("--store", default=DEFAULT_ROOT, help="solution store directory")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("score", help="validate and score a submission CSV")
    p.add_argument("file")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("import", help="merge a submission CSV into the store")
    p.add_argument("file")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("export", help="write a submission CSV from the store")
    p.add_argument("file")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("status", help="summarise the store")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
