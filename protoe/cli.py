import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protoe.process import run


def main():
    p = argparse.ArgumentParser(description="Protoe beat slice & stretch")
    p.add_argument("input", help="input audio file")
    p.add_argument("output", help="output wav path")
    p.add_argument("--mode", choices=["fill", "fill2a", "fill2b", "anchor", "shuffle", "reverse4"], default="fill")
    p.add_argument("--bpm", type=float, default=None)
    p.add_argument("--offset", type=float, default=None)
    p.add_argument("--perm", type=str, default="0,1,2,3", help="comma-separated permutation of 0..3 for shuffle mode")
    p.add_argument("--engine", choices=["formant", "paul", "paulx", "granular"], default="formant")
    args = p.parse_args()
    perm = tuple(int(x) for x in args.perm.split(","))
    bpm, offset, dur = run(args.input, args.output, mode=args.mode, bpm=args.bpm, offset=args.offset, perm=perm, engine=args.engine)
    print(f"bpm={bpm} offset={offset} duration={dur:.2f}s -> {args.output}")


if __name__ == "__main__":
    main()
