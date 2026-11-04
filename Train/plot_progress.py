"""Ve lai do thi phat trien tu <dir>/progress.csv.   python plot_progress.py --dir results"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_ppo_tsc as T  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results")
a = p.parse_args()
T.plot_progress(os.path.join(a.dir, "progress.csv"), os.path.join(a.dir, "progress.png"))
