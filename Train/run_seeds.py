"""Chay nhieu seed song song roi gop ket qua.
  python run_seeds.py                          # 3 seed NGAU NHIEN (in ra de ban ghi lai tai lap)
  python run_seeds.py --seeds 0 1 2            # seed chi dinh
  python run_seeds.py -n 5 --jobs 2 -- --episodes 60 --eval-seeds 5     # sau '--' la tham so chuyen cho train_ppo_tsc.py
Moi seed -> thu muc runs/seed_<s>. Xong tu goi merge_seeds.py."""
import argparse
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="test2.sumocfg")
    ap.add_argument("--seeds", type=int, nargs="+", help="danh sach seed (mac dinh: ngau nhien)")
    ap.add_argument("-n", type=int, default=3, help="so seed ngau nhien neu khong co --seeds")
    ap.add_argument("--jobs", type=int, default=3, help="so lan train chay cung luc")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args(argv)
    seeds = a.seeds or random.sample(range(1, 10000), a.n)
    print("SEEDS:", seeds, "(ghi lai de tai lap)")
    os.makedirs(a.root, exist_ok=True)
    pending, running, dirs = list(seeds), [], []
    while pending or running:
        while pending and len(running) < a.jobs:
            s = pending.pop(0)
            d = os.path.join(a.root, f"seed_{s}")
            dirs.append(d)
            log = open(os.path.join(a.root, f"seed_{s}.log"), "w", encoding="utf-8")
            cmd = [sys.executable, os.path.join(HERE, "train_ppo_tsc.py"), "--cfg", a.cfg, "--seed", str(s), "--out-dir", d] + extra
            running.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"  bat dau seed {s} -> {d}  (log: {a.root}/seed_{s}.log)")
        for item in running[:]:
            if item[1].poll() is not None:
                item[2].close(); running.remove(item)
                print(f"  xong seed {item[0]} (ma thoat {item[1].returncode})")
        time.sleep(5)
    subprocess.run([sys.executable, os.path.join(HERE, "merge_seeds.py")] + dirs + ["--out", os.path.join(a.root, "merged_results.csv")])


if __name__ == "__main__":
    main()
