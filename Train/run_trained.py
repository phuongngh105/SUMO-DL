"""
Chay model da train (best.pt / last.pt): 1 bo trong so DUNG CHUNG, nhung MOI DEN GIAO THONG TU QUYET DINH
DOC LAP tu quan sat cua rieng no (lan vao, hang cho, thoi gian da o pha hien tai).
  - thoi gian xanh/vang/do cua 2 den khong bi rang buoc, khong can giong nhau, khong can dong bo
  - chi co CHU KY QUYET DINH (delta_t giay) la chung; moi den van tu chon giu pha hay doi pha

Co the chay weight nay tren BAT KY mang nao (them/bot nga tu, den, lan, pha): them --cfg mang_moi.sumocfg

Cach dung (chay trong thu muc chua train_ppo_tsc.py va thu muc ket qua):
    python run_trained.py --dir results                    # best.pt, chay nhanh, in bao cao
    python run_trained.py --dir results --weights last --gui
    python run_trained.py --dir results --weights both     # so best va last
Ket qua: in bang so sanh voi den mac dinh + phan bo pha tung den; ghi <dir>/phase_log_<ten>.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_ppo_tsc as T  # noqa: E402


def to_green(p, tl, env):
    """Pha vang -> pha xanh di truoc no (de so sanh cong bang voi den mac dinh)."""
    if p in env.greens[tl]:
        return p
    for g, y in env.yellow_of[tl].items():
        if y is not None and int(y) == p:
            return int(g)
    return p


def segments(trace, env, dt):
    """trace [(t, tl, pha)] (moi muc = pha dang chay trong cua so [t-dt, t]) -> {tl: [(bat_dau, pha, thoi_luong)]}."""
    out = {tl: [] for tl in env.tl_ids}
    last, start = {}, {}
    end_t = max(t for t, _, _ in trace)
    for t, tl, p in trace:
        p = to_green(p, tl, env)
        if tl not in last:
            last[tl], start[tl] = p, t - dt
        elif p != last[tl]:
            out[tl].append((start[tl], last[tl], (t - dt) - start[tl]))
            last[tl], start[tl] = p, t - dt
    for tl in out:
        out[tl].append((start[tl], last[tl], end_t - start[tl]))
    return out


def report(name, trace, env, dt, out_dir):
    seg = segments(trace, env, dt)
    tls = env.tl_ids
    print(f"\n--- Phan bo pha tung den ({name}) ---")
    for tl in tls:
        durs = {}
        for _, p, d in seg[tl]:
            durs.setdefault(p, []).append(d)
        parts = [f"pha {p}: TB {np.mean(v):5.1f}s x{len(v)} lan (min {min(v):.0f}, max {max(v):.0f})"
                 for p, v in sorted(durs.items())]
        print(f"  {tl}: {len(seg[tl])} doan | " + " | ".join(parts))
    for tl in tls[:6]:
        print(f"  8 doan dau {tl}: " + " -> ".join(f"{p}({d:.0f}s)" for _, p, d in seg[tl][:8]))
    if len(tls) >= 2:
        by_t = {}
        for t, tl, p in trace:
            by_t.setdefault(t, {})[tl] = to_green(p, tl, env)
        same = np.mean([len(set(v.values())) == 1 for v in by_t.values() if len(v) == len(tls)])
        print(f"  ti le thoi gian {len(tls)} den CUNG chi so pha: {100 * same:.0f}% (chi mang tinh tham khao)")
    path = os.path.join(out_dir, f"phase_log_{name.replace(' ', '_')}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["den", "bat_dau_s", "pha_xanh", "thoi_luong_s"])
        for tl in tls:
            for s0, p, d in seg[tl]:
                w.writerow([tl, round(s0, 1), p, round(d, 1)])
    print(f"  da ghi {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results", help="thu muc ket qua cua train_ppo_tsc.py (co meta.json, weight)")
    p.add_argument("--weights", default="best", help="best | last | both | duong dan .pt (weight cua BAT KY mang nao)")
    p.add_argument("--cfg", default=None, help="chay tren .sumocfg KHAC (mang khac, so den khac). Mac dinh: trained.sumocfg trong --dir")
    p.add_argument("--net", default=None)
    p.add_argument("--tls", nargs="+", default=None, help="(tuy chon) chi dieu khien cac den nay")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--seed", type=int, default=T.EVAL_SEED_BASE)
    p.add_argument("--episode-len", type=int, default=None, help="mac dinh lay tu meta.json (hoac 3600)")
    p.add_argument("--out", default=None, help="thu muc ghi phase_log (mac dinh --dir)")
    p.add_argument("--sample", action="store_true", help="chon hanh dong ngau nhien thay vi greedy")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--random-demand", action="store_true",
                   help="mang moi chua co file route: tu sinh luu luong ngau nhien (xem gen_demand.py)")
    p.add_argument("--rate", type=float, default=300.0, help="xe/gio moi canh nguon khi dung --random-demand")
    a = p.parse_args()

    mpath = os.path.join(a.dir, "meta.json")
    meta = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {}
    cfg = a.cfg or os.path.join(a.dir, meta.get("sumocfg", "trained.sumocfg"))
    stuck = meta.get("stuck", {})
    out_dir = a.out or a.dir
    os.makedirs(out_dir, exist_ok=True)
    ep_len = a.episode_len or meta.get("episode_len", 3600)
    dt = meta.get("delta_t", 10)
    env = T.MultiJunctionEnv(
        cfg, a.tls, None, None, gui=a.gui,
        delta_t=dt, yellow=meta.get("yellow", 3), episode_len=ep_len,
        seed=a.seed, min_green=meta.get("min_green", 5), max_green=meta.get("max_green", 60),
        wt_memory=meta.get("wt_memory", 300), net_file=a.net,
        demand=dict(slice_len=600, rate=(a.rate, a.rate), level=(1.0, 1.0), weights=None) if a.random_demand else None,
        w4=stuck.get("w4", 2.0), stuck_wait=stuck.get("stuck_wait", 60.0),
        reward_clip=stuck.get("reward_clip", 10.0), stuck_cap=stuck.get("stuck_cap", 30.0))

    nets = {}
    names = ["best", "last"] if a.weights == "both" else [a.weights]
    for n in names:
        path = os.path.join(a.dir, meta["weights"][n]) if n in ("best", "last") else n
        net = T.ActorCritic()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        nets["PPO " + os.path.basename(n)] = net

    results, traces = {}, {}
    if not a.no_baseline:
        tr = []
        results["BASELINE"] = T.run_episode(env, None, fixed=True, seed=a.seed, trace=tr)[0]
        traces["BASELINE"] = tr
    for name, net in nets.items():
        tr = []
        results[name] = T.run_episode(env, net, greedy=not a.sample, seed=a.seed, trace=tr)[0]
        traces[name] = tr
    n_tl = len(env.tl_ids)
    env.close()

    print(f"\nMot bo trong so, {n_tl} den quyet dinh doc lap | mang: {cfg} | seed {a.seed} | {ep_len} s")
    cols = list(results)
    print(f"{'chi so':18s}" + "".join(f" | {c:>12s}" for c in cols))
    for k, label in T.METRICS:
        print(f"{label:18s}" + "".join(f" | {results[c][k]:12.1f}" for c in cols))
    for name in cols:
        report(name, traces[name], env, dt, out_dir)


if __name__ == "__main__":
    main()
