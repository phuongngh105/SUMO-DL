#!/usr/bin/env python3
"""
compare_webster.py - SO SANH best.pt (PPO) VOI DEN CO DINH THEO WEBSTER tren N seed ngau nhien.

  * Cung 1 file .sumocfg (mang + file route cua ban), cung luu luong cho ca hai phuong phap o moi seed
    (luu luong sinh tu file route nhu luc train: Poisson + jitter, xem --rate-jitter).
  * PPO: chay greedy (khong train, khong lay mau), model = best.pt.
  * Den co dinh (3 cach, chon bang --baseline):
      webster   (mac dinh) TU TINH ke hoach Webster tu luu luong cua file route (xem webster_ke_hoach.csv)
      original  chuong trinh tlLogic dang co trong file .net (neu ban da tinh Webster va dat vao mang roi)
      --fixed-cfg B.sumocfg  dung chuong trinh den trong 1 file .sumocfg/.net KHAC (cung cau truc mang)
  * Xuat MOI gia tri: tung seed, TB, do lech chuan, chenh lech, % thay doi, so seed PPO tot hon,
    p-value (t-test cap doi), tham so dang dung, ke hoach Webster, bieu do.

Chay (Windows):
  python compare_webster.py --cfg D:\\Project\\SUMO-DL\\test2\\test2.sumocfg ^
         --run-dir D:\\Project\\SUMO-DL\\Train\\results\\seed_1234 --n-seeds 10

  (--run-dir: thu muc co best.pt + meta.json. Hoac dung --weights duong_dan\\best.pt va tu dat tham so.)
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_ppo_tsc as T      # noqa: E402  (cung thu muc voi script train)
import gen_demand              # noqa: E402
import traci                   # noqa: E402

# chieu "tot hon" cua tung chi so: -1 = thap hon tot hon, +1 = cao hon tot hon, 0 = khong danh gia
DIRECTION = {"awt": -1, "aql": -1, "speed": +1, "thr": +1, "departed": 0, "arrived": +1, "arr_rate": +1,
             "in_net": -1, "tele": -1, "stuck_n": -1, "stuck_max": -1, "pen": -1, "reward": +1,
             "reward_base": +1, "reward_pen": -1, "r_wt": +1, "r_ql": +1, "r_thr": +1}

TRAIN_DEFAULTS = dict(delta_t=10, yellow=3, min_green=5, max_green=60, wt_memory=300, episode_len=7200,
                      w1=0.05, w2=0.5, w3=0.0, w4=2.0, stuck_wait=60.0, reward_clip=10.0, stuck_cap=30.0)


# ============================================================================= WEBSTER
def split_green(y, G, gmin):
    """Chia tong xanh hieu dung G (giay, so nguyen) cho cac pha theo ti le y, moi pha >= gmin."""
    n = len(y)
    Y = sum(y)
    raw = [G * v / Y for v in y] if Y > 0 else [G / n] * n
    fixed = set()
    for _ in range(n):
        low = [i for i in range(n) if i not in fixed and raw[i] < gmin]
        if not low:
            break
        fixed.update(low)
        for i in low:
            raw[i] = float(gmin)
        rest = [i for i in range(n) if i not in fixed]
        if not rest:
            break
        Grest = G - gmin * len(fixed)
        Yr = sum(y[i] for i in rest)
        for i in rest:
            raw[i] = Grest * (y[i] / Yr if Yr > 0 else 1.0 / len(rest))
    fl = [int(math.floor(v)) for v in raw]
    rem = int(round(G)) - sum(fl)
    for i in sorted(range(n), key=lambda i: raw[i] - fl[i], reverse=True)[:max(rem, 0)]:
        fl[i] += 1
    return fl


def estimate_demand(env_est, bin_s):
    """Chay SUMO 1 lan (khong dieu khien), lay LUOT XE THEO HUONG (canh a -> canh b) cua moi xe duoc nap,
    quy doi theo chieu dai (cung cach quy doi voi AQL trong train: (dai + minGap)/7.5 m).
    Tra ve (bins, conn_map, logic_info). bins[k] = {(a, b): so xe quy doi}."""
    env_est.reset(control=False, seed=0)
    net = env_est.net
    tls = list(env_est.tl_ids)
    info = {}
    for tl in tls:
        lg = next(l for l in traci.trafficlight.getAllProgramLogics(tl) if l.programID == env_est.prog[tl])
        states = [p.state for p in lg.phases]
        durs = [float(p.duration) for p in lg.phases]
        greens = list(env_est.greens[tl])
        inter = {}
        for g in greens:                                   # khoang vang+do toan bo sau pha xanh g
            j, s = (g + 1) % len(states), 0.0
            while j not in greens and j != g:
                s += durs[j]
                j = (j + 1) % len(states)
            inter[g] = s
        info[tl] = dict(states=states, durs=durs, greens=greens, inter=inter, prog=env_est.prog[tl])
    conn_map, bins = {}, {}

    def conns(a, b):
        key = (a, b)
        if key not in conn_map:
            out = []
            try:
                for c in net.getEdge(a).getConnections(net.getEdge(b)):
                    tl = c.getTLSID()
                    if tl in info and c.getTLLinkIndex() >= 0:
                        out.append((tl, c.getTLLinkIndex(), c.getFromLane().getID()))
            except Exception:
                pass
            conn_map[key] = out
        return conn_map[key]

    n_veh = 0
    while traci.simulation.getTime() < env_est.T:
        traci.simulationStep()
        k = int(traci.simulation.getTime() // bin_s)
        for vid in traci.simulation.getLoadedIDList():
            try:
                route = traci.vehicle.getRoute(vid)
                w = (traci.vehicle.getLength(vid) + traci.vehicle.getMinGap(vid)) / T.EQ_LEN
            except traci.TraCIException:
                continue
            n_veh += 1
            d = bins.setdefault(k, {})
            for a, b in zip(route[:-1], route[1:]):
                if conns(a, b):
                    d[(a, b)] = d.get((a, b), 0.0) + w
    env_est.close()
    return bins, conn_map, info, n_veh


def webster_plan(env_est, bin_s, basis, sat, gmin, c_min, c_max, common_cycle):
    bins, conn_map, info, n_veh = estimate_demand(env_est, bin_s)
    T_end = env_est.T
    if not bins:
        raise RuntimeError("Khong dem duoc xe nao tu file route -> khong tinh duoc Webster")
    # luu luong xe quy doi / gio theo tung (den, pha, lan) cho tung bin
    per_bin = {}
    for k, mv in bins.items():
        F = {}
        for (a, b), cnt in mv.items():
            cs = conn_map[(a, b)]
            hrs = max(min(bin_s, T_end - k * bin_s), 1) / 3600.0      # do dai that cua khoang (khoang cuoi co the ngan hon)
            q = cnt / hrs / len(cs)           # chia deu cho cac lan duoc phep di huong nay
            for tl, idx, lane in cs:
                st = info[tl]["states"]
                sv = [g for g in info[tl]["greens"] if idx < len(st[g]) and st[g][idx] in "Gg"]
                for g in sv:
                    F.setdefault((tl, g), {})
                    F[(tl, g)][lane] = F[(tl, g)].get(lane, 0.0) + q / len(sv)
        per_bin[k] = F
    totals = {k: sum(sum(v.values()) for v in F.values()) for k, F in per_bin.items()}
    if basis == "peak":
        use = [max(totals, key=totals.get)]
    else:
        use = sorted(per_bin)
    plan, cyc0 = {}, {}
    for tl, inf in info.items():
        ph = []
        for g in inf["greens"]:
            lane_q = {}
            for k in use:
                for lane, q in per_bin[k].get((tl, g), {}).items():
                    lane_q[lane] = lane_q.get(lane, 0.0) + q / len(use)
            if lane_q:
                crit = max(lane_q, key=lane_q.get)
                qc = lane_q[crit]
            else:
                crit, qc = "", 0.0
            ph.append(dict(phase=g, crit_lane=crit, q_crit=qc, y=qc / sat, lost=inf["inter"][g],
                           g_orig=inf["durs"][g]))
        L = sum(p["lost"] for p in ph)
        Y = sum(p["y"] for p in ph)
        c0 = (1.5 * L + 5.0) / (1.0 - Y) if Y < 0.95 else float("inf")
        cyc0[tl] = dict(L=L, Y=Y, C0=c0, orig=sum(inf["durs"]))
        plan[tl] = ph
    for tl in plan:
        c = cyc0[tl]["C0"]
        cyc0[tl]["C_clip"] = int(round(min(max(c if math.isfinite(c) else c_max, c_min), c_max)))
    Cc = max(v["C_clip"] for v in cyc0.values())
    for tl, ph in plan.items():
        C = Cc if common_cycle else cyc0[tl]["C_clip"]
        L = int(round(cyc0[tl]["L"]))
        gs = split_green([p["y"] for p in ph], C - L, gmin)
        for p, g in zip(ph, gs):
            p["g_webster"] = g
        cyc0[tl]["C_used"] = C
    return plan, cyc0, info, dict(n_vehicles=n_veh, bins={k: round(v) for k, v in totals.items()})


def apply_plan(env, plan):
    for tl in env.tl_ids:
        if tl not in plan:
            continue
        lg = next(l for l in traci.trafficlight.getAllProgramLogics(tl) if l.programID == env.prog[tl])
        for p in plan[tl]:
            ph = lg.phases[p["phase"]]
            ph.duration = float(p["g_webster"])
            if hasattr(ph, "minDur"):
                ph.minDur = ph.maxDur = float(p["g_webster"])
        lg.programID = "webster"
        lg.currentPhaseIndex = 0
        traci.trafficlight.setProgramLogic(tl, lg)
        traci.trafficlight.setProgram(tl, "webster")
        traci.trafficlight.setPhase(tl, 0)
        traci.trafficlight.setPhaseDuration(tl, float(lg.phases[0].duration))


def install_webster(env, plan):
    """Moi lan reset KHONG dieu khien (baseline) -> nap ke hoach Webster. PPO (control=True) khong bi anh huong."""
    orig = env.reset

    def reset(control=True, seed=None):
        obs = orig(control=control, seed=seed)
        if not control:
            apply_plan(env, plan)
        return obs
    env.reset = reset


# ============================================================================= THONG KE / XUAT
def ttest_paired(x, y):
    d = np.array(x, float) - np.array(y, float)
    n = len(d)
    if n < 2 or np.allclose(d.std(ddof=1), 0):
        return float("nan")
    try:
        from scipy import stats
        return float(stats.ttest_rel(x, y).pvalue)
    except Exception:
        return float("nan")


def sumo_version():
    try:
        import subprocess
        return subprocess.run([T.checkBinary("sumo"), "--version"], capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
    except Exception:
        return ""


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:12]


def rnd(v, n=4):
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", required=True, help=".sumocfg dung de so sanh (mang + file route)")
    p.add_argument("--run-dir", default=None, help="thu muc ket qua train (co best.pt, meta.json, best_metrics.csv)")
    p.add_argument("--weights", default=None, help="duong dan .pt (mac dinh <run-dir>/best.pt)")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="chi dinh seed cu the (bo qua ngau nhien)")
    p.add_argument("--master-seed", type=int, default=None, help="seed de CHON 10 seed ngau nhien (in ra, de lap lai)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--baseline", choices=["webster", "original"], default="webster")
    p.add_argument("--fixed-cfg", default=None, help="so voi den co dinh trong .sumocfg KHAC (uu tien hon --baseline)")
    p.add_argument("--webster-basis", choices=["mean", "peak"], default="mean",
                   help="luu luong thiet ke: mean = TB cac khoang 1 gio; peak = khoang 1 gio dong nhat")
    p.add_argument("--webster-bin", type=int, default=3600, help="do dai khoang tinh luu luong (s)")
    p.add_argument("--sat-flow", type=float, default=1800.0, help="luu luong bao hoa moi lan (xe quy doi/gio)")
    p.add_argument("--webster-min-green", type=int, default=5)
    p.add_argument("--c-min", type=int, default=30)
    p.add_argument("--c-max", type=int, default=120)
    p.add_argument("--independent-cycles", action="store_true",
                   help="moi den 1 chu ky rieng (mac dinh: chu ky chung = lon nhat, hop voi 2 den gan nhau)")
    # tham so moi truong: mac dinh lay tu meta.json cua run-dir, roi moi toi mac dinh luc train
    for k, v in TRAIN_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    p.add_argument("--rate-jitter", type=float, default=0.1)
    p.add_argument("--mix-jitter", type=float, default=0.0)
    p.add_argument("--no-poisson", action="store_true")
    p.add_argument("--tls", nargs="+", default=None)
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    meta = {}
    if a.run_dir:
        mp = os.path.join(a.run_dir, "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
    st = meta.get("stuck", {})
    src = {}
    for k, dv in TRAIN_DEFAULTS.items():
        cli = getattr(a, k)
        if cli is not None:
            val, why = cli, "dong lenh"
        elif k in meta:
            val, why = meta[k], "meta.json"
        elif k in st:
            val, why = st[k], "meta.json"
        else:
            val, why = dv, "mac dinh luc train"
        setattr(a, k, type(dv)(val))
        src[k] = why
    wpath = a.weights or (os.path.join(a.run_dir, "best.pt") if a.run_dir else None)
    if not wpath or not os.path.exists(wpath):
        raise SystemExit("Khong thay best.pt. Dung --run-dir <thu muc train> hoac --weights <duong dan .pt>")
    if not os.path.exists(a.cfg):
        raise SystemExit(f"Khong thay file cfg: {a.cfg}")

    # ---- seed
    if a.seeds:
        seeds, how = list(a.seeds), "chi dinh bang --seeds"
        ms = None
    else:
        ms = a.master_seed if a.master_seed is not None else int(time.time()) % 1_000_000
        seeds = random.Random(ms).sample(range(20_000, 100_000), a.n_seeds)    # tranh seed train (0..) va seed eval (10000+)
        how = f"ngau nhien tu master-seed {ms} (trong khoang 20000-99999, khong trung seed train/eval)"
    out = a.out_dir or os.path.join(os.getcwd(), "so_sanh_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"Thu muc ket qua : {os.path.abspath(out)}")
    print(f"Model           : {wpath}  (sha1 {sha1(wpath)})")
    print(f"Seed ({len(seeds)})        : {seeds}  [{how}]")

    demand = dict(mode="file", route_files=gen_demand.route_files_from_cfg(a.cfg), rate_jitter=a.rate_jitter,
                  poisson=not a.no_poisson, mix_jitter=a.mix_jitter)

    def mk(cfg, label, dem):
        return T.MultiJunctionEnv(cfg, a.tls, None, None, a.gui, a.delta_t, a.yellow, a.episode_len,
                                  a.w1, a.w2, a.w3, 0, label, a.min_green, a.max_green, a.wt_memory, dem,
                                  None, a.w4, a.stuck_wait, a.reward_clip, a.stuck_cap)

    env = mk(a.cfg, "ppo", demand)
    env.reset(seed=seeds[0])
    print(f"Den dieu khien  : {env.tl_ids} | pha xanh {env.greens}")
    env.close()
    net = T.load_net(env, wpath)
    net.eval()

    # ---- den co dinh
    plan = cyc0 = None
    env_fix = env
    if a.fixed_cfg:
        env_fix = mk(a.fixed_cfg, "fix", demand)
        base_name = f"Co dinh ({os.path.basename(a.fixed_cfg)})"
        print(f"Den co dinh     : tlLogic trong {a.fixed_cfg}")
    elif a.baseline == "original":
        base_name = "Co dinh (tlLogic trong net)"
        print("Den co dinh     : chuong trinh tlLogic dang co trong file net (khong tinh lai)")
    else:
        base_name = "Webster"
        dem_est = dict(demand, rate_jitter=0.0, poisson=False, mix_jitter=0.0)      # luu luong dung bang file, khong jitter
        env_est = mk(a.cfg, "est", dem_est)
        print("Dang tinh Webster tu luu luong file route (chay SUMO 1 lan khong dieu khien)...")
        plan, cyc0, info, dinfo = webster_plan(env_est, a.webster_bin, a.webster_basis, a.sat_flow,
                                               a.webster_min_green, a.c_min, a.c_max, not a.independent_cycles)
        install_webster(env, plan)
        for tl, ph in plan.items():
            c = cyc0[tl]
            print(f"  {tl}: L={c['L']:.0f}s Y={c['Y']:.3f} C0={c['C0']:.1f}s -> chu ky dung {c['C_used']}s (tlLogic goc {c['orig']:.0f}s)")
            for p in ph:
                print(f"     pha {p['phase']}: lan toi han {p['crit_lane'] or '-'} q={p['q_crit']:.0f} xe qd/h  y={p['y']:.3f}"
                      f"  xanh goc {p['g_orig']:.0f}s -> Webster {p['g_webster']}s")
            if c["Y"] >= 0.95:
                print(f"     CANH BAO: Y={c['Y']:.2f} >= 0.95 (qua bao hoa): cong thuc Webster khong con y nghia, "
                      f"dung chu ky toi da {a.c_max}s, chia xanh theo ti le y.")

    # ---- chay
    res = {"PPO": [], "FIX": []}
    t_all = time.time()
    print(f"\n{'seed':>6s} | {base_name[:10]:>10s}: AWT   den  ket  | {'PPO best':>10s}: AWT   den  ket | thoi gian")
    for i, s in enumerate(seeds):
        t0 = time.time()
        rf, _, _ = T.run_episode(env_fix, None, fixed=True, seed=s)
        rp, _, _ = T.run_episode(env, net, greedy=True, seed=s)
        res["FIX"].append(rf)
        res["PPO"].append(rp)
        print(f"{s:6d} | {'':>10s}  {rf['awt']:5.1f} {rf['arrived']:5d} {rf['stuck_n']:5.1f} | {'':>10s}  "
              f"{rp['awt']:5.1f} {rp['arrived']:5d} {rp['stuck_n']:5.1f} | {time.time()-t0:.0f}s  ({i+1}/{len(seeds)})", flush=True)
    env.close()
    env_fix.close()

    # ---- bang tong hop in ra
    n = len(seeds)
    lines = []
    head = f"{'chi so':24s} | {base_name:>20s} | {'PPO best':>20s} | {'PPO - co dinh':>13s} | {'%':>7s} | tot hon | p"
    lines.append(f"So sanh tren {n} seed, TB +- do lech chuan (cung luu luong moi seed)")
    lines.append(head)
    stats = {}
    for k, label in T.METRICS:
        fx = [r[k] for r in res["FIX"]]
        pp = [r[k] for r in res["PPO"]]
        d = DIRECTION.get(k, 0)
        mf, mp = float(np.mean(fx)), float(np.mean(pp))
        pct = 100.0 * (mp - mf) / abs(mf) if abs(mf) > 1e-9 else float("nan")
        win = int(sum((p_ - f_) * d > 1e-12 for f_, p_ in zip(fx, pp))) if d else None
        pv = ttest_paired(pp, fx)
        stats[k] = (mf, mp, pct, win, pv)
        lines.append(f"{label:24s} | {mf:9.2f} +- {np.std(fx):7.2f} | {mp:9.2f} +- {np.std(pp):7.2f} | "
                     f"{mp - mf:13.2f} | {pct:6.1f}% | {'' if win is None else f'{win}/{n}':>7s} | "
                     f"{'' if not math.isfinite(pv) else f'{pv:.3f}'}")
    lines.append("tot hon = so seed ma PPO tot hon den co dinh (AWT/AQL/ket/phat: thap hon tot; toc do/den dich/reward: cao hon tot). "
                 "p = t-test cap doi (can scipy).")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out, "tom_tat.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # ---- CSV chi tiet: TAT CA chi so (tong quan, ket xe, reward, phat, tung canh, tung den)
    fl_f = [T._flat_summary(r) for r in res["FIX"]]
    fl_p = [T._flat_summary(r) for r in res["PPO"]]
    keys = list(dict.fromkeys(k for d in fl_f + fl_p for k in d))
    key2name = {(g, lab): k for k, g, lab in T.SCALAR_ROWS}
    fname = base_name
    with open(os.path.join(out, "so_sanh_chi_tiet.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nhom", "chi_so"] + [f"{fname}_seed_{s}" for s in seeds] + [f"{fname}_TB", f"{fname}_std"]
                   + [f"PPO_seed_{s}" for s in seeds] + ["PPO_TB", "PPO_std", "PPO_tru_codinh", "phan_tram",
                                                         "so_seed_PPO_tot_hon", "p_value_t_cap_doi"])
        for k in keys:
            fx = [d.get(k) for d in fl_f]
            pp = [d.get(k) for d in fl_p]
            xf = [v for v in fx if v is not None]
            xp = [v for v in pp if v is not None]
            mf = float(np.mean(xf)) if xf else None
            mp = float(np.mean(xp)) if xp else None
            diff = (mp - mf) if (mf is not None and mp is not None) else None
            pct = 100.0 * diff / abs(mf) if (diff is not None and abs(mf) > 1e-9) else None
            mk_ = key2name.get(k)
            d = DIRECTION.get(mk_, 0) if mk_ else 0
            win = sum((p_ - f_) * d > 1e-12 for f_, p_ in zip(fx, pp) if f_ is not None and p_ is not None) if d else ""
            pv = ttest_paired(pp, fx) if (len(xf) == n and len(xp) == n) else float("nan")
            w.writerow([k[0], k[1]] + [rnd(v) for v in fx] + [rnd(mf), rnd(np.std(xf)) if xf else ""]
                       + [rnd(v) for v in pp] + [rnd(mp), rnd(np.std(xp)) if xp else "", rnd(diff), rnd(pct, 2), win, rnd(pv, 4)])

    # ---- CSV tung seed (dang doc, tien ve bieu do)
    mkeys = [k for k, _g, _l in T.SCALAR_ROWS if all(k in r for r in res["FIX"] + res["PPO"])]
    with open(os.path.join(out, "so_sanh_tung_seed.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["phuong_phap", "seed"] + mkeys)
        for name, rows in ((fname, res["FIX"]), ("PPO best", res["PPO"])):
            for s, r in zip(seeds, rows):
                w.writerow([name, s] + [rnd(r[k]) for k in mkeys])

    # ---- 1 FILE CSV DUY NHAT de ve do thi: tung seed + TB + std + min + max + chenh lech PPO - co dinh
    cols = [(key2name.get(k) or f"{k[0]}|{k[1]}", k) for k in keys]
    with open(os.path.join(out, "ket_qua_tong_hop.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["phuong_phap", "dong", "seed"] + [c for c, _ in cols])
        agg = {}
        for name, fl in ((fname, fl_f), ("PPO best", fl_p)):
            for s_, d in zip(seeds, fl):
                w.writerow([name, "seed", s_] + [rnd(d.get(k)) for _, k in cols])
            for lab, fn in (("TB", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
                vals = []
                for _, k in cols:
                    xs = [d[k] for d in fl if k in d]
                    vals.append(float(fn(xs)) if xs else None)
                agg[(name, lab)] = vals
                w.writerow([name, lab, ""] + [rnd(v) for v in vals])
        mf, mp = agg[(fname, "TB")], agg[("PPO best", "TB")]
        w.writerow(["PPO_tru_codinh", "TB", ""] + [rnd(None if (a_ is None or b_ is None) else b_ - a_) for a_, b_ in zip(mf, mp)])
        w.writerow(["PPO_tren_codinh", "phan_tram", ""] + [
            rnd(100.0 * (b_ - a_) / abs(a_), 2) if (a_ is not None and b_ is not None and abs(a_) > 1e-9) else ""
            for a_, b_ in zip(mf, mp)])
        pv = []
        for _, k in cols:
            x_f = [d[k] for d in fl_f if k in d]
            x_p = [d[k] for d in fl_p if k in d]
            pv.append(ttest_paired(x_p, x_f) if len(x_f) == n and len(x_p) == n else None)
        w.writerow(["PPO_vs_codinh", "p_value_t_cap_doi", ""] + [rnd(v) for v in pv])

    # ---- tham so da dung
    prm = [("chung", "cfg", os.path.abspath(a.cfg), "dong lenh"),
           ("chung", "weights", os.path.abspath(wpath), "dong lenh"),
           ("chung", "weights_sha1", sha1(wpath), ""),
           ("chung", "weights_sua_luc", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(wpath))), ""),
           ("chung", "so_seed", n, ""), ("chung", "seeds", " ".join(map(str, seeds)), how),
           ("chung", "master_seed", ms if ms is not None else "", ""),
           ("chung", "den_co_dinh", base_name, ""),
           ("chung", "sumo", sumo_version(), ""),
           ("chung", "python/torch", f"{platform.python_version()} / {torch.__version__}", ""),
           ("luu_luong", "demand", "file (Poisson tu file route)" if not a.no_poisson else "file (deu)", ""),
           ("luu_luong", "rate_jitter", a.rate_jitter, ""), ("luu_luong", "mix_jitter", a.mix_jitter, ""),
           ("luu_luong", "route_files", ";".join(demand["route_files"]), "")]
    for k in TRAIN_DEFAULTS:
        prm.append(("moi_truong", k, getattr(a, k), src[k]))
    if a.run_dir and os.path.exists(os.path.join(a.run_dir, "best_metrics.csv")):
        with open(os.path.join(a.run_dir, "best_metrics.csv"), encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if row and row[0] in ("tham_so", "meta"):
                    prm.append(("train_" + row[0], row[1], row[-4] if row[0] == "tham_so" else row[-4], "best_metrics.csv"))
    if cyc0:
        prm += [("webster", "co_so_luu_luong", a.webster_basis, ""), ("webster", "bin_s", a.webster_bin, ""),
                ("webster", "sat_flow_xe_qd_moi_gio_moi_lan", a.sat_flow, "gia tri thuong dung (Webster 1958 / FHWA-HOP-08-024)"),
                ("webster", "min_green", a.webster_min_green, ""), ("webster", "c_min/c_max", f"{a.c_min}/{a.c_max}", ""),
                ("webster", "chu_ky_chung", not a.independent_cycles, ""),
                ("webster", "cong_thuc", "C0=(1.5L+5)/(1-Y); g_i=(C-L)*y_i/Y", "Webster 1958, Road Research Tech. Paper 39")]
    with open(os.path.join(out, "tham_so_su_dung.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nhom", "ten", "gia_tri", "nguon"])
        w.writerows(prm)

    # ---- ke hoach Webster
    if plan:
        with open(os.path.join(out, "webster_ke_hoach.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["den", "pha", "lan_toi_han", "q_toi_han_xe_qd_h", "y", "mat_thoi_gian_L_pha_s",
                        "xanh_goc_s", "xanh_webster_s", "L_tong", "Y_tong", "C0_s", "chu_ky_goc_s", "chu_ky_dung_s"])
            for tl, ph in plan.items():
                c = cyc0[tl]
                for p in ph:
                    w.writerow([tl, p["phase"], p["crit_lane"], rnd(p["q_crit"], 1), rnd(p["y"], 4), p["lost"],
                                p["g_orig"], p["g_webster"], c["L"], rnd(c["Y"], 4), rnd(c["C0"], 1), c["orig"], c["C_used"]])

    # ---- bieu do
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = [("awt", "AWT (s)"), ("aql", "AQL (xe qd)"), ("speed", "toc do TB (m/s)"),
                ("arr_rate", "ti le den dich (%)"), ("stuck_n", "xe ket TB/buoc"), ("pen", "phat ket xe TB")]
        fig, axs = plt.subplots(2, 3, figsize=(13, 7))
        x = np.arange(n)
        for ax, (k, lab) in zip(axs.ravel(), show):
            ax.bar(x - 0.2, [r[k] for r in res["FIX"]], 0.4, label=fname)
            ax.bar(x + 0.2, [r[k] for r in res["PPO"]], 0.4, label="PPO best")
            ax.axhline(stats[k][0], color="C0", ls="--", lw=1)
            ax.axhline(stats[k][1], color="C1", ls="--", lw=1)
            ax.set_title(f"{lab}: {stats[k][2]:+.1f}%")
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in seeds], rotation=60, fontsize=7)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(f"PPO best vs {fname} tren {n} seed ngau nhien (duong gach = trung binh)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "so_sanh.png"), dpi=130)
    except Exception as e:
        print(f"(khong ve duoc bieu do: {e})")

    print(f"\nXONG ({time.time()-t_all:.0f}s). Trong {os.path.abspath(out)}:")
    for fn, mo_ta in (("tom_tat.txt", "bang tong hop (nhu tren)"),
                      ("ket_qua_tong_hop.csv", "FILE CHINH de ve do thi: tung seed + TB + std + min + max + chenh lech + p"),
                      ("so_sanh_chi_tiet.csv", "MOI chi so x tung seed, TB, std, chenh lech, %, p-value"),
                      ("so_sanh_tung_seed.csv", "moi dong = 1 seed x 1 phuong phap"),
                      ("tham_so_su_dung.csv", "moi tham so dang dung + nguon + seed + hash model"),
                      ("webster_ke_hoach.csv", "luu luong, y, xanh goc vs Webster"), ("so_sanh.png", "bieu do")):
        if os.path.exists(os.path.join(out, fn)):
            print(f"  {fn:24s}: {mo_ta}")


if __name__ == "__main__":
    main()
