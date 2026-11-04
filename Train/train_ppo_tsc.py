"""
PPO-TSC: 1 model Actor-Critic dung chung cho cac den giao thong (parameter sharing)
===========================================================================
Ban nay chinh tu ban goc theo 5 muc uu tien + luu luong ngau nhien:
  1. --ent 0.01     (mac dinh cu 0.0: chinh sach co ve 1 pha qua som)
  2. --lam 0.95     (mac dinh cu 0.0: TD 1 buoc, nhieu va lech)  -> GAE
  3. --gamma 0.95   (mac dinh cu 0.9: chi nhin ~100 s)
  4. --batch 64     (mac dinh cu 20: gradient qua nhieu)
  5. TAT TELEPORT   (--time-to-teleport -1) + --waiting-time-memory 300 -> xe ket khong bien mat khoi so lieu
Them:
  - Pha xanh/vang TU DO TU tlLogic (khong can sua tay GREEN_PHASES khi doi pha trong Netedit)
  - min_green / max_green bang action mask (che hanh dong khong hop le, ca luc train lan luc update)
  - dem so xe bi teleport (cot 'tele'), phai = 0
  - luu luong: --demand file (mac dinh) LAY SO LIEU TU FILE ROUTE cua .sumocfg (gen_demand.from_route_files);
    --demand random = luu luong ngau nhien cua gen_demand; --demand cfg = chay thang file route
  - danh gia tren nhieu seed co dinh, chon model tot nhat theo DIEM EVAL = a*stuck/stuck_base + (1-a)*AWT/AWT_base
    (uu tien ket xe truoc, roi den AWT; a = --select-alpha; KHONG theo tong reward,
    vi reward la hieu so lien tiep nen tong ca episode chi con = WT dau - WT cuoi, khong phan anh chat luong)
  - baseline dung chuong trinh den goc cua mang (khong dung vao den), cung luu luong voi PPO
  - sau MOI episode: in + ghi CSV luu luong xe (xe vao mang / den dich / thoat ra / con trong mang)
  - progress.csv + progress.png cap nhat SAU MOI episode (khong con theo khoi 10 episode)
  - best_metrics.csv: MOI chi so cua best.pt (AWT, AQL, reward + thanh phan, phat, xe vao/ra tung canh bien,
    ti le den dich, tung den...) tren cac seed eval + baseline; ghi lai moi khi co best moi
  - train xong: --out-dir chua best.pt, last.pt, meta.json, trained.sumocfg, demo.rou.xml,
    train_log.csv, final_report.csv (baseline vs best vs last). Chay lai bang run_trained.py

Cong thuc goc (Wang et al., EAAI 2025 "PPO-TSC"; Yang et al. T-ITS Modified PPO) giu nguyen:
  State  : QL_i, WT_i tung lan vao + one-hot pha + (moi) thoi gian da o pha hien tai
  Reward : r = w1*(WT_prev - WT) + w2*(QL_prev - QL) o cac canh ngoai cung
  Actor  : L_clip (clip thich nghi, tat bang --fixed-clip)      Critic: MSE voi return muc tieu
  Phat ket xe (MOI, khong co trong 2 bai bao):
     P_tl = min( w4 * (1/L) * sum_{xe i dung lien tuc >= tau tren lan vao cua den tl} (wait_i/tau)^2 , stuck_cap )
     r_tl = clip(r, +-reward_clip) - P_tl         (stuck_cap = 3 x reward_clip: phat NANG HON thuong)
"""
import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402
import sumolib  # noqa: E402
from sumolib import checkBinary  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_demand  # noqa: E402

# ----------------------------------------------------------------------------- cau hinh mac dinh

EQ_LEN = 7.5                  # 1 'xe quy doi' = 7.5 m (o to 5 m + minGap 2.5 m)
V_TOP = 13.89                # toc do toi da (m/s) trong net
LANE_Q_NORM = 10.0
N_FEAT = 8                    # so dac trung moi khe pha: ql_tb, ql_max, wt_tb, wt_max, dang_xanh, thoi_gian_xanh, so_lan, co_hieu_luc
EVAL_SEED_BASE = 10_000       # seed co dinh cho danh gia (khac seed train)


# ----------------------------------------------------------------------------- mang no-ron
def mlp(inp, hid, out):
    """MLP 3 lop day du (Primary: 'three-layer fully connected network')."""
    return nn.Sequential(nn.Linear(inp, hid), nn.ReLU(),
                         nn.Linear(hid, hid), nn.ReLU(),
                         nn.Linear(hid, out))


class ActorCritic(nn.Module):
    """Mang DUNG CHUNG KHONG PHU THUOC CAU TRUC mang giao thong.
    Moi PHA XANH cua 1 den la 1 'khe' (slot) voi N_FEAT dac trung; cung 1 bo trong so cham diem cho moi khe,
    nen so pha / so lan / so den khac nhau deu dung duoc (kieu DeepSets: ma hoa tung khe + ngu canh chung)."""

    def __init__(self, feat_dim=N_FEAT, hid=64):
        super().__init__()
        self.actor = PhaseActor(feat_dim, hid)      # pi_theta(a|s)
        self.critic = PhaseCritic(feat_dim, hid)    # V_phi(s)
        self.feat_dim = feat_dim

    def dist(self, obs, mask=None):
        logits = self.actor(obs)
        if mask is not None:                         # che hanh dong khong hop le (min/max green + khe trong)
            logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    def value(self, obs):
        return self.critic(obs)


class PhaseActor(nn.Module):
    def __init__(self, f, hid):
        super().__init__()
        self.enc = mlp(f, hid, hid)
        self.head = mlp(2 * hid, hid, 1)

    def forward(self, x):                            # x: (B, K, F), cot cuoi = co hieu luc (1/0)
        valid = (x[..., -1:] > 0.5).float()
        h = torch.relu(self.enc(x)) * valid
        ctx = h.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(torch.cat([h, ctx.expand_as(h)], -1)).squeeze(-1)    # (B, K)


class PhaseCritic(nn.Module):
    def __init__(self, f, hid):
        super().__init__()
        self.enc = mlp(f, hid, hid)
        self.head = mlp(2 * hid, hid, 1)

    def forward(self, x):
        valid = (x[..., -1:] > 0.5).float()
        h = torch.relu(self.enc(x)) * valid
        mean = h.sum(1) / valid.sum(1).clamp(min=1.0)
        mx = (h - (1 - valid) * 1e9).max(1).values.clamp(min=0.0)
        return self.head(torch.cat([mean, mx], -1)).squeeze(-1)


# ----------------------------------------------------------------------------- pha den
def detect_phases(tl):
    """Tu tlLogic: pha xanh = khong co 'y' va co 'G'/'g'; pha vang di kem = pha ngay sau neu co 'y'.
    Tra ve (greens, yellow_of, served): served[k] = cac LAN VAO duoc cap xanh o pha xanh thu k."""
    logics = traci.trafficlight.getAllProgramLogics(tl)
    cur = traci.trafficlight.getProgram(tl)
    lg = next((l for l in logics if l.programID == cur), logics[0])
    states = [p.state for p in lg.phases]
    n = len(states)

    def is_yellow(s):
        return "y" in s or "Y" in s

    greens = [i for i, s in enumerate(states) if not is_yellow(s) and any(c in "Gg" for c in s)]
    yellow_of = {}
    for i in greens:
        j = (i + 1) % n
        yellow_of[i] = j if is_yellow(states[j]) else None
    links = traci.trafficlight.getControlledLinks(tl)         # links[i] = [(lan_vao, lan_ra, via)]
    served = []
    for i in greens:
        lanes = []
        for li, grp in enumerate(links):
            if li < len(states[i]) and states[i][li] in "Gg":
                for conn in grp:
                    if conn[0] not in lanes:
                        lanes.append(conn[0])
        served.append(lanes)
    return greens, yellow_of, served, cur


# ----------------------------------------------------------------------------- moi truong
class MultiJunctionEnv:
    """Tu doc TAT CA tu file net/sumocfg: danh sach den, so pha, lan vao tung pha, cac nut ngoai cung (do reward).
    Them/bot nga tu, den, lan, pha trong Netedit -> chay lai, KHONG sua code."""

    def __init__(self, cfg, tl_ids=None, outer_in=None, outer_out=None, gui=False, delta_t=10, yellow=3,
                 episode_len=3600, w1=0.05, w2=0.5, w3=0.0, seed=0, label="sim",
                 min_green=5, max_green=60, wt_memory=300, demand=None, net_file=None,
                 w4=2.0, stuck_wait=60.0, reward_clip=10.0, stuck_cap=30.0):
        self.cfg, self.gui = cfg, gui
        self.w4, self.stuck_wait = w4, float(stuck_wait)          # phat ket xe: he so, nguong tau (s)
        self.reward_clip, self.stuck_cap = reward_clip, stuck_cap
        self.req_tls = list(tl_ids) if tl_ids else None            # None = tu nhan het den co dieu khien
        self.dt, self.ty, self.T = delta_t, yellow, episode_len
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.seed, self.label = seed, label
        self.min_green, self.max_green = min_green, max_green
        self.wt_memory = wt_memory
        self.wt_norm = float(wt_memory)
        self.demand = demand
        self.tmpdir = tempfile.mkdtemp(prefix="tsc_")
        self.net = sumolib.net.readNet(net_file or self._net_from_cfg())
        if outer_in or outer_out:
            self.outer_in, self.outer_out = list(outer_in or []), list(outer_out or [])
        else:                                                      # tu nhan: canh bat dau/ket thuc o bien mang
            self.outer_in, self.outer_out = gen_demand.boundary_edges(self.net)
        self.tl_ids = []
        self.demand_info = None
        self.n_override = 0

    def _net_from_cfg(self):
        root = ET.parse(self.cfg).getroot()
        node = root.find("./input/net-file")
        return os.path.join(os.path.dirname(os.path.abspath(self.cfg)), node.get("value"))

    def _gen_routes(self, out_file, seed):
        """Sinh file route cho 1 episode. mode='file': TU file route cua .sumocfg (giu so lieu cua ban);
        con lai: luu luong ngau nhien cua gen_demand.make_demand."""
        d = dict(self.demand)
        if d.pop("mode", None) == "file":
            return gen_demand.from_route_files(d.pop("route_files"), out_file, seed=seed, end=self.T, **d)
        return gen_demand.make_demand(self.net, out_file, seed=seed, end=self.T, **d)

    # -- vong doi
    def reset(self, control=True, seed=None):
        """control=False: de nguyen chuong trinh den goc cua mang (baseline)."""
        self.close()
        if seed is None:
            seed = self.seed
            self.seed += 1
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [binary, "-c", self.cfg, "--seed", str(seed), "--no-warnings", "--no-step-log",
               "--end", str(self.T),
               "--time-to-teleport", "-1",                      # TAT teleport
               "--waiting-time-memory", str(int(self.wt_memory))]
        if self.gui:
            cmd += ["--start", "--quit-on-end", "--delay", "50"]
        if self.demand is not None:                              # luu luong ngau nhien theo seed
            rou = os.path.join(self.tmpdir, f"{self.label}.rou.xml")
            self.demand_info = self._gen_routes(rou, seed)
            cmd += ["-r", rou]
        traci.start(cmd, label=self.label)

        # ---- tu nhan den + pha
        all_tls = list(traci.trafficlight.getIDList())
        want = self.req_tls if self.req_tls else all_tls
        missing = [t for t in want if t not in all_tls]
        if missing:
            raise RuntimeError(f"Khong co den {missing} trong mang. Cac den co: {all_tls}")
        self.greens, self.yellow_of, self.served, self.prog = {}, {}, {}, {}
        self.tl_ids, skipped = [], []
        for tl in want:
            g, yo, sv, prog = detect_phases(tl)
            if len(g) < 2:
                skipped.append(tl)
                continue
            self.tl_ids.append(tl)
            self.greens[tl], self.yellow_of[tl], self.served[tl], self.prog[tl] = g, yo, sv, prog
        if not self.tl_ids:
            raise RuntimeError("Khong co den nao co >= 2 pha xanh de dieu khien.")
        if skipped and not getattr(self, "_warned_skip", False):
            print(f"LUU Y: den {skipped} co < 2 pha xanh -> de nguyen chuong trinh goc, khong dieu khien.")
            self._warned_skip = True
        self.K = max(len(self.greens[tl]) for tl in self.tl_ids)     # so khe pha toi da (chi la do dem, khong anh huong model)
        self.n_actions = self.K
        self.obs_dim = (self.K, N_FEAT)

        self.lanes = {}
        for tl in self.tl_ids:
            seen = []
            for ln in traci.trafficlight.getControlledLanes(tl):
                if ln not in seen:
                    seen.append(ln)
            self.lanes[tl] = seen
        oi = [e for e in self.outer_in if e in set(traci.edge.getIDList())]
        oo = [e for e in self.outer_out if e in set(traci.edge.getIDList())]
        self.outer_edges = list(dict.fromkeys(oi + oo))
        if not self.outer_edges:                                   # mang khong co bien (vd vong xuyen): do tai canh vao den
            self.outer_edges = sorted({ln.rsplit("_", 1)[0] for tl in self.tl_ids for ln in self.lanes[tl]})
            self.outer_out = []
            if not getattr(self, "_warned_outer", False):
                print("LUU Y: khong tim thay canh bien -> do reward tai cac canh vao den.")
                self._warned_outer = True
        else:
            self.outer_out = oo
        self.outer_lanes = [f"{e}_{i}" for e in self.outer_edges
                            for i in range(traci.edge.getLaneNumber(e))]

        # thong ke theo canh bien / theo den (cho best_metrics.csv)
        self.edge_veh = {e: set() for e in self.outer_edges}          # xe KHAC NHAU da di qua moi canh (dem moi giay)
        self.edge_halt, self.edge_n = {e: 0.0 for e in self.outer_edges}, 0
        self.tl_stat = {tl: dict(pen=0.0, switches=0, steps=0, phase_steps={p: 0 for p in self.greens[tl]})
                        for tl in self.tl_ids}
        for e in self.outer_edges:
            traci.edge.subscribe(e, [traci.constants.LAST_STEP_VEHICLE_ID_LIST])
        self.control = control
        self.cur_phase = {tl: self.greens[tl][0] for tl in self.tl_ids}
        self.green_time = {tl: float(self.min_green) for tl in self.tl_ids}
        self.overridden = {tl: False for tl in self.tl_ids}
        if control:
            for tl in self.tl_ids:
                self._set_green(tl, self.greens[tl][0])
        self.seen_exit = set()
        self.prev_wt, self.prev_ql = self._outer_metrics()[:2]
        self.stats = dict(wt=[], ql=[], speed=[], thr=0, tele=0, departed=0, arrived=0, in_net=0,
                          stuck_n=[], stuck_max=0.0, pen=[], r_wt=[], r_ql=[], r_thr=[], r_base=[])
        self.pen = {tl: 0.0 for tl in self.tl_ids}
        return {tl: self.observe(tl) for tl in self.tl_ids}

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    # -- mo phong
    def _advance(self, n):
        ids_key = traci.constants.LAST_STEP_VEHICLE_ID_LIST
        for _ in range(int(n)):
            traci.simulationStep()
            self.stats["tele"] += traci.simulation.getStartingTeleportNumber()
            self.stats["departed"] += traci.simulation.getDepartedNumber()     # xe moi vao mang
            self.stats["arrived"] += traci.simulation.getArrivedNumber()       # xe da den dich
            for e, d in traci.edge.getAllSubscriptionResults().items():        # dem xe qua tung canh bien MOI GIAY
                ids = d.get(ids_key)
                if ids:
                    self.edge_veh[e].update(ids)

    def _halt_eq(self, ln):
        """Hang cho cua 1 lan tinh bang 'xe quy doi' (1 = 7.5 m = 1 o to + khoang cach).
        Moi xe dung (toc do < 0.1 m/s) tinh = (chieu dai + minGap) / 7.5, nen xe may ~0.5, o to ~1, xe tai ~2,
        thay vi dem moi xe la 1 (dem xe khong phan biet duoc 10 xe may voi 10 xe tai)."""
        tot = 0.0
        for vid in traci.lane.getLastStepVehicleIDs(ln):
            if traci.vehicle.getSpeed(vid) < 0.1:
                tot += (traci.vehicle.getLength(vid) + traci.vehicle.getMinGap(vid)) / EQ_LEN
        return tot

    # -- state: moi PHA XANH la 1 khe: QL/WT cua cac lan duoc cap xanh o pha do + dang xanh? + thoi gian xanh
    def observe(self, tl):
        ql, wt = {}, {}
        for ln in self.lanes[tl]:
            length = traci.lane.getLength(ln)
            ql[ln] = min(self._halt_eq(ln) / max(length / EQ_LEN, 1.0), 1.0)
            veh = traci.lane.getLastStepVehicleIDs(ln)
            if veh:
                head = max(veh, key=traci.vehicle.getLanePosition)       # xe dau hang
                wt[ln] = min(traci.vehicle.getAccumulatedWaitingTime(head) / self.wt_norm, 1.0)
            else:
                wt[ln] = 0.0
        x = np.zeros((self.K, N_FEAT), dtype=np.float32)
        for k, ph in enumerate(self.greens[tl]):
            lns = self.served[tl][k] or self.lanes[tl]
            q = [ql[l] for l in lns if l in ql] or [0.0]
            w = [wt[l] for l in lns if l in wt] or [0.0]
            cur = 1.0 if self.cur_phase[tl] == ph else 0.0
            x[k] = [np.mean(q), np.max(q), np.mean(w), np.max(w), cur,
                    cur * min(self.green_time[tl] / self.max_green, 1.0), min(len(lns) / 8.0, 1.0), 1.0]
        return x

    def action_mask(self, tl):
        """True = hanh dong duoc phep. Chua du min_green -> phai giu; qua max_green -> phai doi; khe trong -> cam."""
        n = len(self.greens[tl])
        m = np.zeros(self.K, dtype=bool)
        m[:n] = True
        cur_i = self.greens[tl].index(self.cur_phase[tl])
        if self.green_time[tl] < self.min_green:
            m[:] = False
            m[cur_i] = True
        elif self.green_time[tl] >= self.max_green and n > 1:
            m[cur_i] = False
        return m

    # -- do luu luong tai cac nut ngoai cung
    def _outer_metrics(self):
        waits, speeds = [], []
        for e in self.outer_edges:
            for vid in traci.edge.getLastStepVehicleIDs(e):
                waits.append(traci.vehicle.getAccumulatedWaitingTime(vid))
                speeds.append(traci.vehicle.getSpeed(vid))
        wt = float(np.mean(waits)) if waits else 0.0                              # AWT
        ql = float(np.mean([self._halt_eq(l) for l in self.outer_lanes]))  # AQL (xe quy doi)
        vaver = float(np.mean(speeds)) if speeds else V_TOP
        out_now = set(v for e in self.outer_out for v in traci.edge.getLastStepVehicleIDs(e))
        thr = len(out_now - self.seen_exit)
        self.seen_exit |= out_now
        return wt, ql, vaver, thr

    def _stuck_penalty(self):
        """Phat xe ket lau o ngan vao cua tung den.
        Xe 'ket' = dung lien tuc (toc do < 0.1 m/s) >= stuck_wait giay (traci getWaitingTime, tu reset khi xe di chuyen).
        P_tl = min(w4 * (1/L) * sum (wait_i/tau)^2, stuck_cap): ket cang lau phat cang tang theo binh phuong."""
        tau, pen, n_all, wmax = self.stuck_wait, {}, 0, 0.0
        for tl in self.tl_ids:
            acc, n = 0.0, 0
            for ln in self.lanes[tl]:
                if traci.lane.getLastStepHaltingNumber(ln) == 0:
                    continue
                for vid in traci.lane.getLastStepVehicleIDs(ln):
                    if traci.vehicle.isStopped(vid):          # xe buyt dang dung o tram theo lich: khong phai ket xe
                        continue
                    w = traci.vehicle.getWaitingTime(vid)
                    if w >= tau:
                        acc += (w / tau) ** 2
                        n += 1
                        wmax = max(wmax, w)
            pen[tl] = float(min(self.w4 * acc / max(len(self.lanes[tl]), 1), self.stuck_cap))
            n_all += n
        return pen, n_all, wmax

    def _collect(self):
        """Tinh reward + phat ket xe + ghi thong ke sau moi buoc Delta_t."""
        wt, ql, vaver, thr = self._outer_metrics()
        rw, rq, rt = self.w1 * (self.prev_wt - wt), self.w2 * (self.prev_ql - ql), self.w3 * thr
        r = rw + rq + rt
        r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        self.pen, n_stuck, wmax = self._stuck_penalty()
        self.stats["r_wt"].append(rw); self.stats["r_ql"].append(rq); self.stats["r_thr"].append(rt)
        self.stats["r_base"].append(r)                                     # reward truoc phat (da cat)
        self.edge_n += 1
        for e in self.outer_edges:
            self.edge_halt[e] += traci.edge.getLastStepHaltingNumber(e)
        for tl in self.tl_ids:
            self.tl_stat[tl]["pen"] += self.pen[tl]
        self.stats["stuck_n"].append(n_stuck); self.stats["stuck_max"] = max(self.stats["stuck_max"], wmax)
        self.stats["pen"].append(float(np.mean(list(self.pen.values()))))
        self.prev_wt, self.prev_ql = wt, ql
        self.stats["wt"].append(wt); self.stats["ql"].append(ql)
        self.stats["speed"].append(vaver); self.stats["thr"] += thr
        self.stats["in_net"] = traci.vehicle.getIDCount()                  # xe con trong mang
        done = traci.simulation.getTime() >= self.T
        return r, vaver, done

    # -- dat pha
    def _set_green(self, tl, phase):
        if self.overridden[tl]:                                # dang o che do RYG tay (vang gia lap) -> tra ve chuong trinh goc
            traci.trafficlight.setProgram(tl, self.prog[tl])
            self.overridden[tl] = False
        traci.trafficlight.setPhase(tl, phase)
        traci.trafficlight.setPhaseDuration(tl, 10_000)
        self.cur_phase[tl] = phase

    def _set_yellow(self, tl):
        y = self.yellow_of[tl][self.cur_phase[tl]]
        if y is not None:
            traci.trafficlight.setPhase(tl, y)
            traci.trafficlight.setPhaseDuration(tl, 10_000)
        else:                                                  # tlLogic khong co pha vang di kem: tu tao (G/g -> y)
            st = traci.trafficlight.getRedYellowGreenState(tl)
            traci.trafficlight.setRedYellowGreenState(tl, st.replace("G", "y").replace("g", "y"))
            self.overridden[tl] = True
            self.n_override += 1

    def step(self, actions):
        """actions: {tl: chi so khe pha}. Thuc thi Delta_t giay."""
        target = {tl: self.greens[tl][a] for tl, a in actions.items()}
        switching = [tl for tl in self.tl_ids if target[tl] != self.cur_phase[tl]]
        for tl in self.tl_ids:
            ts = self.tl_stat[tl]
            ts["steps"] += 1
            ts["phase_steps"][target[tl]] += 1
            if tl in switching:
                ts["switches"] += 1
        for tl in switching:                                   # vang T_y giay
            self._set_yellow(tl)
        ty = self.ty if switching else 0
        self._advance(ty)
        for tl in switching:
            self._set_green(tl, target[tl])
        self._advance(self.dt - ty)
        for tl in self.tl_ids:
            self.green_time[tl] = (self.dt - ty) if tl in switching else self.green_time[tl] + self.dt
        r, vaver, done = self._collect()
        obs = {tl: self.observe(tl) for tl in self.tl_ids}
        # thuong chung (Eq.8) tru phat ket xe cua RIENG den do (phat nang hon thuong: stuck_cap > reward_clip)
        return obs, {tl: r - self.pen[tl] for tl in self.tl_ids}, done, vaver


# ----------------------------------------------------------------------------- PPO
class PPO:
    def __init__(self, net, a):
        self.net, self.a = net, a
        self.opt_a = torch.optim.Adam(net.actor.parameters(), lr=a.lr_actor)
        self.opt_c = torch.optim.Adam(net.critic.parameters(), lr=a.lr_critic)

    def clip_beta(self, vaver, it):
        a = self.a
        if a.fixed_clip:
            return a.eps
        if random.random() < a.eps_explore:                        # Eq.11: noi long rang buoc
            return a.base
        cpa = a.base * (V_TOP - vaver) / V_TOP                     # Eq.9
        cpb = a.base * (1 + 0.002 * it) ** -0.75                   # Eq.10
        return float(np.clip(cpa + cpb, 0.02, 0.5))

    def advantages(self, r, v, v_last):
        """GAE: A_t = delta_t + gamma*lam*A_{t+1}; lam=0 -> TD 1 buoc (Eq.14 cua Primary)."""
        T, adv, last = len(r), np.zeros(len(r), np.float32), 0.0
        v_ext = np.append(v, v_last)
        for t in reversed(range(T)):
            delta = r[t] + self.a.gamma * v_ext[t + 1] - v_ext[t]
            last = delta + self.a.gamma * self.a.lam * last
            adv[t] = last
        return adv, adv + v                                        # (advantage, return muc tieu)

    def update(self, traj, vaver, it):
        obs = torch.as_tensor(np.concatenate([t["obs"] for t in traj]))
        act = torch.as_tensor(np.concatenate([t["act"] for t in traj]))
        old_lp = torch.as_tensor(np.concatenate([t["lp"] for t in traj]))
        adv = torch.as_tensor(np.concatenate([t["adv"] for t in traj]))
        ret = torch.as_tensor(np.concatenate([t["ret"] for t in traj]))
        mask = torch.as_tensor(np.concatenate([t["mask"] for t in traj]))
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        n, la, lc, beta = len(obs), 0.0, 0.0, 0.2
        for _ in range(self.a.epochs):
            beta = self.clip_beta(vaver, it)
            idx = torch.randperm(n)
            for s in range(0, n, self.a.batch):
                b = idx[s:s + self.a.batch]
                d = self.net.dist(obs[b], mask[b])
                ratio = torch.exp(d.log_prob(act[b]) - old_lp[b])               # Eq.13
                l_clip = torch.min(ratio * adv_n[b],
                                   torch.clamp(ratio, 1 - beta, 1 + beta) * adv_n[b]).mean()  # Eq.12
                loss_a = -l_clip - self.a.ent * d.entropy().mean()              # gradient ascent
                self.opt_a.zero_grad(); loss_a.backward()
                nn.utils.clip_grad_norm_(self.net.actor.parameters(), self.a.grad_clip)
                self.opt_a.step()
                loss_c = ((ret[b] - self.net.value(obs[b])) ** 2).mean()        # Eq.15
                self.opt_c.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(self.net.critic.parameters(), self.a.grad_clip)
                self.opt_c.step()
                la, lc = l_clip.item(), loss_c.item()
        return la, lc, beta


# ----------------------------------------------------------------------------- chay
def run_episode(env, net, ppo=None, greedy=False, fixed=False, seed=None, trace=None):
    obs = env.reset(control=not fixed, seed=seed)
    tls = env.tl_ids
    buf = {tl: dict(obs=[], act=[], lp=[], r=[], v=[], mask=[]) for tl in tls}
    done, total, vaver_hist = False, 0.0, []
    while not done:
        if fixed:                                   # baseline: den chay chuong trinh goc cua mang
            env._advance(env.dt)
            r, vaver, done = env._collect()
            rew = {tl: r - env.pen[tl] for tl in tls}
        else:
            actions = {}
            for tl in tls:
                o = torch.as_tensor(obs[tl]).unsqueeze(0)
                m = env.action_mask(tl)
                with torch.no_grad():
                    d = net.dist(o, torch.as_tensor(m).unsqueeze(0))
                    a = d.probs.argmax(-1) if greedy else d.sample()
                    buf[tl]["obs"].append(obs[tl]); buf[tl]["act"].append(a.item())
                    buf[tl]["lp"].append(d.log_prob(a).item()); buf[tl]["v"].append(net.value(o).item())
                    buf[tl]["mask"].append(m)
                actions[tl] = a.item()
            obs, rew, done, vaver = env.step(actions)
            for tl in tls:
                buf[tl]["r"].append(rew[tl])
        if trace is not None:                      # vet pha tung den: (thoi gian, den, chi so pha)
            t_now = traci.simulation.getTime()
            for tl in tls:
                trace.append((t_now, tl, traci.trafficlight.getPhase(tl) if fixed else env.cur_phase[tl]))
        vaver_hist.append(env.stats["speed"][-1])
        total += float(np.mean([rew[tl] for tl in tls]))
    traj = []
    if ppo is not None:
        for tl in tls:
            b = buf[tl]
            with torch.no_grad():
                v_last = net.value(torch.as_tensor(obs[tl]).unsqueeze(0)).item()   # bootstrap
            adv, ret = ppo.advantages(np.array(b["r"], np.float32), np.array(b["v"], np.float32), v_last)
            traj.append(dict(obs=np.array(b["obs"], np.float32), act=np.array(b["act"], np.int64),
                             lp=np.array(b["lp"], np.float32), adv=adv, ret=ret,
                             mask=np.array(b["mask"], dtype=bool)))
    s = env.stats
    n_steps = max(len(s["pen"]), 1)
    tl_sum = {}
    for t in tls:
        ts = env.tl_stat[t]
        d = {"phat_TB_moi_buoc": ts["pen"] / n_steps}
        if not fixed:                                   # baseline khong qua step() nen khong co so lieu doi pha
            d["so_lan_doi_pha"] = ts["switches"]
            for p, c in ts["phase_steps"].items():
                d[f"thoi_gian_pha_{p}_%"] = 100.0 * c / max(ts["steps"], 1)
        tl_sum[t] = d
    summary = dict(reward=total, awt=np.mean(s["wt"]), aql=np.mean(s["ql"]),
                   speed=np.mean(s["speed"]), thr=s["thr"], tele=s["tele"],
                   departed=s["departed"], arrived=s["arrived"], in_net=s["in_net"],
                   stuck_n=float(np.mean(s["stuck_n"])) if s["stuck_n"] else 0.0,
                   stuck_max=s["stuck_max"], pen=float(np.mean(s["pen"])) if s["pen"] else 0.0,
                   arr_rate=100.0 * s["arrived"] / max(s["departed"], 1),
                   reward_base=float(np.sum(s["r_base"])), reward_pen=float(np.sum(s["pen"])),
                   r_wt=float(np.sum(s["r_wt"])), r_ql=float(np.sum(s["r_ql"])), r_thr=float(np.sum(s["r_thr"])),
                   edge_in={e: len(env.edge_veh[e]) for e in env.outer_edges if e in env.outer_in},
                   edge_out={e: len(env.edge_veh[e]) for e in env.outer_edges if e in env.outer_out},
                   edge_other={e: len(env.edge_veh[e]) for e in env.outer_edges
                               if e not in env.outer_in and e not in env.outer_out},
                   edge_stop={e: env.edge_halt[e] / max(env.edge_n, 1) for e in env.outer_edges},
                   tl=tl_sum)
    return summary, traj, float(np.mean(vaver_hist))


def evaluate(env, net, seeds, fixed=False):
    """Chay greedy (hoac baseline neu fixed=True) tren cac seed co dinh. Tra ve list summary."""
    return [run_episode(env, net, greedy=True, fixed=fixed, seed=s)[0] for s in seeds]


STUCK_FLOOR = 0.1      # xe: mau so toi thieu khi baseline gan nhu khong ket (tranh chia cho ~0)


def select_score(rows, base_rows, alpha):
    """Diem chon best.pt (THAP hon = TOT hon), chuan hoa theo baseline (dieu khien goc, cung seed eval):
        score = alpha * stuck/stuck_base + (1 - alpha) * AWT/AWT_base
    stuck = so xe ket TB moi buoc (xe dung lien tuc >= tau s); stuck * T ti le voi TONG THOI GIAN KET (xe-giay).
    alpha = 1: chi theo ket xe (AWT chi de phan dinh khi bang nhau). alpha [T], xem GHI_CHU."""
    st = float(np.mean([r["stuck_n"] for r in rows]))
    aw = float(np.mean([r["awt"] for r in rows]))
    st0 = max(float(np.mean([r["stuck_n"] for r in base_rows])), STUCK_FLOOR) if base_rows else 1.0
    aw0 = max(float(np.mean([r["awt"] for r in base_rows])), 1e-6) if base_rows else 1.0
    return alpha * st / st0 + (1 - alpha) * aw / aw0 + 1e-6 * aw / aw0, st, aw


METRICS = [("awt", "AWT (s)"), ("aql", "AQL"), ("speed", "toc do (m/s)"),
           ("thr", "xe thoat ra"), ("departed", "xe vao mang"), ("arrived", "xe den dich"),
           ("arr_rate", "ti le den dich (%)"),
           ("in_net", "xe con trong mang"), ("tele", "teleport"),
           ("stuck_n", "xe ket (TB/buoc)"), ("stuck_max", "ket lau nhat (s)"), ("pen", "phat ket xe TB"),
           ("reward", "tong reward"), ("reward_pen", "tong phat")]


PROGRESS_COLS = ["ep", "train_AWT", "train_AQL", "train_speed", "train_arrived", "train_reward", "train_stuck", "train_pen",
                 "eval_AWT", "eval_AQL", "eval_speed", "eval_departed", "eval_arrived", "eval_stuck", "eval_pen",
                 "base_AWT", "base_arrived", "best_AWT", "best_ep", "is_best", "weight", "eval_score", "best_score"]


def plot_progress(csv_path, png_path, quiet=False):
    """Ve do thi phat trien tu progress.csv (moi diem = 1 EPISODE). Bo qua neu thieu matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker  # noqa: F401
    except ImportError:
        if not quiet:
            print("  (chua cai matplotlib -> bo qua ve do thi; pip install matplotlib roi chay: python plot_progress.py)")
        return
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    if "ep" not in rows[0]:
        print("  progress.csv o dinh dang cu (theo khoi) -> bo qua ve; train lai de co dinh dang moi (moi episode)")
        return

    def col(k):
        return [float(r[k]) if r.get(k) not in ("", None) else float("nan") for r in rows]

    def line(ax, x, y, *args, **kw):                      # bo diem NaN (episode khong eval) nhung van noi cac diem co
        pts = [(u, v) for u, v in zip(x, y) if v == v]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], *args, **kw)

    x = col("ep")
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    a0, a1, a2, a3 = ax.ravel()
    line(a0, x, col("train_AWT"), ".-", color="tab:blue", alpha=0.45, label="train (lay mau ngau nhien)")
    line(a0, x, col("eval_AWT"), "s-", color="tab:orange", label="eval (greedy, seed co dinh)")
    line(a0, x, col("best_AWT"), "-", color="tab:green", lw=2, label="best.pt hien tai (eval)")
    line(a0, x, col("base_AWT"), "k:", label="baseline den co dinh")
    stars = [(float(r["ep"]), float(r["eval_AWT"] if r["eval_AWT"] != "" else r["train_AWT"]))
             for r in rows if r.get("is_best") == "1"]
    if stars:
        a0.plot([p[0] for p in stars], [p[1] for p in stars], "*", color="red", ms=13, label="episode luu best.pt")
    a0.set_title("Thoi gian cho TB (AWT, s) - thap hon la tot hon")
    line(a1, x, col("train_arrived"), ".-", color="tab:blue", alpha=0.45, label="train")
    line(a1, x, col("eval_arrived"), "s-", color="tab:orange", label="eval")
    line(a1, x, col("base_arrived"), "k:", label="baseline")
    a1.set_title("Xe den dich")
    line(a2, x, col("train_speed"), ".-", color="tab:blue", alpha=0.45, label="train")
    line(a2, x, col("eval_speed"), "s-", color="tab:orange", label="eval")
    a2.set_title("Toc do TB (m/s)")
    line(a3, x, col("train_stuck"), ".-", color="tab:blue", alpha=0.45, label="train")
    line(a3, x, col("eval_stuck"), "s-", color="tab:orange", label="eval")
    a3.set_title("Xe ket TB moi buoc (thap hon la tot hon)")
    for a_ in (a0, a1, a2, a3):
        a_.set_xlabel("episode")
        a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(png_path, dpi=120)
        if not quiet:
            print(f"  da ve {png_path}")
    except OSError as ex:                                 # vd file dang mo trong trinh xem anh (Windows)
        print(f"  khong ghi duoc {png_path}: {ex}")
    plt.close(fig)


# ---- best_metrics.csv: moi chi so cua best.pt
SCALAR_ROWS = [
    ("awt", "tong_quan", "AWT: thoi gian cho TB cua xe tren canh bien (s)"),
    ("aql", "tong_quan", "AQL: hang cho TB tren canh bien (xe quy doi)"),
    ("speed", "tong_quan", "toc do TB (m/s)"),
    ("departed", "tong_quan", "xe vao mang"),
    ("arrived", "tong_quan", "xe den dich"),
    ("arr_rate", "tong_quan", "ti le den dich = den dich / vao mang (%)"),
    ("thr", "tong_quan", "xe thoat ra (dem moi delta_t, thuong thieu)"),
    ("in_net", "tong_quan", "xe con trong mang cuoi episode"),
    ("tele", "tong_quan", "xe bi teleport"),
    ("stuck_n", "ket_xe", "xe ket TB moi buoc"),
    ("stuck_max", "ket_xe", "xe ket lau nhat (s)"),
    ("reward", "reward", "tong reward (da tru phat)"),
    ("reward_base", "reward", "tong reward truoc phat (da cat +-reward_clip)"),
    ("r_wt", "reward", "thanh phan w1*(WT_prev-WT), chua cat"),
    ("r_ql", "reward", "thanh phan w2*(QL_prev-QL), chua cat"),
    ("r_thr", "reward", "thanh phan w3*thr, chua cat"),
    ("reward_pen", "phat", "tong phat ket xe (TB cac den)"),
    ("pen", "phat", "phat ket xe TB moi buoc"),
]
BEST_HPARAMS = ["w1", "w2", "w3", "w4", "stuck_wait", "reward_clip", "stuck_cap", "gamma", "lam", "ent",
                "lr_actor", "lr_critic", "eps", "base", "epochs", "batch", "delta_t", "yellow", "min_green",
                "max_green", "episode_len", "eval_seeds", "seed"]


def _flat_summary(sm):
    d = {}
    for k, g, label in SCALAR_ROWS:
        if k in sm:
            d[(g, label)] = float(sm[k])
    for g, key in (("xe_vao_canh", "edge_in"), ("xe_ra_canh", "edge_out"), ("xe_qua_canh_khac", "edge_other"),
                   ("xe_dung_TB_canh", "edge_stop")):
        for e, v in sm.get(key, {}).items():
            d[(g, e)] = float(v)
    for tl, dd in sm.get("tl", {}).items():
        for k, v in dd.items():
            d[(f"den {tl}", k)] = float(v)
    return d


def write_best_metrics(path, ep, eval_seeds, eval_rows, train_res, base_rows, best_awt, a, best_score=None):
    """Ghi MOI chi so cua best.pt vao 1 file CSV rieng (ghi de moi khi co best moi).
    Cot: seed eval, TB, do lech chuan, episode train luc luu, baseline (cung seed eval)."""
    ev = [_flat_summary(r) for r in (eval_rows or [])]
    tr = _flat_summary(train_res)
    bs = [_flat_summary(r) for r in (base_rows or [])]
    keys = list(dict.fromkeys(k for d in ev + [tr] + bs for k in d))

    def rnd(v):
        return "" if v is None else round(float(v), 4)

    head = (["nhom", "chi_so"] + [f"eval_seed_{s_}" for s_ in eval_seeds]
            + ["eval_TB", "eval_std", f"train_ep{ep}", "baseline_TB"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        pad = [""] * len(eval_seeds)
        w.writerow(["meta", "best_episode"] + pad + [ep, "", "", ""])
        w.writerow(["meta", "eval_AWT_cua_best"] + pad + [rnd(best_awt), "", "", ""])
        w.writerow(["meta", f"diem_chon_best = {a.select_alpha}*ket_xe/ket_xe_base + {1 - a.select_alpha:.2f}*AWT/AWT_base (thap hon = tot hon)"]
                   + pad + [rnd(best_score), "", "", ""])
        for n in BEST_HPARAMS:
            if hasattr(a, n):
                w.writerow(["tham_so", n] + pad + [getattr(a, n), "", "", ""])
        for k in keys:
            vals = [d.get(k) for d in ev]
            nums = [v for v in vals if v is not None]
            bn = [d[k] for d in bs if k in d]
            w.writerow([k[0], k[1]] + [rnd(v) for v in vals]
                       + [rnd(np.mean(nums)) if nums else "", rnd(np.std(nums)) if nums else "",
                          rnd(tr.get(k)), rnd(np.mean(bn)) if bn else ""])


def fmt(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return f"{v.mean():8.1f} +- {v.std():6.1f}"


def load_net(env, path):
    net = ActorCritic()
    net.load_state_dict(torch.load(path))
    return net


def comparison_report(env, nets, seeds, csv_path=None):
    """Chay baseline + tung model tren CUNG cac seed (cung luu luong). In bang va ghi CSV."""
    results = {"BASELINE": evaluate(env, None, seeds, fixed=True)}
    for name, net in nets.items():
        results[name] = evaluate(env, net, seeds)
    names = list(results)
    print(f"\nTren {len(seeds)} seed co dinh (cung luu luong cho moi cot), trung binh +- do lech chuan")
    print(f"{'chi so':18s}" + "".join(f" | {n:>20s}" for n in names))
    for k, label in METRICS:
        print(f"{label:18s}" + "".join(f" | {fmt(results[n], k):>20s}" for n in names))
    if csv_path:
        keys = [k for k, _ in METRICS]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "seed"] + keys)
            for n in names:
                for s, r in zip(seeds, results[n]):
                    w.writerow([n, s] + [round(float(r[k]), 3) for k in keys])
                w.writerow([n, "mean"] + [round(float(np.mean([r[k] for r in results[n]])), 3) for k in keys])
                w.writerow([n, "std"] + [round(float(np.std([r[k] for r in results[n]])), 3) for k in keys])
    return results


# ----------------------------------------------------------------------------- luu ket qua
def _rel(path, base):
    try:
        return os.path.relpath(path, base).replace("\\", "/")
    except ValueError:                                  # khac o dia tren Windows
        return os.path.abspath(path).replace("\\", "/")


def write_sumocfg(a, env, out_dir):
    """trained.sumocfg: mang + luu luong demo + cau hinh dung voi luc train.
    Luu y: file .sumocfg khong nap duoc weight; weight duoc nap boi run_trained.py."""
    cdir = os.path.dirname(os.path.abspath(a.cfg))
    root = ET.parse(a.cfg).getroot()

    def vals(tag):
        node = root.find(f"./input/{tag}")
        if node is None:
            return []
        return [os.path.join(cdir, v.strip()) for v in node.get("value").split(",") if v.strip()]

    net_file = os.path.abspath(a.net) if a.net else vals("net-file")[0]
    if env.demand is not None:                          # luu luong demo co dinh (seed danh gia dau tien)
        demo = os.path.join(out_dir, "demo.rou.xml")
        env._gen_routes(demo, EVAL_SEED_BASE)
        routes = [demo]
    else:
        routes = vals("route-files")
    extra = vals("additional-files")
    gui = vals("gui-settings-file")

    lat_node = root.find("./processing/lateral-resolution")      # xe may chia lan ngang: phai giong luc train
    lat = f'<lateral-resolution value="{lat_node.get("value")}"/>' if lat_node is not None else ""
    lines = ["<configuration>",
             "  <!-- Mang + luu luong dung luc train. Chay thuan: den chay chuong trinh goc cua mang.",
             "       De dieu khien den bang model best hoac last: dung script run_trained.py (xem huong dan trong file do) -->",
             "  <input>",
             f'    <net-file value="{_rel(net_file, out_dir)}"/>',
             f'    <route-files value="{",".join(_rel(r, out_dir) for r in routes)}"/>']
    if extra:
        lines.append(f'    <additional-files value="{",".join(_rel(r, out_dir) for r in extra)}"/>')
    if gui:
        lines.append(f'    <gui-settings-file value="{_rel(gui[0], out_dir)}"/>')
    lines += ["  </input>",
              f'  <time><begin value="0"/><end value="{a.episode_len}"/></time>',
              f'  <processing><time-to-teleport value="-1"/><waiting-time-memory value="{a.wt_memory}"/>{lat}</processing>',
              '  <output><tripinfo-output value="trained_tripinfo.xml"/>'
              '<statistic-output value="trained_stats.xml"/></output>',
              "</configuration>\n"]
    path = os.path.join(out_dir, "trained.sumocfg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_meta(a, env, out_dir, best_ep, best_awt):
    meta = dict(model="phase-slot shared (topology independent)", feat_dim=N_FEAT, trained_on=dict(
                    tls_ids=env.tl_ids, outer_in=env.outer_in, outer_out=env.outer_out,
                    greens={t: env.greens[t] for t in env.tl_ids}),
                delta_t=a.delta_t, yellow=a.yellow, min_green=a.min_green, max_green=a.max_green,
                wt_memory=a.wt_memory, episode_len=a.episode_len,
                stuck=dict(w4=a.w4, stuck_wait=a.stuck_wait, reward_clip=a.reward_clip, stuck_cap=a.stuck_cap),
                episodes=a.episodes, best_episode=best_ep, best_eval_awt=best_awt,
                weights=dict(best="best.pt", last="last.pt"), sumocfg="trained.sumocfg")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="test2.sumocfg")
    p.add_argument("--net", default=None, help="file .net.xml (mac dinh doc tu .sumocfg)")
    p.add_argument("--tls", nargs="+", default=None, help="(tuy chon) chi dieu khien cac den nay; mac dinh: TAT CA den trong mang")
    p.add_argument("--outer-in", nargs="+", default=None, help="(tuy chon) ghi de canh vao do reward; mac dinh: tu nhan o bien mang")
    p.add_argument("--outer-out", nargs="+", default=None, help="(tuy chon) ghi de canh ra do reward")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--episode-len", type=int, default=7200)
    p.add_argument("--delta-t", type=int, default=10)
    p.add_argument("--yellow", type=int, default=3)
    p.add_argument("--min-green", type=int, default=5)
    p.add_argument("--max-green", type=int, default=60)
    p.add_argument("--wt-memory", type=int, default=300)
    # --- 5 muc uu tien
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--fixed-clip", action="store_true", help="PPO goc, khong dung clip thich nghi")
    p.add_argument("--base", type=float, default=0.2)
    p.add_argument("--eps-explore", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--w1", type=float, default=0.05)
    p.add_argument("--w2", type=float, default=0.5)
    p.add_argument("--w3", type=float, default=0.0)
    # --- phat ket xe o ngan vao den (nang hon thuong)
    p.add_argument("--w4", type=float, default=2.0, help="he so phat ket xe (0 = tat)")
    p.add_argument("--stuck-wait", type=float, default=60.0, help="tau: xe dung lien tuc >= tau giay moi bi tinh la ket")
    p.add_argument("--reward-clip", type=float, default=10.0, help="gioi han |thuong| moi buoc (Eq.8)")
    p.add_argument("--stuck-cap", type=float, default=30.0, help="gioi han phat moi buoc; phai > reward-clip")
    # --- luu luong
    p.add_argument("--demand", choices=["file", "random", "cfg"], default="file",
                   help="file: LAY SO LIEU TU FILE ROUTE cua .sumocfg (giu nguyen luu luong, loai xe, 2 giai doan), "
                        "seed chi lam ngau nhien thoi diem xe; random: luu luong ngau nhien cua gen_demand; "
                        "cfg: chay thang file route (khong ngau nhien)")
    p.add_argument("--rate-jitter", type=float, default=0.1,
                   help="(--demand file) moi episode, luu luong tung flow duoc nhan ngau nhien trong [1-j, 1+j] "
                        "(sinh them / bot xe quanh so lieu trong file; 0 = dung chinh xac, 0.1 = +-10%%)")
    p.add_argument("--mix-jitter", type=float, default=0.0,
                   help="(--demand file) ti le loai xe trong vTypeDistribution lech +-j (0 = chinh xac)")
    p.add_argument("--no-poisson", action="store_true",
                   help="(--demand file) xe xuat hien deu nhu file goc thay vi Poisson (cung luu luong trung binh)")
    p.add_argument("--slice", type=int, default=600, help="do dai moi khoang thoi gian (giay)")
    p.add_argument("--rate-lo", type=float, default=150.0, help="xe/gio moi canh nguon (muc x1)")
    p.add_argument("--rate-hi", type=float, default=500.0)
    p.add_argument("--level-lo", type=float, default=0.4, help="he so luu luong thap nhat moi khoang")
    p.add_argument("--level-hi", type=float, default=1.6)
    p.add_argument("--origin-weight", default="", help="vd: E0=2,-E0=1 (canh nguon nao dong hon)")
    # --- danh gia va ket qua
    p.add_argument("--out-dir", default="results", help="thu muc luu best.pt, last.pt, trained.sumocfg, bao cao")
    p.add_argument("--block", type=int, default=10, help="moi bao nhieu episode luu 1 checkpoint trong so (progress.csv/png van cap nhat MOI episode)")
    p.add_argument("--eval-every", type=int, default=1, help="0 = tat danh gia trong luc train")
    p.add_argument("--select-alpha", type=float, default=0.7,
                   help="trong so ket xe khi chon best.pt: diem = a*ket_xe/base + (1-a)*AWT/base (1 = chi ket xe, 0 = chi AWT)")
    p.add_argument("--eval-seeds", type=int, default=3, help="so seed cho moi lan danh gia trong luc train")
    p.add_argument("--final-seeds", type=int, default=5, help="so seed cho bao cao cuoi (0 = bo qua)")
    p.add_argument("--eval", action="store_true", help="khong train: so baseline voi weight da luu")
    p.add_argument("--weights", default="both", help="dung voi --eval: best | last | both | duong dan .pt")
    p.add_argument("--eval-episodes", type=int, default=5, help="so seed khi dung --eval")
    p.add_argument("--init", default=None, help="tiep tuc train tu weight .pt (vd train tiep tren mang khac)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.w4 > 0 and a.stuck_cap <= a.reward_clip:
        raise SystemExit(f"--stuck-cap ({a.stuck_cap}) phai lon hon --reward-clip ({a.reward_clip}) de phat nang hon thuong")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    best_path = os.path.join(a.out_dir, "best.pt")
    last_path = os.path.join(a.out_dir, "last.pt")

    demand = None
    if a.demand == "file":
        demand = dict(mode="file", route_files=gen_demand.route_files_from_cfg(a.cfg), rate_jitter=a.rate_jitter,
                      poisson=not a.no_poisson, mix_jitter=a.mix_jitter)
    elif a.demand == "random":
        demand = dict(slice_len=a.slice, rate=(a.rate_lo, a.rate_hi), level=(a.level_lo, a.level_hi),
                      weights=gen_demand.parse_weights(a.origin_weight))
    env = MultiJunctionEnv(a.cfg, a.tls, a.outer_in, a.outer_out, a.gui, a.delta_t, a.yellow,
                           a.episode_len, a.w1, a.w2, a.w3, a.seed, "sim", a.min_green, a.max_green,
                           a.wt_memory, demand, a.net, a.w4, a.stuck_wait, a.reward_clip, a.stuck_cap)
    env.reset(seed=a.seed)
    net = ActorCritic()
    if a.init:
        net.load_state_dict(torch.load(a.init))
        print(f"khoi tao tu weight: {a.init}")
    print(f"TU NHAN: {len(env.tl_ids)} den dieu khien: " + "; ".join(
        f"{t} (pha xanh {env.greens[t]}, {len(env.lanes[t])} lan vao, "
        f"vang {'tu tlLogic' if all(v is not None for v in env.yellow_of[t].values()) else 'TU TAO'})" for t in env.tl_ids))
    print(f"canh bien do reward: vao={env.outer_in} | ra={env.outer_out}")
    if env.demand_info and env.demand_info.get("mode") == "file":
        di = env.demand_info
        print(f"luu luong TU FILE ROUTE: {di['routes']} flow, loai xe {di['types']}, "
              f"jitter luu luong +-{a.rate_jitter}, ti le loai xe +-{a.mix_jitter}, "
              f"{'Poisson' if not a.no_poisson else 'deu nhu file goc'}")
        print("  tong luu luong theo giai doan (xe/gio): " + " | ".join(f"{b:.0f}-{e:.0f}s: {v}" for b, e, v in di["slices"]))
    elif env.demand_info:
        print(f"luu luong ngau nhien: {len(env.demand_info['sources'])} canh nguon "
              f"{env.demand_info['sources']} -> {len(env.demand_info['sinks'])} canh dich "
              f"{env.demand_info['sinks']}")
    if env.demand is None:
        print("luu luong: dung file route trong .sumocfg (khong ngau nhien)")

    # ------------------------------------------------------------- che do --eval
    if a.eval:
        seeds = [EVAL_SEED_BASE + i for i in range(a.eval_episodes)]
        nets = {}
        if a.weights in ("best", "both"):
            nets["PPO best"] = load_net(env, best_path)
        if a.weights in ("last", "both"):
            nets["PPO last"] = load_net(env, last_path)
        if a.weights not in ("best", "last", "both"):
            nets["PPO " + os.path.basename(a.weights)] = load_net(env, a.weights)
        comparison_report(env, nets, seeds, os.path.join(a.out_dir, "eval_report.csv"))
        env.close(); return

    # ------------------------------------------------------------- train
    ppo = PPO(net, a)
    eval_seeds = [EVAL_SEED_BASE + i for i in range(a.eval_seeds)]
    log_path = os.path.join(a.out_dir, "train_log.csv")
    best, best_ep, best_score = float("inf"), -1, float("inf")      # best = EVAL AWT cua model best
    ck_dir = os.path.join(a.out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    prog_path = os.path.join(a.out_dir, "progress.csv")
    base_rows = evaluate(env, None, eval_seeds, fixed=True) if a.eval_every else None
    base_awt = float(np.mean([r["awt"] for r in base_rows])) if base_rows else ""
    base_arr = float(np.mean([r["arrived"] for r in base_rows])) if base_rows else ""
    eval_rows = None
    ck_every = max(a.block, 1)                           # --block chi con dung cho tan suat luu checkpoint trong so
    with open(prog_path, "w", newline="", encoding="utf-8") as pf:
        csv.writer(pf).writerow(PROGRESS_COLS)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "AWT", "AQL", "avg_speed", "xe_thoat_ra", "xe_vao_mang", "xe_den_dich",
                    "xe_con_trong_mang", "teleports", "xe_ket_TB", "ket_lau_nhat_s", "phat_ket_TB", "reward", "actor_obj", "critic_loss", "beta", "eval_AWT"])
        for ep in range(a.episodes):
            t0 = time.time()
            res, traj, vaver = run_episode(env, net, ppo)
            la, lc, beta = ppo.update(traj, vaver, ep)
            torch.save(net.state_dict(), last_path)
            eval_awt, eval_rows, is_best, ev_score, ev_stuck = "", None, False, "", ""
            if a.eval_every and ((ep + 1) % a.eval_every == 0 or ep == a.episodes - 1):
                eval_rows = evaluate(env, net, eval_seeds)
                eval_awt = float(np.mean([r["awt"] for r in eval_rows]))
                ev_score, ev_stuck, _ = select_score(eval_rows, base_rows, a.select_alpha)
                if ev_score < best_score:                # chon theo diem (ket xe truoc, roi AWT), thap hon = tot hon
                    best, best_ep, is_best, best_score = eval_awt, ep, True, ev_score
                    torch.save(net.state_dict(), best_path)
            elif not a.eval_every:                       # khong danh gia -> best = last
                best, best_ep, is_best = res["awt"], ep, True
                torch.save(net.state_dict(), best_path)
            if is_best:                                  # ghi MOI chi so cua best.pt (ghi de moi khi co best moi)
                write_best_metrics(os.path.join(a.out_dir, "best_metrics.csv"), ep,
                                   eval_seeds if eval_rows else [], eval_rows, res, base_rows, best, a, best_score)
            w.writerow([ep, res["awt"], res["aql"], res["speed"], res["thr"], res["departed"],
                        res["arrived"], res["in_net"], res["tele"], res["stuck_n"], res["stuck_max"], res["pen"], res["reward"], la, lc, beta, eval_awt])
            f.flush()
            wfile = ""
            if (ep + 1) % ck_every == 0 or ep == a.episodes - 1:
                wfile = os.path.join("checkpoints", f"ep{ep:04d}.pt")
                torch.save(net.state_dict(), os.path.join(a.out_dir, wfile))

            def em(k):
                return round(float(np.mean([r[k] for r in eval_rows])), 3) if eval_rows else ""
            with open(prog_path, "a", newline="", encoding="utf-8") as pf:       # 1 dong MOI episode
                csv.writer(pf).writerow([
                    ep, round(res["awt"], 3), round(res["aql"], 3), round(res["speed"], 3), res["arrived"],
                    round(res["reward"], 3), round(res["stuck_n"], 3), round(res["pen"], 3),
                    em("awt"), em("aql"), em("speed"), em("departed"), em("arrived"), em("stuck_n"), em("pen"),
                    round(base_awt, 3) if base_rows else "", round(base_arr, 1) if base_rows else "",
                    round(best, 3) if best != float("inf") else "", best_ep, int(is_best), wfile,
                    round(ev_score, 4) if ev_score != "" else "", round(best_score, 4) if best_score != float("inf") else ""])
            plot_progress(prog_path, os.path.join(a.out_dir, "progress.png"), quiet=True)   # ve lai SAU MOI episode
            extra = (f" | EVAL AWT {eval_awt:6.1f}s ket {ev_stuck:4.1f} diem {ev_score:.3f}") if eval_awt != "" else ""
            extra += "  <- BEST moi" if is_best else ""
            print(f"ep {ep:3d} | AWT {res['awt']:6.1f}s | AQL {res['aql']:5.2f} | v {res['speed']:5.2f} m/s | "
                  f"vao {res['departed']:5d} | den {res['arrived']:5d} | thoat {res['thr']:5d} | "
                  f"con {res['in_net']:4d} | tele {res['tele']:3d} | ket {res['stuck_n']:4.1f} (max {res['stuck_max']:4.0f}s) "
                  f"| phat {res['pen']:5.2f} | {time.time()-t0:.0f}s{extra}")

    # ------------------------------------------------------------- sau khi train xong
    torch.save(net.state_dict(), last_path)
    if not os.path.exists(best_path):
        torch.save(net.state_dict(), best_path)
    cfg_path = write_sumocfg(a, env, a.out_dir)
    write_meta(a, env, a.out_dir, best_ep, None if best == float("inf") else best)
    if a.final_seeds > 0:
        seeds = [EVAL_SEED_BASE + i for i in range(a.final_seeds)]
        comparison_report(env, {"PPO best": load_net(env, best_path), "PPO last": load_net(env, last_path)},
                          seeds, os.path.join(a.out_dir, "final_report.csv"))
    env.close()
    print(f"\nTRAIN XONG. Ket qua trong thu muc: {os.path.abspath(a.out_dir)}")
    print(f"  best.pt   : model tot nhat (episode {best_ep}, EVAL AWT {best:.1f}s, diem {best_score:.3f})")
    print(f"  last.pt   : model sau episode cuoi ({a.episodes - 1})")
    print(f"  {os.path.basename(cfg_path)} + meta.json : cau hinh de chay lai")
    print(f"  train_log.csv, final_report.csv : so lieu tung episode va bao cao cuoi")
    print(f"  progress.csv, progress.png : 1 dong / 1 do thi cap nhat MOI episode; checkpoints/ : trong so moi {a.block} episode")
    print(f"  best_metrics.csv : moi chi so cua best.pt (reward, phat, xe vao/ra tung canh, tung den, so voi baseline)")
    print(f"  Chay thu: python run_trained.py --dir {a.out_dir} --weights best --gui")


if __name__ == "__main__":
    main()
