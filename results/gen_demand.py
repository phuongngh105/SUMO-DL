"""
Sinh luu luong xe NGAU NHIEN THEO TUNG KHOANG THOI GIAN cho SUMO.

Y tuong
-------
- Chia thoi gian mo phong thanh cac "slice" (vd 600 s).
- Moi slice boc ngau nhien 1 "muc luu luong" (level): thap diem / binh thuong / cao diem...
- Moi canh nguon (o bien mang) co luu luong rieng = rate * level * trong_so_canh,
  chia ngau nhien cho cac canh dich (o bien mang) co duong di toi duoc.
- Moi flow dung `probability` (xac suat xuat hien moi giay) nen thoi diem xe sinh ra la ngau nhien.
- Cung seed -> cung file route (de tai lap va so sanh PPO voi baseline tren cung luu luong).

Xe KHONG QUAY LAI (ban sua)
---------------------------
- Moi cap nguon -> dich co 1 ROUTE TUONG MINH (danh sach canh) ghi san trong file route.
  Xe di het route o bien ben kia thi bien mat o cuoi canh dich, khong con route nao dua no quay lai.
  (Ban cu dung flow from/to nen SUMO tu tim duong, co the cho xe quay dau giua mang.)
- Duong di duoc tim bang Dijkstra rieng, CAM quay dau (U-turn) o moi nut.
- Loai cap nguon -> dich nam CUNG PHIA (xe vao tu phia nao thi khong thoat ra o chinh phia do),
  ke ca khi co duong vong qua mot khoi nha.

CHE DO "FILE" (from_route_files) - uu tien SO LIEU TRONG FILE ROUTE CUA BAN
-------------------------------------------------------------------------
Doc cac file route trong .sumocfg va GIU NGUYEN: vType, vTypeDistribution (ti le xe may / o to / bus),
tung flow (from, to, via, begin, end, type, luu luong xe/gio). Seed chi lam ngau nhien:
  - thoi diem xe xuat hien: kieu Poisson (period="exp(lambda)") voi CUNG luu luong trung binh   [poisson=True]
  - (tuy chon) rate_jitter = 0.1 -> luu luong tung flow lech ngau nhien +-10% quanh gia tri trong file
  - (tuy chon) mix_jitter  = 0.1 -> ti le loai xe trong vTypeDistribution lech +-10% (van chuan hoa tong)
from_route_files mac dinh jitter = 0 (chinh xac); train_ppo_tsc.py dung --rate-jitter 0.1 (+-10%) de moi episode co them / bot xe quanh so lieu cua ban.

Dung doc lap:
    python gen_demand.py --net map.net.xml --out routes.rou.xml --seed 1 --end 3600 --slice 600
Hoac import:
    gen_demand.make_demand(net, "routes.rou.xml", seed=1, end=3600)
    gen_demand.from_route_files(gen_demand.route_files_from_cfg("x.sumocfg"), "out.rou.xml", seed=1)
"""
import argparse
import heapq
import math
import os
import random
import xml.etree.ElementTree as ET

import sumolib


def _is_road(e):
    return e.getFunction() in ("", "normal") and e.allows("passenger")


def boundary_edges(net):
    """(nguon, dich): canh bat dau / ket thuc o bien mang (nut cut hoac nut chi co 1 canh noi)."""
    srcs, sinks = [], []
    for e in net.getEdges():
        if not _is_road(e):
            continue
        frm, to = e.getFromNode(), e.getToNode()
        # bo qua canh nguoc chieu cua chinh no (duong 2 chieu)
        if not [x for x in frm.getIncoming() if x.getFromNode() != to and _is_road(x)]:
            srcs.append(e.getID())
        if not [x for x in to.getOutgoing() if x.getToNode() != frm and _is_road(x)]:
            sinks.append(e.getID())
    return srcs, sinks


def _is_uturn(a, b):
    """b la canh nguoc chieu cua a (di tu nut cuoi cua a quay lai nut dau cua a)."""
    return b.getToNode() == a.getFromNode() and b.getFromNode() == a.getToNode()


def _cost(e):
    return e.getLength() / max(e.getSpeed(), 0.1)


def routes_from(net, src_id, no_uturn=True):
    """Dijkstra tu 1 canh nguon, cam quay dau. Tra ve {id_canh_dich: [danh sach id canh]}."""
    src = net.getEdge(src_id)
    dist = {src: _cost(src)}
    prev = {}
    heap = [(dist[src], 0, src)]
    tie = 1
    while heap:
        d, _, e = heapq.heappop(heap)
        if d > dist.get(e, math.inf):
            continue
        for nxt in e.getOutgoing():                      # canh ke tiep (co ket noi that trong mang)
            if not _is_road(nxt) or (no_uturn and _is_uturn(e, nxt)):
                continue
            nd = d + _cost(nxt)
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = e
                heapq.heappush(heap, (nd, tie, nxt))
                tie += 1
    out = {}
    for e in dist:
        path, cur = [e.getID()], e
        while cur in prev:
            cur = prev[cur]
            path.append(cur.getID())
        out[e.getID()] = path[::-1]
    return out


def make_demand(net, out_file, seed=0, end=3600, slice_len=600, rate=(150.0, 500.0),
                level=(0.4, 1.6), weights=None, sources=None, sinks=None, no_uturn=True,
                vtype='<vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="13.89" sigma="0.5"/>'):
    """
    net      : doi tuong sumolib Net hoac duong dan file .net.xml
    rate     : (thap, cao) xe/gio tren MOI canh nguon o muc level = 1
    level    : (thap, cao) he so nhan luu luong, boc ngau nhien moi slice
    weights  : {edge_id: he so} de canh nao do dong hon (vd huong uu tien do > xanh)
    no_uturn : True = cam quay dau va cam thoat ra cung phia voi luc vao (mac dinh)
    Tra ve dict {"sources", "sinks", "slices", "routes", "dropped_same_side"}
    """
    if isinstance(net, str):
        net = sumolib.net.readNet(net)
    rng = random.Random(seed)
    weights = weights or {}
    auto_src, auto_sink = boundary_edges(net)
    srcs = list(sources) if sources else auto_src
    dsts = list(sinks) if sinks else auto_sink

    pairs, paths, dropped = {}, {}, 0
    for s in srcs:
        reach = routes_from(net, s, no_uturn)
        s_edge = net.getEdge(s)
        pairs[s] = []
        for d in dsts:
            if d == s or d not in reach:
                continue
            if no_uturn and net.getEdge(d).getToNode() == s_edge.getFromNode():
                dropped += 1                              # thoat ra cung phia voi luc vao -> bo
                continue
            pairs[s].append(d)
            paths[(s, d)] = reach[d]
    srcs = [s for s in srcs if pairs[s]]
    if not srcs:
        raise RuntimeError("Khong tim thay cap nguon-dich nao di duoc. Hay truyen --sources/--sinks "
                           "hoac dung --demand cfg de giu file route cua .sumocfg.")

    lines = ["<routes>", "  " + vtype]
    rid = {}
    for (s, d), edges in paths.items():                   # route tuong minh, dinh nghia 1 lan
        rid[(s, d)] = f"r{len(rid)}"
        lines.append(f'  <route id="{rid[(s, d)]}" edges="{" ".join(edges)}"/>')

    info, n = [], 0
    for k in range(int(math.ceil(end / slice_len))):
        b, e = k * slice_len, min((k + 1) * slice_len, end)
        lvl = rng.uniform(*level)
        info.append((b, e, round(lvl, 2)))
        for s in srcs:
            total = rng.uniform(*rate) * lvl * weights.get(s, 1.0)     # xe/gio cua canh nguon nay
            ws = [rng.random() + 0.2 for _ in pairs[s]]
            sw = sum(ws)
            for d, w in zip(pairs[s], ws):
                p = min(total * w / sw / 3600.0, 1.0)
                lines.append(f'  <flow id="f{n}" type="car" route="{rid[(s, d)]}" begin="{b}" end="{e}" '
                             f'probability="{p:.5f}" departLane="best" departSpeed="max"/>')
                n += 1
    lines.append("</routes>\n")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"sources": srcs, "sinks": dsts, "slices": info,
            "routes": len(paths), "dropped_same_side": dropped}


def route_files_from_cfg(cfg):
    """Cac file route trong <route-files> cua .sumocfg (duong dan tuyet doi)."""
    node = ET.parse(cfg).getroot().find("./input/route-files")
    if node is None:
        raise RuntimeError(f"{cfg} khong co <route-files>")
    d = os.path.dirname(os.path.abspath(cfg))
    return [os.path.join(d, v.strip()) for v in node.get("value").split(",") if v.strip()]


def from_route_files(route_files, out_file, seed=0, end=None, rate_jitter=0.1, poisson=True, mix_jitter=0.05):
    """Sinh file route moi TU file route cua ban: giu nguyen moi thu, chi doi cach xe xuat hien.
    rate_jitter: luu luong moi flow nhan (1 + U(-j, j)); mix_jitter: ti le loai xe trong vTypeDistribution lech +-j.
    poisson: True = xe xuat hien ngau nhien (Poisson) voi cung luu luong trung binh; False = deu nhu file goc.
    Tra ve dict cung khoa voi make_demand + 'mode': 'file'."""
    rng = random.Random(seed)
    if isinstance(route_files, str):
        route_files = [route_files]
    head, flows = [], []
    for rf in route_files:
        for el in ET.parse(rf).getroot():
            (flows if el.tag == "flow" else head).append(el)

    if mix_jitter > 0:
        for el in head:
            if el.tag == "vTypeDistribution":
                kids = [k for k in el if k.get("probability") is not None]
                ps = [float(k.get("probability")) for k in kids]
                new = [p * (1 + rng.uniform(-mix_jitter, mix_jitter)) for p in ps]
                sc = sum(ps) / max(sum(new), 1e-12)
                for k, p in zip(kids, new):
                    k.set("probability", f"{p * sc:.4f}")

    rows, per = [], {}
    for fl in sorted(flows, key=lambda x: float(x.get("begin", 0))):      # sap theo thoi diem bat dau (on dinh)
        a = dict(fl.attrib)
        b, e = float(a.get("begin", 0)), float(a.get("end", end if end is not None else 3600))
        rate = None                                                        # xe / giay
        for key in ("perHour", "vehsPerHour"):
            if key in a:
                rate = float(a.pop(key)) / 3600.0
        if rate is None and "probability" in a:
            rate = float(a.pop("probability"))
        if rate is None and "period" in a and not a["period"].startswith("exp"):
            rate = 1.0 / float(a.pop("period"))
        if rate is not None:
            for k in ("probability", "period", "perHour", "vehsPerHour"):
                a.pop(k, None)
            if rate_jitter > 0:
                rate *= 1 + rng.uniform(-rate_jitter, rate_jitter)
            if poisson:
                a["period"] = f"exp({rate:.8f})"
            else:
                a["perHour"] = f"{rate * 3600:.3f}"
            per[(b, e)] = per.get((b, e), 0.0) + rate * 3600
        rows.append(a)

    root = ET.Element("routes")
    for el in head:
        root.append(el)
    for a in rows:
        ET.SubElement(root, "flow", a)
    ET.indent(root)
    ET.ElementTree(root).write(out_file, encoding="utf-8", xml_declaration=True)
    return {"mode": "file", "sources": sorted({r["from"] for r in rows if "from" in r}),
            "sinks": sorted({r["to"] for r in rows if "to" in r}),
            "slices": [(b, e, round(v)) for (b, e), v in sorted(per.items())],
            "routes": len(rows), "dropped_same_side": 0,
            "types": sorted({r["type"] for r in rows if "type" in r})}


def parse_weights(text):
    """'E0=2,-E0=1' -> {'E0': 2.0, '-E0': 1.0}"""
    out = {}
    for item in (text or "").split(","):
        if "=" in item:
            k, v = item.rsplit("=", 1)
            out[k.strip()] = float(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--out", default="routes.rou.xml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--end", type=int, default=3600)
    p.add_argument("--slice", type=int, default=600)
    p.add_argument("--rate-lo", type=float, default=150.0)
    p.add_argument("--rate-hi", type=float, default=500.0)
    p.add_argument("--level-lo", type=float, default=0.4)
    p.add_argument("--level-hi", type=float, default=1.6)
    p.add_argument("--origin-weight", default="", help="vd: E0=2,-E0=1")
    p.add_argument("--sources", nargs="*")
    p.add_argument("--sinks", nargs="*")
    p.add_argument("--allow-uturn", action="store_true", help="cho phep quay dau / thoat cung phia (khong khuyen dung)")
    a = p.parse_args()
    info = make_demand(a.net, a.out, a.seed, a.end, a.slice, (a.rate_lo, a.rate_hi),
                       (a.level_lo, a.level_hi), parse_weights(a.origin_weight), a.sources, a.sinks,
                       no_uturn=not a.allow_uturn)
    print("Canh nguon:", info["sources"])
    print("Canh dich :", info["sinks"])
    print(f"So route: {info['routes']} (bo {info['dropped_same_side']} cap thoat ra cung phia voi luc vao)")
    for b, e, lvl in info["slices"]:
        print(f"  {b:5d}-{e:5d}s : muc luu luong x{lvl}")
    print("Da ghi", a.out)


if __name__ == "__main__":
    main()
