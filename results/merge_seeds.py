"""Gop ket qua nhieu lan train (moi seed 1 thu muc) thanh 1 bang: trung binh +- do lech chuan.
Doc <thu_muc>/best_metrics.csv (cot eval_TB = trung binh cac seed eval cua best.pt; baseline_TB = dieu khien goc).

  python merge_seeds.py run_s*                 # liet ke thu muc hoac mau
  python merge_seeds.py run_s0 run_s1 run_s2 --out merged.csv
Moi lan train duoc coi la 1 mau; std la do lech chuan GIUA CAC LAN TRAIN (ddof=1 neu >= 2 lan)."""
import argparse
import csv
import glob
import os
import statistics as st


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(d):
    p = os.path.join(d, "best_metrics.csv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["nhom"], r["chi_so"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="cac thu muc train (ho tro ky tu *)")
    ap.add_argument("--out", default="merged_results.csv")
    a = ap.parse_args()
    dirs = sorted({d for p in a.dirs for d in (glob.glob(p) or [p]) if os.path.isdir(d)})
    runs = {d: load(d) for d in dirs}
    runs = {d: r for d, r in runs.items() if r}
    if not runs:
        raise SystemExit("khong thay best_metrics.csv trong cac thu muc da cho")
    names = list(runs)
    keys = list(dict.fromkeys(k for r in runs.values() for k in r if k[0] not in ("meta", "tham_so")))

    def ms(v):
        v = [x for x in v if x is not None]
        if not v:
            return "", ""
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    rows = []
    for k in keys:
        ev = [num(runs[d].get(k, {}).get("eval_TB")) for d in names]
        bs = [num(runs[d].get(k, {}).get("baseline_TB")) for d in names]
        m, s = ms(ev)
        bm, _ = ms(bs)
        chg = (m - bm) / bm * 100 if m != "" and bm not in ("", 0) else ""
        rows.append([k[0], k[1]] + ["" if x is None else x for x in ev] + [m, s, bm, chg])
    best_ep = [runs[d].get(("meta", "best_episode"), {}).get("eval_TB", "") for d in names]
    head = ["nhom", "chi_so"] + names + ["TB", "std", "baseline_TB", "so_voi_baseline_%"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerow(["meta", "best_episode"] + best_ep + ["", "", "", ""])
        for r in rows:
            w.writerow([round(x, 4) if isinstance(x, float) else x for x in r])

    def f(x, n=2):
        return "" if x == "" else f"{x:.{n}f}"
    print(f"\nGop {len(names)} lan train: {', '.join(names)}   (best_episode: {', '.join(map(str, best_ep))})")
    print(f"{'chi so':<34}{'TB +- std':>20}{'baseline':>12}{'doi %':>9}")
    for r in rows:
        if r[0] in ("tong", "") or r[0].startswith("den ") or r[0] in ("chung", "ket_xe", "toc_do", "hang_cho", "thong_luong"):
            label = f"{r[0]}/{r[1]}"
        else:
            label = f"{r[0]}/{r[1]}"
        if r[0].startswith("canh") or r[0].startswith("edge"):      # bo qua so xe tung canh trong bang in (van co trong CSV)
            continue
        print(f"{label:<34}{f(r[-4]) + ' +- ' + f(r[-3]):>20}{f(r[-2]):>12}{f(r[-1], 1):>9}")
    print(f"\nDa ghi day du (ca so xe tung canh, tung den): {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
