#!/usr/bin/env python
"""Summarise an nsys report produced by run_profile.sh.

    python analyze.py reports/diffusion_rank0.nsys-rep [--top 15]

Prints, for the capture window:
  * wall time, GPU busy time (union of kernels + memcpy + memset), GPU idle %
  * per NVTX range (main process and DataLoader workers): count, total, mean
  * GPU time attributed to each NVTX range (kernels launched inside it)
  * top kernels, memcpy by direction, the CUDA API calls that block the host
"""
import argparse
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict


def export_sqlite(rep):
    out = os.path.splitext(rep)[0] + ".sqlite"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(rep):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "--output", out, rep], check=True, stdout=subprocess.DEVNULL)
    return out


def table_exists(cur, name):
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (name,)).fetchone() is not None


def union_length(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def fmt_ms(ns):
    return f"{ns / 1e6:10.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    db = sqlite3.connect(export_sqlite(args.report))
    cur = db.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))

    # ---------------- GPU activity ----------------
    gpu = []  # (start, end, kind, name)
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for s, e, nid in cur.execute(
                "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            gpu.append((s, e, "kernel", strings.get(nid, str(nid))))
    copy_kind = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for s, e, k, b in cur.execute(
                "SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            gpu.append((s, e, "memcpy", (copy_kind.get(k, str(k)), b)))
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for s, e in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            gpu.append((s, e, "memset", None))
    if not gpu:
        print("no GPU activity in report")
        return

    # Capture window = first to last traced event of any kind.
    t0 = min(g[0] for g in gpu)
    t1 = max(g[1] for g in gpu)
    if table_exists(cur, "NVTX_EVENTS"):
        r = cur.execute("SELECT min(start), max(end) FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchone()
        if r[0] is not None:
            t0, t1 = min(t0, r[0]), max(t1, r[1])
    wall = t1 - t0
    busy = union_length([(s, e) for s, e, *_ in gpu])
    kern_busy = union_length([(s, e) for s, e, k, _ in gpu if k == "kernel"])
    print(f"== {args.report}")
    print(f"capture wall time      {wall / 1e9:8.2f} s")
    print(f"GPU busy (any)         {busy / 1e9:8.2f} s  ({100 * busy / wall:5.1f} %)")
    print(f"GPU busy (kernels)     {kern_busy / 1e9:8.2f} s  ({100 * kern_busy / wall:5.1f} %)")
    print(f"GPU idle               {(wall - busy) / 1e9:8.2f} s  ({100 * (wall - busy) / wall:5.1f} %)")

    # ---------------- NVTX ranges ----------------
    if table_exists(cur, "NVTX_EVENTS"):
        tid_pid = {}
        rows = cur.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND eventType IN (59, 60)").fetchall()
        stats = defaultdict(lambda: [0, 0, None, 0])  # count, total, min, max
        main_pid = None
        # The process that launched kernels is the "main" one; others are workers.
        if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
            r = cur.execute("SELECT globalTid, count(*) c FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                            "GROUP BY globalTid ORDER BY c DESC LIMIT 1").fetchone()
            if r:
                main_pid = r[0] >> 24
        for s, e, text, tid, gtid in rows:
            name = text if text is not None else strings.get(tid, "?")
            if name.startswith("NCCL") or name.startswith("nccl"):
                continue
            where = "main" if (gtid >> 24) == main_pid else "worker"
            st = stats[(where, name)]
            d = e - s
            st[0] += 1
            st[1] += d
            st[2] = d if st[2] is None else min(st[2], d)
            st[3] = max(st[3], d)
        print("\nNVTX ranges (CPU wall time; worker = DataLoader processes, summed over workers)")
        print(f"{'where':6} {'range':28} {'count':>6} {'total ms':>10} {'mean ms':>10} "
              f"{'min ms':>10} {'max ms':>10} {'% wall':>7}")
        for (where, name), (n, tot, mn, mx) in sorted(stats.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
            print(f"{where:6} {name[:28]:28} {n:6d} {fmt_ms(tot)} {fmt_ms(tot / n)} "
                  f"{fmt_ms(mn)} {fmt_ms(mx)} {100 * tot / wall:7.1f}")

        # GPU time of kernels whose launch (runtime API call) falls inside each
        # NVTX range on the same thread — innermost range wins.
        if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME") and table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
            launches = cur.execute(
                "SELECT r.start, r.globalTid, k.end - k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId").fetchall()
            ranges_by_tid = defaultdict(list)
            for s, e, text, tid, gtid in rows:
                name = text if text is not None else strings.get(tid, "?")
                ranges_by_tid[gtid].append((s, e, name))
            gpu_by_range = defaultdict(int)
            for ls, gtid, kd in launches:
                best = None
                for s, e, name in ranges_by_tid.get(gtid, ()):
                    if s <= ls <= e and (best is None or (e - s) < (best[1] - best[0])):
                        best = (s, e, name)
                gpu_by_range[best[2] if best else "(no range)"] += kd
            print("\nGPU kernel time by innermost NVTX range at launch")
            tot = sum(gpu_by_range.values())
            for name, t in sorted(gpu_by_range.items(), key=lambda kv: -kv[1])[:args.top]:
                print(f"  {name[:40]:40} {fmt_ms(t)} ms  {100 * t / tot:5.1f} %")

    # ---------------- kernels ----------------
    kern = defaultdict(lambda: [0, 0])
    for s, e, k, name in gpu:
        if k == "kernel":
            kern[name][0] += 1
            kern[name][1] += e - s
    tot_k = sum(v[1] for v in kern.values())
    print(f"\nTop kernels ({len(kern)} distinct, {sum(v[0] for v in kern.values())} launches, "
          f"total {tot_k / 1e6:.1f} ms)")
    for name, (n, t) in sorted(kern.items(), key=lambda kv: -kv[1][1])[:args.top]:
        print(f"  {100 * t / tot_k:5.1f} %  {fmt_ms(t)} ms  n={n:6d}  avg {t / n / 1e3:8.1f} us  {name[:90]}")
    small = sum(1 for _, _, k, _ in gpu if k == "kernel")
    short = sum(1 for s, e, k, _ in gpu if k == "kernel" and e - s < 10_000)
    print(f"  kernels shorter than 10 us: {short} / {small} ({100 * short / max(small, 1):.0f} %)")

    # ---------------- memcpy ----------------
    mc = defaultdict(lambda: [0, 0, 0])
    for s, e, k, info in gpu:
        if k == "memcpy":
            d, b = info
            mc[d][0] += 1
            mc[d][1] += e - s
            mc[d][2] += b
    if mc:
        print("\nMemcpy")
        for d, (n, t, b) in mc.items():
            print(f"  {d:5} n={n:7d}  {fmt_ms(t)} ms  {b / 1e9:8.2f} GB  "
                  f"{(b / 1e9) / max(t / 1e9, 1e-9):6.1f} GB/s")

    # ---------------- CUDA API (host side) ----------------
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        api = defaultdict(lambda: [0, 0])
        for nid, s, e in cur.execute("SELECT nameId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            name = strings.get(nid, str(nid))
            api[name][0] += 1
            api[name][1] += e - s
        print("\nCUDA API host time (top)")
        for name, (n, t) in sorted(api.items(), key=lambda kv: -kv[1][1])[:args.top]:
            print(f"  {fmt_ms(t)} ms  n={n:7d}  {name}")

    # ---------------- OS runtime ----------------
    if table_exists(cur, "OSRT_API"):
        osrt = defaultdict(lambda: [0, 0])
        for nid, s, e in cur.execute("SELECT nameId, start, end FROM OSRT_API"):
            name = strings.get(nid, str(nid))
            osrt[name][0] += 1
            osrt[name][1] += e - s
        print("\nOS runtime (summed over threads; blocking waits included)")
        for name, (n, t) in sorted(osrt.items(), key=lambda kv: -kv[1][1])[:10]:
            print(f"  {fmt_ms(t)} ms  n={n:7d}  {name}")


if __name__ == "__main__":
    sys.exit(main())
