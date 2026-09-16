#!/usr/bin/env python3
"""One-line-per-job progress table for a running 2Q sweep.

    python3 check_2q.py                 # all logs/*.out in this directory
    python3 check_2q.py --pattern 'logs/job_s101_*.out'
    python3 check_2q.py --watch 300     # reprint every 5 minutes

Reads the streamed job logs (sweep_2q.sub sets stream_output = True), so it works while the
jobs are still running. For each job it reports how far POUNDERS has got, how much of the
shot budget has been revealed, the per-iteration cost broken into its three logged pieces,
and a projected finish -- which is what decides whether the run fits inside 72 hours.

The columns that matter on a first check:
  iter      should advance ~1 per (jac + trsp + comb) seconds
  shots     must reach 100% by roughly iteration 150 (that is what --schedule-n-min buys)
  proj_h    projected total hours at the current rate; anything near 72 needs attention
"""
import argparse
import glob
import os
import re
import time

RE_ITER = re.compile(r"\[POUDERS\] nf: (\d+), delta: ([\d.eE+-]+), f\(x\): ([\d.eE+-]+), "
                     r"ng: ([\d.eE+-]+)")
RE_SHOTS = re.compile(r"\[SHOTS\] \S+ iter (\d+): (\d+)/(\d+) cumulative revealed shots")
RE_JAC = re.compile(r"Residual/Jacobian evaluation took ([\d.]+) seconds")
RE_TRSP = re.compile(r"PyROL trust-region solve took ([\d.]+) seconds")
RE_COMB = re.compile(r"Model combine took ([\d.]+) seconds")
RE_BASE = re.compile(r"baseline (\d+) sh/circ x (\d+) = ([\d,]+)")
RE_OVER = re.compile(r"config overrides: (\{.*\})")
RE_SHA = re.compile(r"config present: (\S+)\s+sha256=(\S+)")
RE_DONE = re.compile(r"\[run_one\] seed \d+ done in (\d+)s")
# job.sh's first line: "=== 2026-09-07T15:27:01Z host=... seed=101 tag=..."
RE_START = re.compile(r"^=== (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z host=")
RE_ERR = re.compile(r"(Traceback|MemoryError|ModuleNotFoundError|Killed|raise )")


def tail_mean(values, n=10):
    v = values[-n:]
    return sum(v) / len(v) if v else float("nan")


def scan(path):
    out = {"job": os.path.basename(path).replace("job_", "").replace(".out", "")}
    jac, trsp, comb, iters, shots = [], [], [], [], None
    err, done, sha, nfmax = "", None, "", None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_ITER.search(line)
            if m:
                iters.append((int(m.group(1)), float(m.group(2)), float(m.group(4))))
                continue
            m = RE_SHOTS.search(line)
            if m:
                shots = (int(m.group(2)), int(m.group(3)))
                continue
            for rx, acc in ((RE_JAC, jac), (RE_TRSP, trsp), (RE_COMB, comb)):
                m = rx.search(line)
                if m:
                    acc.append(float(m.group(1)))
                    break
            m = RE_OVER.search(line)
            if m and "nfmax" in m.group(1):
                nm = re.search(r"'nfmax': (\d+)", m.group(1))
                nfmax = int(nm.group(1)) if nm else None
            m = RE_SHA.search(line)
            if m and "2q" in m.group(1):
                sha = m.group(2)
            m = RE_DONE.search(line)
            if m:
                done = int(m.group(1))
            m = RE_START.match(line)
            if m:
                out["start_epoch"] = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) \
                    - time.timezone
            if RE_ERR.search(line):
                err = line.strip()[:40]
    out["iter"] = iters[-1][0] if iters else 0
    out["delta"] = iters[-1][1] if iters else float("nan")
    out["ng"] = iters[-1][2] if iters else float("nan")
    out["jac"] = tail_mean(jac)
    out["trsp"] = tail_mean(trsp)
    out["comb"] = tail_mean(comb)
    out["shots_pct"] = 100.0 * shots[0] / shots[1] if shots and shots[1] else float("nan")
    out["nfmax"] = nfmax
    out["sha"] = sha[:8]
    out["done_s"] = done
    out["err"] = err
    # Wall clock from job.sh's own header stamp to the last write, so it INCLUDES the
    # allocator time the log never prints -- that unlogged gap was 73% of the held job.
    # (st_ctime is not a creation time on Linux, hence the header stamp.)
    mtime = os.path.getmtime(path)
    age_h = (time.time() - mtime) / 3600.0
    started_h = float("nan")
    if out.get("start_epoch"):
        started_h = (mtime - out["start_epoch"]) / 3600.0
    out["elapsed_h"] = started_h
    per_iter = (started_h * 3600.0 / out["iter"]) if out["iter"] else float("nan")
    out["s_per_it"] = per_iter
    out["proj_h"] = per_iter * nfmax / 3600.0 if (nfmax and out["iter"]) else float("nan")
    out["idle_m"] = age_h * 60.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="logs/*.out")
    ap.add_argument("--watch", type=int, default=0, help="seconds between reprints")
    a = ap.parse_args()
    while True:
        rows = [scan(p) for p in sorted(glob.glob(a.pattern))]
        if not rows:
            print(f"no logs matching {a.pattern}")
            return
        print(f"\n{time.strftime('%F %T')}   {len(rows)} jobs")
        print(f"{'job':<22}{'iter':>6}{'shots%':>8}{'jac':>7}{'trsp':>7}{'comb':>7}"
              f"{'s/it':>8}{'elap_h':>8}{'proj_h':>8}{'idle_m':>8}  {'cfg':<9}note")
        for r in sorted(rows, key=lambda x: x["job"]):
            note = r["err"] or ("DONE " + str(r["done_s"]) + "s" if r["done_s"] else "")
            print(f"{r['job']:<22}{r['iter']:>6}{r['shots_pct']:>8.0f}{r['jac']:>7.0f}"
                  f"{r['trsp']:>7.0f}{r['comb']:>7.0f}{r['s_per_it']:>8.0f}"
                  f"{r['elapsed_h']:>8.1f}{r['proj_h']:>8.1f}{r['idle_m']:>8.0f}  "
                  f"{r['sha']:<9}{note}")
        stuck = [r["job"] for r in rows if r["idle_m"] > 30 and not r["done_s"]]
        if stuck:
            print(f"!! no output for >30 min: {', '.join(stuck[:6])}")
        slow = [r["job"] for r in rows if r["proj_h"] > 60]
        if slow:
            print(f"!! projected past 60 h: {', '.join(slow[:6])}")
        if not a.watch:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
