#!/usr/bin/env python3
"""reproduce.py — one-shot reproduction of the Copenhagen benchmark suite.

Does everything end to end:
  1. builds the C++ extension (src/build.sh) and the standalone quantization
     harnesses (src/*_test.cpp),
  2. installs the comparison libraries (faiss-cpu, hnswlib, h5py) if missing,
  3. downloads the ANN-benchmark datasets used by the suite,
  4. runs the DYNAMICS benchmarks (Copenhagen vs FAISS / HNSW under churn,
     drift, and insert scaling) and the COMPRESSION benchmarks (TurboQuant
     scalar vs block-VQ recall-per-byte — the path meant to replace IVFPQ),
  5. collects every figure and writes a detailed Markdown report under
     benchmarks/results/.

The thesis the report is built to demonstrate:
  • dominate FAISS / HNSW on DYNAMICS (insert / delete / drift), and
  • dominate IVFPQ (and match TurboVec) on COMPRESSION (recall-per-byte).

Usage
  python3 benchmarks/reproduce.py                 # full suite (~20-40 min)
  python3 benchmarks/reproduce.py --quick         # small sizes, fast smoke
  python3 benchmarks/reproduce.py --only dynamics # dynamics | compression | drift
  python3 benchmarks/reproduce.py --skip-deps     # skip pip install + build
  python3 benchmarks/reproduce.py --skip-data     # skip dataset downloads
"""
import argparse
import os
import datetime as _dt
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmarks"
SRC = REPO / "src"
FIGS = REPO / "figures"
RESULTS = BENCH / "results"
LOGS = RESULTS / "logs"
PY = sys.executable


try:
    import turbovec as _tv  # noqa: F401
    _HAS_TURBOVEC = True
except Exception:  # noqa: BLE001
    _HAS_TURBOVEC = False


def sh(cmd, cwd=REPO, timeout=None, env=None):
    """Run a command, capture combined output, return (ok, output, seconds)."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=env)
        return p.returncode == 0, p.stdout, time.time() - t0
    except subprocess.TimeoutExpired as e:
        out = e.output or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return False, out + f"\n[TIMEOUT after {timeout}s]", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return False, f"[ERROR launching {cmd}: {e}]", time.time() - t0


# ───────────────────────────── prepare ──────────────────────────────────────
def ensure_deps():
    need = []
    for mod, pkg in [("faiss", "faiss-cpu"), ("hnswlib", "hnswlib"),
                     ("h5py", "h5py"), ("turbovec", "turbovec")]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print(f"  installing: {need}")
        sh([PY, "-m", "pip", "install", *need], timeout=600)


def build_ext():
    ok, out, secs = sh(["bash", str(SRC / "build.sh")], timeout=600)
    print(f"  build.sh: {'OK' if ok else 'FAILED'} ({secs:.1f}s)")
    if not ok:
        print(out[-1500:])
    # Integrated block-VQ pybind module (the Copenhagen-TQ row in vs_turbovec).
    ok2, out2, _ = sh(["bash", str(SRC / "build_block_vq.sh")], timeout=600)
    print(f"  build_block_vq.sh: {'OK' if ok2 else 'FAILED'}")
    if not ok2:
        print(out2[-1500:])
    return ok


# The standalone C++ quantization harnesses (compression section).
QUANT_TESTS = [
    ("tq_recall_per_byte", "tq_standalone_test.cpp",
     "Scalar TurboQuant recall-per-byte vs exact + rerank@200 (dim sweep 128/768/1536)"),
    ("tq_block_vs_scalar", "tq_block_test.cpp",
     "Block-VQ vs scalar at MATCHED bytes/vector (lever #1)"),
    # (ScaNN anisotropic test lives on the experiments/anisotropic branch — culled from main.)
]


def build_quant_harnesses():
    built = []
    for key, srcfile, _ in QUANT_TESTS:
        out_bin = f"/tmp/{key}"
        ok, log, _ = sh(["c++", "-O3", "-std=c++17", "-march=native",
                         str(SRC / srcfile), "-o", out_bin], timeout=300)
        print(f"  compile {srcfile}: {'OK' if ok else 'FAILED'}")
        if not ok:
            print(log[-800:])
        built.append((key, out_bin, ok))
    return built


# ─────────────────────────── benchmark registry ─────────────────────────────
def dynamics_benches(quick):
    churn_args = (["--n", "8000", "--rounds", "4"] if quick else [])
    # TurboVec joins the churn tables as a +rebuild column (normalizes data).
    tv_arg = ["--with-turbovec"] if _HAS_TURBOVEC else []
    return [
        dict(key="ivf_churn", section="dynamics",
             title="Streaming churn vs FAISS IVF + TurboVec (30% delete/round)",
             cmd=[PY, str(BENCH / "benchmark_ivf_churn.py"), *churn_args, *tv_arg],
             timeout=1800, fig="ivf_churn.png", needs=[]),
        dict(key="hnsw_churn", section="dynamics",
             title="Streaming churn vs HNSW + TurboVec (30% delete/round)",
             cmd=[PY, str(BENCH / "benchmark_hnsw_churn.py"), *churn_args, *tv_arg],
             timeout=1800, fig="hnsw_churn.png", needs=[]),
        dict(key="vs_faiss_gauss", section="dynamics",
             title="Static recall / throughput vs FAISS (synthetic Gaussian)",
             cmd=[PY, str(BENCH / "benchmark_vs_faiss.py"), "gauss"],
             timeout=1200, fig=None, needs=[]),
        dict(key="insert_scaling", section="dynamics",
             title="Insert scaling O(1) vs O(log n) (SIFT-128)",
             cmd=[PY, str(BENCH / "benchmark_insert_scaling.py")],
             timeout=1800, fig=None, needs=["sift"]),
        dict(key="vs_hnsw_sift", section="dynamics",
             title="Static recall / QPS vs HNSW (SIFT-128)",
             cmd=[PY, str(BENCH / "benchmark_vs_hnsw.py"), "sift"],
             timeout=1800, fig=None, needs=["sift"]),
        dict(key="vs_turbovec", section="dynamics",
             title="Copenhagen vs TurboVec — compression + static recall + dynamics "
                   "(normalized SIFT; the FAISS-on-dynamics / TurboVec-on-compression claim)",
             cmd=[PY, str(BENCH / "benchmark_vs_turbovec.py"),
                  "--n", ("8000" if quick else "50000")],
             timeout=1800, fig=None, needs=["sift"]),
    ]


def drift_benches(quick):
    return [
        dict(key="drift", section="drift",
             title="Distribution drift MNIST→Fashion-MNIST (784-d)",
             cmd=[PY, str(BENCH / "benchmark_drift.py")],
             timeout=1800, fig=None, needs=["mnist", "fashion"]),
        dict(key="drift_streaming", section="drift",
             title="Gradual streaming drift (Fashion in batches)",
             cmd=[PY, str(BENCH / "benchmark_drift_streaming.py")],
             timeout=1800, fig=None, needs=["mnist", "fashion"]),
    ]


def have_dataset(name):
    paths = {
        "sift": REPO / "data/sift/sift-128-euclidean.hdf5",
        "mnist": REPO / "data/MNIST/mnist-784-euclidean.hdf5",
        "fashion": REPO / "data/fashion-mnist/fashion-mnist-784-euclidean.hdf5",
    }
    return paths.get(name) and paths[name].exists()


# ─────────────────────────────── report ─────────────────────────────────────
def tail(s, n=120):
    lines = s.rstrip().splitlines()
    return "\n".join(lines[-n:])


def write_report(stamp, results, quant_results, args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rep = RESULTS / f"REPORT_{stamp}.md"
    latest = RESULTS / "REPORT_latest.md"
    L = []
    L.append(f"# Copenhagen benchmark reproduction — {stamp}")
    L.append("")
    L.append(f"- host: `{shutil.which('uname') and subprocess.getoutput('uname -msr')}`")
    L.append(f"- python: `{sys.version.split()[0]}`  mode: "
             f"`{'quick' if args.quick else 'full'}`")
    L.append("")
    L.append("**Thesis:** dominate FAISS/HNSW on *dynamics* and IVFPQ on "
             "*compression* (recall-per-byte). Tables below are raw benchmark "
             "stdout.")
    L.append("")
    # status table
    L.append("## Run status")
    L.append("")
    L.append("| benchmark | section | status | seconds |")
    L.append("|---|---|---|---|")
    for r in results:
        st = "✅ ok" if r["ok"] else ("⏭️ skipped" if r.get("skipped") else "❌ fail")
        L.append(f"| {r['key']} | {r['section']} | {st} | {r['secs']:.0f} |")
    for q in quant_results:
        st = "✅ ok" if q["ok"] else "❌ fail"
        L.append(f"| {q['key']} | compression | {st} | {q['secs']:.0f} |")
    L.append("")
    # sections
    for section in ("dynamics", "drift"):
        sec = [r for r in results if r["section"] == section]
        if not sec:
            continue
        L.append(f"## {section.capitalize()} benchmarks")
        L.append("")
        for r in sec:
            L.append(f"### {r['title']}")
            L.append("")
            if r.get("skipped"):
                L.append(f"_skipped: {r['skipped']}_\n")
                continue
            L.append("```")
            L.append(tail(r["output"]))
            L.append("```")
            if r.get("fig") and (FIGS / r["fig"]).exists():
                L.append("")
                L.append(f"![{r['key']}](../../figures/{r['fig']})")
            L.append("")
    # compression
    L.append("## Compression benchmarks (TurboQuant — IVFPQ replacement)")
    L.append("")
    for q in quant_results:
        L.append(f"### {q['title']}")
        L.append("")
        L.append("```")
        L.append(tail(q["output"], 60))
        L.append("```")
        L.append("")
    rep.write_text("\n".join(L))
    latest.write_text("\n".join(L))
    return rep


# ──────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes, fast smoke")
    ap.add_argument("--only", default="", help="comma list: dynamics,compression,drift")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many benchmarks concurrently (each capped to "
                         "cores/jobs threads). On a 30-vCPU box, --jobs 6 is a good start.")
    args = ap.parse_args()
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    want = lambda s: (not only) or (s in only)  # noqa: E731

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"=== Copenhagen reproduce.py  ({stamp}, {'quick' if args.quick else 'full'}) ===")

    # 1. prepare
    if not args.skip_deps:
        print("[1/4] deps + build")
        ensure_deps()
        build_ext()
    quant_built = build_quant_harnesses() if want("compression") else []

    # 2. data
    needed = set()
    if want("dynamics"):
        needed |= {"sift"}
    if want("drift"):
        needed |= {"mnist", "fashion"}
    if needed and not args.skip_data:
        print(f"[2/4] datasets: {sorted(needed)}")
        sh([PY, str(BENCH / "download_data.py"), *sorted(needed)], timeout=3600)

    # 3. run dynamics + drift
    print("[3/4] benchmarks")
    benches = []
    if want("dynamics"):
        benches += dynamics_benches(args.quick)
    if want("drift"):
        benches += drift_benches(args.quick)
    runnable, results = [], []
    for b in benches:
        missing = [d for d in b["needs"] if not have_dataset(d)]
        if missing:
            print(f"  - {b['key']}: SKIP (missing {missing})")
            results.append({**b, "ok": False, "skipped": f"missing data {missing}",
                            "output": "", "secs": 0.0})
        else:
            runnable.append(b)

    # Per-job thread cap so concurrent jobs don't oversubscribe the cores; each
    # job's FAISS/BLAS/OpenMP still uses `per_job` threads. cores/jobs is the
    # natural split (min 1).
    cores = os.cpu_count() or 1
    per_job = max(1, cores // max(1, args.jobs))

    def run_one(b):
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = str(per_job)
        ok, out, secs = sh(b["cmd"], timeout=b["timeout"], env=env)
        (LOGS / f"{b['key']}.log").write_text(out)
        return {**b, "ok": ok, "output": out, "secs": secs}

    if args.jobs > 1 and len(runnable) > 1:
        print(f"  running {len(runnable)} benchmarks, {args.jobs} concurrent "
              f"× {per_job} threads each ({cores} cores)")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, b): b for b in runnable}
            for f in as_completed(futs):
                r = f.result()
                print(f"  - {r['key']}: {'ok' if r['ok'] else 'FAIL'} ({r['secs']:.0f}s)")
                results.append(r)
    else:
        for b in runnable:
            print(f"  - {b['key']} …", end="", flush=True)
            r = run_one(b)
            print(f" {'ok' if r['ok'] else 'FAIL'} ({r['secs']:.0f}s)")
            results.append(r)

    # 4. compression harnesses
    quant_results = []
    if want("compression"):
        print("[4/4] compression harnesses")
        for (key, out_bin, ok_build), (_, _, title) in zip(quant_built, QUANT_TESTS):
            if not ok_build:
                quant_results.append(dict(key=key, title=title, ok=False,
                                          output="[build failed]", secs=0.0))
                continue
            print(f"  - {key} …", end="", flush=True)
            ok, out, secs = sh([out_bin], timeout=1200)
            (LOGS / f"{key}.log").write_text(out)
            print(f" {'ok' if ok else 'FAIL'} ({secs:.0f}s)")
            quant_results.append(dict(key=key, title=title, ok=ok, output=out, secs=secs))

    rep = write_report(stamp, results, quant_results, args)
    print(f"\n=== report: {rep} ===")
    print(f"=== figures: {FIGS} ===")


if __name__ == "__main__":
    main()
