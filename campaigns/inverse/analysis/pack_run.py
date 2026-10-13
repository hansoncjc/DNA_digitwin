"""
Pack an inverse.py run for download: a run-level summary plus the files the
local analysis needs. Standard library only (no torch, HOOMD or OVITO), so
it runs on a login node of the cluster; out_root may belong to a finished
run or to one that is still running. Nothing in out_root is changed.

    python campaigns/inverse/analysis/pack_run.py OUT_ROOT --top 3 --name r6.6
    python campaigns/inverse/analysis/pack_run.py OUT_ROOT --summary-only   # no tar

Writes, under --dest (default: the parent of out_root):

    <name>_pack_summary/   summary.json, summary.txt, evals.csv, batches.csv,
                           gp_lengthscale.csv, manifest.txt
    <name>_pack.tar.gz     <name>/pack_summary/  (the folder above)
                           <name>/Optimization_Results/ with
        run level   bo_trajectory.csv, loss_components.txt, inverse_config.json,
                    inverse_summary.{json,txt}, gp_log/
        every eval  <ds>/S(q)_data/*.npy, <ds>/shift_rmse_diagnostics.json,
                    <ds>/sim_params_*.csv, <ds>/ flag files (DONE, FAILED,
                    METRIC_FAILED, ABANDONED), _jobs/**/summary.json,
                    _jobs/**/launcher.log, _jobs/**/*.out (timing sources)
        top evals   <ds>/DNA_assembly_*.gsd, <ds>/potential_energy.csv,
                    <ds>/*.png, <ds>/S(q)_data/*.png

Summary contents: the best and top-k evaluations (parameters with m, loss,
M4, spacing term, q1 ratio); convergence per batch and cumulative; the first
batch (initial point + Sobol points) against the rest; timing of every
evaluation; dispersed and failed counts; GP lengthscales per batch.

Timing of an evaluation (one dataset; the last Slurm attempt when retried):

    submit    sacct Submit, else the mtime of _jobs/jobs/<job>.sbatch
    start/end sacct Start/End, else the "Start:"/"End:" lines of the job log
    queue_s   start - submit (summed over attempts)
    worker_s  run_time_seconds of the DONE / METRIC_FAILED flag (worker.py:
              initialization, production, S(q), loss)
    sim_s     mtime(DNA_assembly_*.gsd) - (mtime(flag) - worker_s):
              initialization and production
    sq_s      mtime(S(q)_data/average_structure_factor.npy) - mtime(gsd)
    total_s   end of the last attempt - submit of the first

File mtimes are the cluster's file system clock, sacct and job-log times the
Slurm controller's and the node's; they agree to about a second.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tarfile
import time
from datetime import datetime
from pathlib import Path

FLAGS = ("DONE", "METRIC_FAILED", "FAILED", "ABANDONED")
RUN_FILES = ("bo_trajectory.csv", "loss_components.txt", "inverse_config.json",
             "inverse_summary.json", "inverse_summary.txt")
# Columns of a FAILED block's rows (bo._TRAJECTORY_PARAM_COLUMNS): those
# blocks carry no header row of their own.
FAILED_ROW_COLUMNS = ("iteration", "dataset_id", "loss", "k", "A", "K_s", "a_m",
                      "delta", "density", "n", "m", "r0", "U0", "m_rel")
PARAMS = ("U0", "r0", "n", "m", "m_rel")
SACCT_FIELDS = ("JobIDRaw", "JobName", "Submit", "Start", "End", "Elapsed",
                "State", "NodeList", "Partition")
LOG_DATE_FORMATS = ("%a %b %d %H:%M:%S %Y", "%a %b %d %I:%M:%S %p %Y",
                    "%a %d %b %Y %H:%M:%S", "%a %d %b %Y %I:%M:%S %p")


# ------------------------------------------------------------------ helpers


def _float(value):
    """float(value), or None for blanks, None and non-numbers."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _iso(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def _stats(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    p90 = vals[min(len(vals) - 1, int(math.ceil(0.9 * len(vals))) - 1)]
    return {"n": len(vals), "min": vals[0], "median": statistics.median(vals),
            "p90": p90, "max": vals[-1]}


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def _spearman(x, y):
    return _pearson(_ranks(x), _ranks(y)) if len(x) >= 3 else None


# ------------------------------------------------------------------ run files


def read_trajectory(path):
    """
    ``{eval_id: block}`` from bo_trajectory.csv. A block has ``status``
    (``ok``, ``FAILED``, ``METRIC_FAILED``, ``ABANDONED``), ``total_loss``
    (ok only), ``reason`` and ``rows`` (``{column: value}`` per dataset).
    """
    blocks = {}
    if not Path(path).is_file():
        return blocks
    header, current = None, None
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                header, current = None, None
                continue
            if row[0].startswith("# Iteration"):
                try:
                    eval_id = int(row[0].split()[-1])
                except ValueError:
                    current = None
                    continue
                if len(row) >= 3 and row[1] == "total_loss":
                    blocks[eval_id] = {"status": "ok", "total_loss": _float(row[2]),
                                       "reason": "", "rows": []}
                else:
                    blocks[eval_id] = {"status": row[2] if len(row) > 2 else "FAILED",
                                       "total_loss": None,
                                       "reason": row[3] if len(row) > 3 else "",
                                       "rows": []}
                    header = None
                current = eval_id
                continue
            if row[0] == "iteration" and len(row) > 1 and row[1] == "dataset_id":
                header = row
                continue
            if current is None or row[0] != str(current):
                continue
            cols = header if header is not None else FAILED_ROW_COLUMNS
            if len(row) < len(cols):
                continue                      # row cut by a kill
            blocks[current]["rows"].append(dict(zip(cols, row)))
    return blocks


def read_gp_log(gp_log_dir):
    """Rows of gp_log.jsonl; a cut last line (no newline, not JSON) is skipped."""
    path = Path(gp_log_dir) / "gp_log.jsonl"
    rows = []
    if not path.is_file():
        return rows
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1:
                continue
            raise
    return rows


def eval_dirs(out_root):
    """``{eval_id: Path}`` of every eval_XXX folder."""
    out = {}
    for p in Path(out_root).glob("eval_*"):
        m = re.fullmatch(r"eval_(\d+)", p.name)
        if m and p.is_dir():
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def dataset_dirs(eval_dir):
    return sorted(p for p in eval_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))


def read_sim_params(ds_dir):
    for path in sorted(ds_dir.glob("sim_params_*.csv")):
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            return rows[0]
    return {}


def flag_status(ds_dir):
    for name in FLAGS:
        if (ds_dir / name).exists():
            return name
    if (ds_dir / "RUNNING").exists():
        return "RUNNING"
    return None


def read_flag(ds_dir, name):
    """``(run_time_seconds, host, text)`` of a flag file (JSON or key=value text)."""
    path = ds_dir / name
    if not path.is_file():
        return None, None, None
    text = path.read_text(errors="replace")
    data = None
    try:
        data = json.loads(text)
    except ValueError:
        pass
    if isinstance(data, dict):
        return _float(data.get("run_time_seconds")), data.get("host"), data.get("reason")
    rt = re.search(r"run_time_seconds=([\d.eE+-]+)", text)
    host = re.search(r"host=(\S+)", text)
    return (_float(rt.group(1)) if rt else None, host.group(1) if host else None,
            text.strip().splitlines()[-1] if text.strip() else None)


# ------------------------------------------------------------------ timing


def _parse_log_date(text):
    tokens = [t for t in text.split() if not re.fullmatch(r"[A-Z]{3,4}", t)]
    cleaned = " ".join(tokens)
    for fmt in LOG_DATE_FORMATS:
        try:
            return time.mktime(datetime.strptime(cleaned, fmt).timetuple())
        except ValueError:
            continue
    return None


def _parse_sacct_time(text):
    if not text or text in ("Unknown", "None"):
        return None
    try:
        return time.mktime(datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").timetuple())
    except ValueError:
        return None


def parse_job_log(path):
    """Start, end, node, GPU name and HOOMD average TPS from a worker job log."""
    out = {"start": None, "end": None, "node": None, "gpu": None, "tps": None}
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return out
    tps = []
    for i, line in enumerate(lines):
        if line.startswith("Start:"):
            out["start"] = _parse_log_date(line.split(":", 1)[1])
        elif line.startswith("End:"):
            out["end"] = _parse_log_date(line.split(":", 1)[1])
        elif line.startswith("Node:"):
            out["node"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("name, memory.total") and i + 1 < len(lines):
            out["gpu"] = lines[i + 1].split(",")[0].strip()
        m = re.search(r"Average TPS:\s*([\d.eE+-]+)", line)
        if m:
            tps.append(float(m.group(1)))
    out["tps"] = tps[-1] if tps else None      # the last hoomd.run is production
    return out


def job_attempts(eval_dir):
    """One dict per Slurm job of the evaluation, first attempt first."""
    jobs_root = eval_dir / "_jobs"
    attempts = []
    summaries = sorted(jobs_root.glob("**/summary.json"),
                       key=lambda p: (len(p.relative_to(jobs_root).parts), str(p)))
    for summary_path in summaries:
        data = _read_json(summary_path) or {}
        for job in data.get("jobs", []):
            name = job.get("name", "")
            run_dir = summary_path.parent
            attempts.append({
                "name": name,
                "ds_id": job.get("ds_id"),
                "slurm_id": job.get("slurm_id"),
                "status": job.get("status"),
                "host": job.get("host"),
                "run_time_seconds": _float(job.get("run_time_seconds")),
                "log": run_dir / "jobs" / f"{name}.out",
                "sbatch": run_dir / "jobs" / f"{name}.sbatch",
            })
    if not attempts and jobs_root.is_dir():
        # summary.json is written when the launcher returns; a job still
        # running or a master killed mid-batch leaves only the job files.
        for sbatch in sorted(jobs_root.glob("**/jobs/*.sbatch")):
            attempts.append({"name": sbatch.stem, "ds_id": None, "slurm_id": None,
                             "status": None, "host": None, "run_time_seconds": None,
                             "log": sbatch.with_suffix(".out"), "sbatch": sbatch})
    return attempts


def sacct_times(job_ids, use_sacct=True):
    """``{job_id: {Submit, Start, End, ...}}`` from one sacct call per 200 ids."""
    out = {}
    ids = sorted({str(j) for j in job_ids if j})
    if not use_sacct or not ids:
        return out, "not used"
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        try:
            text = subprocess.check_output(
                ["sacct", "-j", ",".join(chunk), "-X", "-n", "-P",
                 "-o", ",".join(SACCT_FIELDS)],
                text=True, timeout=120, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return out, f"sacct unavailable ({type(exc).__name__})"
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) == len(SACCT_FIELDS):
                rec = dict(zip(SACCT_FIELDS, parts))
                out[rec["JobIDRaw"]] = rec
    return out, "sacct"


def eval_timing(eval_dir, ds_dir, attempts, sacct):
    """Timing of one evaluation (see the module docstring)."""
    t = {"n_attempts": len(attempts), "submit": None, "start": None, "end": None,
         "queue_s": None, "worker_s": None, "sim_s": None, "sq_s": None,
         "total_s": None, "slurm_elapsed": None, "host": None, "gpu": None,
         "partition": None, "tps": None, "timing_source": None}
    if not attempts:
        return t
    queue_total, sources = 0.0, set()
    first_submit, last = None, None
    for att in attempts:
        rec = sacct.get(str(att["slurm_id"])) if att["slurm_id"] else None
        log = parse_job_log(att["log"])
        if rec:
            submit = _parse_sacct_time(rec["Submit"])
            start = _parse_sacct_time(rec["Start"])
            end = _parse_sacct_time(rec["End"])
            sources.add("sacct")
        else:
            submit, start, end = _mtime(att["sbatch"]), log["start"], log["end"]
            sources.add("files")
        if start is not None and submit is not None:
            queue_total += max(0.0, start - submit)
        if first_submit is None:
            first_submit = submit
        last = {"att": att, "rec": rec, "log": log, "start": start, "end": end}
    att, rec, log = last["att"], last["rec"], last["log"]
    t.update({
        "submit": first_submit, "start": last["start"], "end": last["end"],
        "queue_s": queue_total if last["start"] is not None else None,
        "total_s": (last["end"] - first_submit
                    if last["end"] is not None and first_submit is not None else None),
        "slurm_elapsed": rec["Elapsed"] if rec else None,
        "host": att["host"] or log["node"] or (rec["NodeList"] if rec else None),
        "gpu": log["gpu"],
        "partition": rec["Partition"] if rec else None,
        "tps": log["tps"],
        "timing_source": "+".join(sorted(sources)),
    })
    status = flag_status(ds_dir)
    if status in ("DONE", "METRIC_FAILED", "FAILED"):
        worker_s, _, _ = read_flag(ds_dir, status)
        flag_time = _mtime(ds_dir / status)
        t["worker_s"] = worker_s if worker_s is not None else att["run_time_seconds"]
        gsds = sorted(ds_dir.glob("DNA_assembly_*.gsd"))
        gsd_time = _mtime(gsds[-1]) if gsds else None
        sq_time = _mtime(ds_dir / "S(q)_data" / "average_structure_factor.npy")
        if t["worker_s"] is not None and flag_time is not None and gsd_time is not None:
            t["sim_s"] = gsd_time - (flag_time - t["worker_s"])
        if gsd_time is not None and sq_time is not None:
            t["sq_s"] = sq_time - gsd_time
    return t


# ------------------------------------------------------------------ summary


def _init_matches(params, search):
    for name, spec in search.items():
        init = spec.get("init")
        value = _float(params.get(name))
        if init is None or value is None or abs(value - float(init)) > 1e-6 * max(1.0, abs(init)):
            return False
    return True


def collect(out_root, top_k, use_sacct=True):
    """Every number of the summary, as plain Python objects."""
    out_root = Path(out_root)
    snapshot = _read_json(out_root / "inverse_config.json") or {}
    locked = snapshot.get("locked", {})
    search = locked.get("search", {})
    blocks = read_trajectory(out_root / "bo_trajectory.csv")
    gp_dir = out_root / (locked.get("gp_log_dir") or "gp_log")
    if not gp_dir.is_absolute() and not gp_dir.is_dir():
        gp_dir = out_root / "gp_log"
    gp_rows = read_gp_log(gp_dir)
    folders = eval_dirs(out_root)

    batch_of = {}
    for row in gp_rows:
        if row.get("stage") == "acquisition" and row.get("eval_ids"):
            for eid in row["eval_ids"]:
                batch_of[int(eid)] = int(row["batch"])

    attempts_by_eval = {eid: job_attempts(d) for eid, d in folders.items()}
    sacct, sacct_source = sacct_times(
        [a["slurm_id"] for atts in attempts_by_eval.values() for a in atts], use_sacct)

    evals = []
    for eid in sorted(set(blocks) | set(folders)):
        block = blocks.get(eid)
        eval_dir = folders.get(eid)
        ds = dataset_dirs(eval_dir)[0] if eval_dir is not None and dataset_dirs(eval_dir) else None
        row = block["rows"][0] if block and block["rows"] else {}
        sim_params = read_sim_params(ds) if ds is not None else {}
        source = row if block and block["status"] == "ok" else sim_params
        diag = _read_json(ds / "shift_rmse_diagnostics.json") if ds is not None else None
        diag = diag or {}
        status = block["status"] if block else "no_block"
        reason = block["reason"] if block else ""
        if ds is not None and status in ("FAILED", "METRIC_FAILED") and reason in ("", status):
            reason = read_flag(ds, status)[2] or reason
        batch = batch_of.get(eid, 0)
        rec = {
            "eval_id": eid,
            "status": status,
            "batch": batch,
            "origin": ("gp" if batch > 0
                       else "init" if _init_matches(source, search) else "sobol"),
            **{p: _float(source.get(p)) for p in PARAMS},
            "loss": block["total_loss"] if block and status == "ok" else None,
            "rmse": _float(row.get("rmse")),
            "shift": _float(row.get("shift")),
            "q1_tgt": _float(diag.get("q1_tgt")),
            "q1_sim": _float(diag.get("q1_sim")),
            "q1_ratio": _float(diag.get("q1_ratio")),
            "aligned": diag.get("aligned"),
            "dispersed_sim": diag.get("dispersed_sim"),
            "anchor_scale": _float(diag.get("anchor_scale")),
            "rmax_over_r0": _float(row.get("rmax_over_r0")),
            "n_pairs_below_rmin": _float(row.get("n_pairs_below_rmin")),
            "flag": flag_status(ds) if ds is not None else None,
            "reason": reason,
            "dataset_dir": str(ds.relative_to(out_root)) if ds is not None else None,
        }
        timing = (eval_timing(eval_dir, ds, attempts_by_eval.get(eid, []), sacct)
                  if ds is not None else {})
        rec.update({k: timing.get(k) for k in (
            "n_attempts", "queue_s", "worker_s", "sim_s", "sq_s", "total_s",
            "slurm_elapsed", "host", "gpu", "partition", "tps", "timing_source")})
        rec["submit"] = _iso(timing.get("submit"))
        rec["start"] = _iso(timing.get("start"))
        rec["end"] = _iso(timing.get("end"))
        rec["_submit_epoch"] = timing.get("submit")
        rec["_end_epoch"] = timing.get("end")
        evals.append(rec)

    best_so_far = None
    for rec in evals:
        if rec["loss"] is not None:
            best_so_far = rec["loss"] if best_so_far is None else min(best_so_far, rec["loss"])
        rec["cum_best"] = best_so_far

    ok = [r for r in evals if r["status"] == "ok" and r["loss"] is not None]
    ranked = sorted(ok, key=lambda r: (r["loss"], r["eval_id"]))
    for i, rec in enumerate(ranked):
        rec["rank"] = i + 1
    top = ranked[:top_k]
    for rec in top:
        gsds = sorted((out_root / rec["dataset_dir"]).glob("DNA_assembly_*.gsd"))
        rec["gsd"] = gsds[-1].name if gsds else None
        rec["gsd_md5"] = _md5(gsds[-1]) if gsds else None

    batches = []
    cum = None
    for b in sorted({r["batch"] for r in evals}):
        members = [r for r in evals if r["batch"] == b]
        losses = [r["loss"] for r in members if r["loss"] is not None]
        best = min((r for r in members if r["loss"] is not None),
                   key=lambda r: r["loss"], default=None)
        if losses:
            cum = min(losses) if cum is None else min(cum, min(losses))
        submits = [r["_submit_epoch"] for r in members if r["_submit_epoch"] is not None]
        ends = [r["_end_epoch"] for r in members if r["_end_epoch"] is not None]
        slowest = max((r for r in members if r["total_s"] is not None),
                      key=lambda r: r["total_s"], default=None)
        batches.append({
            "batch": b,
            "eval_ids": [r["eval_id"] for r in members],
            "n_ok": len(losses),
            "n_failed": sum(r["status"] in ("FAILED", "METRIC_FAILED") for r in members),
            "best_loss": best["loss"] if best else None,
            "best_eval": best["eval_id"] if best else None,
            "median_loss": statistics.median(losses) if losses else None,
            "cum_best": cum,
            "wall_s": (max(ends) - min(submits)) if submits and ends else None,
            "slowest_eval": slowest["eval_id"] if slowest else None,
        })

    first = [r for r in evals if r["batch"] == 0]
    first_ok = [r for r in first if r["loss"] is not None]
    first_best = min(first_ok, key=lambda r: r["loss"], default=None)
    later_ok = [r for r in ok if r["batch"] > 0]
    first_batch = {
        "evals": [{k: r.get(k) for k in ("eval_id", "origin", *PARAMS, "loss", "rank",
                                         "dispersed_sim", "status")} for r in first],
        "best_eval": first_best["eval_id"] if first_best else None,
        "best_loss": first_best["loss"] if first_best else None,
        "n_later_better_than_first_best": (sum(r["loss"] < first_best["loss"] for r in later_ok)
                                           if first_best else None),
        "n_later": len(later_ok),
        "top_k_from_first_batch": [r["eval_id"] for r in top if r["batch"] == 0],
    }

    timed = [r for r in evals if r["total_s"] is not None]
    timing = {"source": sacct_source}
    for key in ("total_s", "queue_s", "worker_s", "sim_s", "sq_s"):
        timing[key] = _stats(r[key] for r in timed)
    sim_pairs = [(r["rmax_over_r0"] ** 3, r["sim_s"]) for r in evals
                 if r["sim_s"] is not None and r["rmax_over_r0"] is not None]
    timing["sim_s_vs_rmax_over_r0_cubed"] = {
        "n": len(sim_pairs),
        "pearson": _pearson(*zip(*sim_pairs)) if len(sim_pairs) >= 3 else None,
        "spearman": _spearman(*zip(*sim_pairs)) if len(sim_pairs) >= 3 else None,
    }
    tps_pairs = [(r["rmax_over_r0"] ** 3, r["tps"]) for r in evals
                 if r["tps"] is not None and r["rmax_over_r0"] is not None]
    timing["tps_vs_rmax_over_r0_cubed_spearman"] = (
        _spearman(*zip(*tps_pairs)) if len(tps_pairs) >= 3 else None)
    timing["by_gpu"] = {}
    for gpu in sorted({r["gpu"] for r in timed if r["gpu"]}):
        timing["by_gpu"][gpu] = _stats(r["sim_s"] for r in timed if r["gpu"] == gpu)

    names = []
    lengthscales = []
    for row in gp_rows:
        names = row.get("param_names") or names
        ls = row.get("lengthscale") or []
        entry = {"batch": row.get("batch"), "stage": row.get("stage"),
                 "attempt": row.get("attempt"), "n_train": row.get("n_train"),
                 "outputscale": row.get("outputscale"), "noise": row.get("noise"),
                 "best_observed_loss": row.get("best_observed_loss")}
        for name, value in zip(row.get("param_names") or [], ls):
            entry[f"ls_{name}"] = value
            bounds = (search.get(name) or {}).get("bounds")
            entry[f"ls_{name}_phys"] = (value * (bounds[1] - bounds[0])
                                        if bounds and value is not None else None)
        lengthscales.append(entry)

    counts = {
        "eval_folders": len(folders),
        "blocks": len(blocks),
        "ok": len(ok),
        "FAILED": sum(r["status"] == "FAILED" for r in evals),
        "METRIC_FAILED": sum(r["status"] == "METRIC_FAILED" for r in evals),
        "ABANDONED": sum(r["status"] == "ABANDONED" for r in evals),
        "no_block": sum(r["status"] == "no_block" for r in evals),
        "dispersed_sim_ok": sum(bool(r["dispersed_sim"]) for r in ok),
        "not_aligned_ok": sum(r["aligned"] is False for r in ok),
        "retried": sum((r["n_attempts"] or 0) > 1 for r in evals),
    }
    starts = snapshot.get("starts", [])
    return {
        "written": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "description": snapshot.get("description"),
        "target": snapshot.get("target"),
        "search": search,
        "max_evals": starts[-1].get("max_evals") if starts else None,
        "batch_size": locked.get("batch_size"),
        "commits": sorted({s.get("commit") for s in starts if s.get("commit")}),
        "starts": [{k: s.get(k) for k in ("time", "commit", "dirty", "host", "slurm_job_id")}
                   for s in starts],
        "complete": (len(ok) >= starts[-1]["max_evals"]) if starts and starts[-1].get("max_evals") else None,
        "counts": counts,
        "top": [{k: r.get(k) for k in (
            "rank", "eval_id", "batch", "origin", *PARAMS, "loss", "rmse", "shift",
            "q1_tgt", "q1_sim", "q1_ratio", "aligned", "dispersed_sim", "anchor_scale",
            "rmax_over_r0", "n_pairs_below_rmin", "dataset_dir", "gsd", "gsd_md5")}
            for r in top],
        "batches": batches,
        "first_batch": first_batch,
        "timing": timing,
        "gp_lengthscale": lengthscales,
        "gp_param_names": names,
        "evals": [{k: v for k, v in r.items() if not k.startswith("_")} for r in evals],
    }


# ------------------------------------------------------------------ writing


EVAL_COLUMNS = ("eval_id", "status", "batch", "origin", *PARAMS, "loss", "rmse", "shift",
                "rank", "cum_best", "q1_tgt", "q1_sim", "q1_ratio", "aligned",
                "dispersed_sim", "anchor_scale", "rmax_over_r0", "n_pairs_below_rmin",
                "submit", "start", "end", "n_attempts", "queue_s", "worker_s", "sim_s",
                "sq_s", "total_s", "slurm_elapsed", "host", "gpu", "partition", "tps",
                "timing_source", "flag", "reason", "dataset_dir")


def _write_csv(path, rows, columns):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


def _fmt(value, spec=".4g"):
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _hours(seconds):
    return "-" if seconds is None else f"{seconds / 3600:.2f} h"


def slow_evals(summary, slow_hours):
    rows = []
    for r in summary["evals"]:
        if r["total_s"] is None or r["total_s"] < slow_hours * 3600:
            continue
        parts = {"queue": r["queue_s"] or 0.0, "simulation": r["sim_s"] or 0.0,
                 "S(q)": r["sq_s"] or 0.0}
        dominant = max(parts, key=parts.get)
        rows.append({**{k: r[k] for k in ("eval_id", "batch", "status", "total_s", "queue_s",
                                          "sim_s", "sq_s", "worker_s", "rmax_over_r0",
                                          "tps", "host", "gpu", "n_attempts", "U0", "r0",
                                          "n", "m")},
                     "dominant": dominant if parts[dominant] >= 0.5 * r["total_s"] else "mixed"})
    return rows


def summary_text(s, slow_hours):
    c = s["counts"]
    lines = [
        f"Inverse run pack summary ({s['written']})",
        f"out_root     : {s['out_root']}",
        f"description  : {s['description']}",
        f"target       : {(s['target'] or {}).get('path')}  md5 {(s['target'] or {}).get('md5')}",
        f"commit(s)    : {', '.join(s['commits']) or 'unknown'}; starts: {len(s['starts'])}",
        f"evaluations  : {c['ok']} ok of max_evals={s['max_evals']} (batch_size "
        f"{s['batch_size']}); FAILED {c['FAILED']}, METRIC_FAILED {c['METRIC_FAILED']}, "
        f"ABANDONED {c['ABANDONED']}, no block yet {c['no_block']}; retried jobs {c['retried']}"
        + ("" if s["complete"] else "  (not complete)"),
        f"dispersed    : {c['dispersed_sim_ok']} ok evaluations with a dispersed simulated "
        f"S(q); {c['not_aligned_ok']} not aligned",
        "",
        "Top evaluations (loss = M4 + |ln q1 ratio|):",
        "  rank eval batch origin      U0      r0       n       m   m_rel     loss       M4"
        "    shift  q1_ratio",
    ]
    for r in s["top"]:
        lines.append(
            f"  {r['rank']:>4} {r['eval_id']:>4} {r['batch']:>5} {r['origin']:>6} "
            + " ".join(f"{_fmt(r[p], '.4f'):>7}" for p in PARAMS)
            + f" {_fmt(r['loss'], '.5f'):>8} {_fmt(r['rmse'], '.5f'):>8} "
              f"{_fmt(r['shift'], '.5f'):>8} {_fmt(r['q1_ratio'], '.5f'):>9}")
    lines += ["", "Batches (cumulative best after each batch):",
              "  batch  n_ok  failed   best(eval)        median    cum_best    wall"]
    for b in s["batches"]:
        lines.append(
            f"  {b['batch']:>5} {b['n_ok']:>5} {b['n_failed']:>7}   "
            f"{_fmt(b['best_loss'], '.5f')} ({_fmt(b['best_eval'])})".ljust(36)
            + f"{_fmt(b['median_loss'], '.5f'):>8}  {_fmt(b['cum_best'], '.5f'):>10}  "
              f"{_hours(b['wall_s'])}")
    fb = s["first_batch"]
    lines += ["", "First batch (batch 0: initial point + Sobol points):"]
    for r in fb["evals"]:
        lines.append(f"  eval {r['eval_id']:>3} {r['origin']:>6}  loss {_fmt(r['loss'], '.5f')}"
                     f"  rank {_fmt(r['rank'])}  " + "  ".join(
                         f"{p}={_fmt(r[p], '.4g')}" for p in PARAMS))
    lines.append(f"  best of batch 0: eval {fb['best_eval']} loss {_fmt(fb['best_loss'], '.5f')}; "
                 f"{fb['n_later_better_than_first_best']} of {fb['n_later']} later evaluations "
                 f"are better; top-k from batch 0: {fb['top_k_from_first_batch']}")
    t = s["timing"]
    lines += ["", f"Timing (source: {t['source']}; hours):",
              "  part        n    min  median    p90    max"]
    for key, label in (("total_s", "total"), ("queue_s", "queue"), ("worker_s", "worker"),
                       ("sim_s", "simulation"), ("sq_s", "S(q)")):
        st = t[key]
        if st:
            lines.append(f"  {label:<10} {st['n']:>3} " + " ".join(
                f"{st[k] / 3600:>6.2f}" for k in ("min", "median", "p90", "max")))
    corr = t["sim_s_vs_rmax_over_r0_cubed"]
    lines.append(f"  simulation time vs (rmax/r0)^3: n={corr['n']}, Pearson "
                 f"{_fmt(corr['pearson'], '.3f')}, Spearman {_fmt(corr['spearman'], '.3f')}")
    for gpu, st in t["by_gpu"].items():
        if st:
            lines.append(f"  {gpu}: simulation median {st['median'] / 3600:.2f} h (n={st['n']})")
    slow = slow_evals(s, slow_hours)
    lines.append(f"  evaluations with total >= {slow_hours} h: {len(slow)}")
    for r in slow:
        lines.append(
            f"    eval {r['eval_id']:>3} (batch {r['batch']}): total {_hours(r['total_s'])} = "
            f"queue {_hours(r['queue_s'])} + simulation {_hours(r['sim_s'])} + S(q) "
            f"{_hours(r['sq_s'])}; dominant {r['dominant']}; rmax/r0 "
            f"{_fmt(r['rmax_over_r0'], '.3f')}, TPS {_fmt(r['tps'], '.0f')}, {r['gpu']} on "
            f"{r['host']}, attempts {r['n_attempts']}")
    lines += ["", "GP lengthscale per batch (unit cube; *_phys = times the bound width):"]
    names = s["gp_param_names"]
    if names:
        lines.append("  batch stage        n_train " + " ".join(f"{n:>8}" for n in names))
        for row in s["gp_lengthscale"]:
            lines.append(f"  {_fmt(row['batch']):>5} {row['stage']:<12} {_fmt(row['n_train']):>7} "
                         + " ".join(f"{_fmt(row.get(f'ls_{n}'), '.3f'):>8}" for n in names))
    else:
        lines.append("  no gp_log rows")
    failed = [r for r in s["evals"] if r["status"] in ("FAILED", "METRIC_FAILED")]
    if failed:
        lines += ["", "Failed evaluations:"]
        lines += [f"  eval {r['eval_id']:>3} {r['status']}: {r['reason']}" for r in failed]
    return "\n".join(lines) + "\n"


def pack_files(out_root, top_ids):
    """``[(path, arcname under Optimization_Results/)]`` of every file to pack."""
    out_root = Path(out_root)
    files = []

    def add(path):
        if path.is_file():
            files.append((path, path.relative_to(out_root).as_posix()))

    for name in RUN_FILES:
        add(out_root / name)
    gp_dir = out_root / "gp_log"
    if gp_dir.is_dir():
        for path in sorted(gp_dir.rglob("*")):
            add(path)
    for eid, eval_dir in eval_dirs(out_root).items():
        for ds in dataset_dirs(eval_dir):
            for pattern in ("S(q)_data/*.npy", "shift_rmse_diagnostics.json",
                            "sim_params_*.csv", *FLAGS):
                for path in sorted(ds.glob(pattern)):
                    add(path)
            if eid in top_ids:
                for pattern in ("DNA_assembly_*.gsd", "potential_energy.csv", "*.png",
                                "S(q)_data/*.png"):
                    for path in sorted(ds.glob(pattern)):
                        add(path)
        jobs = eval_dir / "_jobs"
        if jobs.is_dir():
            for pattern in ("**/summary.json", "**/launcher.log", "**/*.out"):
                for path in sorted(jobs.glob(pattern)):
                    add(path)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_root", help="the run's out_root (Optimization_Results)")
    ap.add_argument("--top", type=int, default=3, help="evaluations whose GSD is packed")
    ap.add_argument("--name", default=None,
                    help="pack name and top folder in the tar (default: parent folder name)")
    ap.add_argument("--dest", default=None, help="output folder (default: parent of out_root)")
    ap.add_argument("--slow-hours", type=float, default=1.5,
                    help="list evaluations whose total time is at least this")
    ap.add_argument("--no-sacct", action="store_true", help="time from files only")
    ap.add_argument("--summary-only", action="store_true", help="write the summary, no tar")
    a = ap.parse_args(argv)

    out_root = Path(a.out_root)
    if not (out_root / "bo_trajectory.csv").is_file():
        raise SystemExit(f"{out_root}: no bo_trajectory.csv")
    name = a.name or out_root.resolve().parent.name
    dest = Path(a.dest) if a.dest else out_root.resolve().parent
    summary_dir = dest / f"{name}_pack_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    s = collect(out_root, a.top, use_sacct=not a.no_sacct)
    s["slow_evals"] = slow_evals(s, a.slow_hours)
    s["slow_hours"] = a.slow_hours
    (summary_dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    text = summary_text(s, a.slow_hours)
    (summary_dir / "summary.txt").write_text(text)
    _write_csv(summary_dir / "evals.csv", s["evals"], EVAL_COLUMNS)
    _write_csv(summary_dir / "batches.csv",
               [{**b, "eval_ids": " ".join(map(str, b["eval_ids"]))} for b in s["batches"]],
               ("batch", "eval_ids", "n_ok", "n_failed", "best_loss", "best_eval",
                "median_loss", "cum_best", "wall_s", "slowest_eval"))
    ls_cols = ["batch", "stage", "attempt", "n_train", "outputscale", "noise",
               "best_observed_loss"]
    for n in s["gp_param_names"]:
        ls_cols += [f"ls_{n}", f"ls_{n}_phys"]
    _write_csv(summary_dir / "gp_lengthscale.csv", s["gp_lengthscale"], ls_cols)
    print(text)

    top_ids = {r["eval_id"] for r in s["top"]}
    files = pack_files(out_root, top_ids)
    total = sum(p.stat().st_size for p, _ in files)
    manifest = [f"{p.stat().st_size}\t{arc}" for p, arc in files]
    (summary_dir / "manifest.txt").write_text(
        f"# {len(files)} files, {total / 1e6:.1f} MB before compression, "
        f"under {name}/Optimization_Results/\n" + "\n".join(manifest) + "\n")
    print(f"summary -> {summary_dir}")
    if a.summary_only:
        return 0
    tar_path = dest / f"{name}_pack.tar.gz"
    tmp = tar_path.with_name(tar_path.name + ".part")
    with tarfile.open(tmp, "w:gz") as tar:
        for path in sorted(summary_dir.iterdir()):
            tar.add(path, arcname=f"{name}/pack_summary/{path.name}")
        for path, arc in files:
            tar.add(path, arcname=f"{name}/Optimization_Results/{arc}")
    os.replace(tmp, tar_path)
    print(f"tar     -> {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB; "
          f"{len(files)} run files, {total / 1e6:.1f} MB before compression)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
