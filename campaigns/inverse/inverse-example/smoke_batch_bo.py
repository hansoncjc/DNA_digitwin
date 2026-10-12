"""
Cluster smoke tests for batch BO (C24) and a cut gp_log.jsonl line (C22).

``run`` replaces HOOMD and saxs-fft with a cheap analytic model: the
"simulated" S(q) is an FCC-like curve whose first peak, height and width
follow r0, U0 and n; the loss is the real shift_rmse against a target from
the same model. Everything else (objective, trajectory, resume, GP log) is
the code of the repo given by --repo. Needs the cluster venv (botorch) and
MC-DFM on PYTHONPATH ($MC_DFM_ROOT), like inverse.py.

    python smoke_batch_bo.py run --repo REPO --out DIR --max-evals N [--batch-size B] [--seed S]
    python smoke_batch_bo.py compare DIR_OLD DIR_NEW            # batch_size=1 equivalence
    python smoke_batch_bo.py check-run DIR --max-evals N --batch-size B [--sobol-seed S]
    python smoke_batch_bo.py cut-gp-log DIR                     # cut the last gp_log line
    python smoke_batch_bo.py check-cut DIR                      # after the resume

``run`` works with the API before C24 (n_iters = max_evals - 1, batch size 1)
and after it. ``check-run`` also checks an inverse.py run (out_root).
Every check prints PASS or FAIL; the exit code is 1 if any check failed.
"""
import argparse
import csv
import importlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NEW_REPO = HERE.parents[2]

SEARCH = {
    "U0":    {"bounds": (0.5, 8.0),  "init": 3.0},
    "r0":    {"bounds": (2.0, 3.5),  "init": 2.5},
    "n":     {"bounds": (5.5, 15.0), "init": 12.0},
    "m_rel": {"bounds": (0.0, 1.0),  "init": 2.0 / 7.0},
    "density": {"fixed": 0.005},
}
N = 5017
TARGET = {"r0": 2.8, "U0": 4.0, "n": 10.0}
FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def _import_bo(repo):
    repo = str(Path(repo).resolve())
    sys.path.insert(0, repo)
    mc_dfm = os.environ.get("MC_DFM_ROOT", "")
    if mc_dfm:
        sys.path.insert(1, mc_dfm)
    bo = importlib.import_module("bo")
    if not str(Path(bo.__file__).resolve()).startswith(repo):
        raise RuntimeError(f"imported {bo.__file__}, not from {repo}")
    return bo


# ------------------------------------------------------------ analytic model


def _q_grid():
    """saxs-fft q grid of the inverse targets (N_grid 600, trim 3 bins per end)."""
    dq = 2.0 * np.pi / 100.0
    edges = np.arange(0.0, np.sqrt(3.0) * 300 * dq + dq, dq)
    return (0.5 * (edges[:-1] + edges[1:]))[3:-3] / 246.0


def model_curve(r0, U0, n):
    q = _q_grid()
    q1 = 0.0129 * 2.5 / float(r0)
    amp = 1.0 + 1.5 * float(U0)
    width = 3e-4 * (1.0 + 6.0 / float(n))
    s = 1.0 + 200.0 * np.exp(-q / 0.0012)
    for i, ratio in enumerate((1.0, 1.1547, 1.6330, 1.9149, 2.0, 2.3094)):
        s = s + amp / (1.0 + 0.6 * i) * np.exp(-0.5 * ((q - q1 * ratio) / width) ** 2)
    return np.column_stack([q, s])


def _patch_forward_model(bo):
    from simulation import resolve_table_bounds

    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        Path(outdir, "analytic_params.json").write_text(json.dumps(
            {"U0": U_0, "r0": r0, "n": n, "m": m}))
        b = resolve_table_bounds(U_0, n, m, r0, N=kw.get("N", N), density=density)
        return {"rmax": b["rmax"], "t_tol": b["t_tol"], "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        p = json.loads(Path(save_dir, "analytic_params.json").read_text())
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", model_curve(p["r0"], p["U0"], p["n"]))

    bo.run_simulation = fake_sim
    bo.convert_to_SAXS_fft = fake_saxs


def cmd_run(a):
    import torch
    torch.set_num_threads(1)          # same arithmetic in both runs of a comparison
    bo = _import_bo(a.repo)
    _patch_forward_model(bo)
    from datasets import Dataset, ExperimentalParams, SimulationParams

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "target_sq.npy"
    if not target.exists():
        np.save(target, model_curve(**TARGET))
    ds = Dataset(id="d0", exp_path=str(target), exp=ExperimentalParams(),
                 sim=SimulationParams(density=0.005), datatype="sq")
    ps = bo.ParamSpace(SEARCH)
    obj = bo.make_global_objective(
        [ds], ps, ffpath=str(Path(a.repo) / "formfactors" / "sasmodels_sphere_fit.txt"),
        out_root=str(out), trim_tail=0, sim_defaults={"N": N}, mode="sim",
        scattering_method="saxsfft", scattering_kwargs={}, metric="shift_rmse",
        compare_q_range=None, metric_kwargs={}, parallel=False,
    )
    params = inspect.signature(bo.run_bo_resumable).parameters
    if "max_evals" in params:
        kw = {"max_evals": a.max_evals, "batch_size": a.batch_size}
    else:
        if a.batch_size != 1:
            sys.exit("this repo has no batch BO (n_iters API): use --batch-size 1")
        kw = {"n_iters": a.max_evals - 1}
    print(f"[smoke] repo {a.repo} (bo from {bo.__file__}); {kw}; seed {a.seed}")
    best, history = bo.run_bo_resumable(obj, ps, "", str(out), seed=a.seed,
                                        gp_log_dir=str(out / "gp_log"), **kw)
    info = {"repo": str(a.repo), "max_evals": a.max_evals,
            "batch_size": a.batch_size, "seed": a.seed, "n_history": len(history),
            "best_loss": min(history)}
    with open(out / "smoke_runs.jsonl", "a") as fh:
        fh.write(json.dumps(info) + "\n")
    print(f"[smoke] {len(history)} successful evaluations; best loss {min(history):.6f}")


# ------------------------------------------------------------ checks


def _block_statuses(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("# Iteration"):
                eval_id = int(row[0].split()[-1])
                if eval_id in out:
                    out.setdefault("_dup", []).append(eval_id)
                out[eval_id] = "ok" if row[1] == "total_loss" else row[2]
    return out


def _sim_params(run, eval_id):
    path = Path(run) / f"eval_{eval_id:03d}" / "d0" / "sim_params_d0.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        return next(csv.DictReader(fh))


def cmd_compare(a):
    import torch
    bo = _import_bo(NEW_REPO)
    old, new = Path(a.old), Path(a.new)
    for name in ("bo_trajectory.csv", "loss_components.txt"):
        same = (old / name).read_bytes() == (new / name).read_bytes()
        detail = ""
        if not same:
            lo, ln = (old / name).read_text().splitlines(), (new / name).read_text().splitlines()
            i = next((k for k, (x, y) in enumerate(zip(lo, ln)) if x != y), min(len(lo), len(ln)))
            detail = f"first difference at line {i + 1}"
        check(f"{name} identical", same, detail)
    dirs_old = sorted(p.name for p in old.glob("eval_*"))
    dirs_new = sorted(p.name for p in new.glob("eval_*"))
    check("same eval folders", dirs_old == dirs_new, f"{len(dirs_old)} vs {len(dirs_new)}")
    same_params = all(
        (old / d / "d0" / "sim_params_d0.csv").read_bytes()
        == (new / d / "d0" / "sim_params_d0.csv").read_bytes()
        for d in dirs_old if (old / d / "d0" / "sim_params_d0.csv").exists())
    check("candidates (sim_params_d0.csv of every eval) identical", same_params)

    rows_old = bo.read_gp_log(old / "gp_log")
    rows_new = bo.read_gp_log(new / "gp_log")
    check("gp_log versions 1 (old) and 2 (new)",
          {r["gp_log_version"] for r in rows_old} == {1}
          and {r["gp_log_version"] for r in rows_new} == {2})
    check("same number of gp_log rows", len(rows_old) == len(rows_new),
          f"{len(rows_old)} vs {len(rows_new)}")
    skip = {"gp_log_version", "units"}
    diffs = []
    for i, (ro, rn) in enumerate(zip(rows_old, rows_new)):
        keys = (set(ro) | set(rn)) - skip
        bad = sorted(k for k in keys if ro.get(k) != rn.get(k))
        if bad:
            diffs.append((i, bad))
    check("gp_log rows equal except gp_log_version and units (v1 read as v2)", not diffs,
          f"rows/keys differing: {diffs[:3]}")
    state_diffs = []
    for i, (ro, rn) in enumerate(zip(rows_old, rows_new)):
        s_old, s_new = (bo._torch_load(Path(d) / "gp_log" / r["state_file"])
                        for d, r in ((old, ro), (new, rn)))
        same = (torch.equal(s_old["train_x"], s_new["train_x"])
                and torch.equal(s_old["train_y"], s_new["train_y"])
                and s_old["state_dict"].keys() == s_new["state_dict"].keys()
                and all(torch.equal(s_old["state_dict"][k], s_new["state_dict"][k])
                        for k in s_old["state_dict"]))
        if not same:
            state_diffs.append(i)
    check("GP states (train_x, train_y, state_dict) identical", not state_diffs,
          f"rows {state_diffs[:5]}")


def cmd_check_run(a):
    import torch
    bo = _import_bo(NEW_REPO)
    run = Path(a.dir)
    statuses = _block_statuses(run / "bo_trajectory.csv")
    dups = statuses.pop("_dup", [])
    ids = sorted(statuses)
    check("every eval id has exactly one trajectory block", not dups, f"duplicates {dups}")
    check("eval ids are consecutive from 0", ids == list(range(len(ids))), f"{ids[:3]}..{ids[-3:]}")
    folders = sorted(int(p.name[5:]) for p in run.glob("eval_*") if p.name[5:].isdigit())
    check("eval folders match the blocks", folders == ids)
    ok_ids = [i for i in ids if statuses[i] == "ok"]
    abandoned = [i for i in ids if statuses[i] == "ABANDONED"]
    failed = [i for i in ids if statuses[i] == "FAILED"]
    check(f"successful evaluations == max_evals={a.max_evals}", len(ok_ids) == a.max_evals,
          f"{len(ok_ids)} successful, {len(failed)} failed, abandoned {abandoned}")
    check("abandoned folders carry the ABANDONED flag",
          all((run / f"eval_{i:03d}" / "ABANDONED").exists() for i in abandoned))

    gp_dir = Path(a.gp_log) if a.gp_log else run / "gp_log"
    rows = bo.read_gp_log(gp_dir)
    acq = [r for r in rows if r["stage"] == "acquisition"]
    finals = [r for r in rows if r["stage"] == "final"]
    check("gp_log rows are version 2", {r["gp_log_version"] for r in rows} == {2})
    check("batch numbers 1..K in order", [r["batch"] for r in acq] == list(range(1, len(acq) + 1)))
    first = [i for i in ids if not acq or i < acq[0]["eval_ids"][0]]
    if all(statuses[i] == "ok" for i in first):
        check(f"first batch: {min(a.batch_size, a.max_evals)} evals before the first GP",
              len(first) == min(a.batch_size, a.max_evals), f"eval ids {first}")
    else:
        print(f"note  first batch had failures: eval ids {first} {[statuses[i] for i in first]}")
    seen, problems = set(), []
    for r in acq:
        e = r["eval_ids"]
        if set(e) & seen or e != list(range(e[0], e[0] + len(e))):
            problems.append(f"batch {r['batch']}: eval_ids {e}")
        seen |= set(e)
        n_before = sum(1 for i in ok_ids if i < e[0])
        if r["n_train"] != n_before:
            problems.append(f"batch {r['batch']}: n_train {r['n_train']} != {n_before} "
                            "successful evals before it")
        if r["q"] != len(e) or r["q"] != min(a.batch_size, a.max_evals - r["n_train"]):
            problems.append(f"batch {r['batch']}: q={r['q']}")
        if len(r["acq_value_steps"]) != r["q"] or r["acq_value"] != r["acq_value_steps"][-1]:
            problems.append(f"batch {r['batch']}: acq_value_steps")
        for eval_id, phys in zip(e, r["candidates_phys"]):
            sp = _sim_params(run, eval_id)
            if sp is None:
                continue
            G = dict(zip(r["param_names"], phys))
            if any(abs(float(sp[k]) - G[k]) > 1e-9 * max(1.0, abs(G[k])) for k in G):
                problems.append(f"eval {eval_id}: candidate differs from sim_params")
    check("each batch: eval ids fresh and consecutive; GP refit on every success before it; "
          "q = min(batch_size, max_evals - n_train); candidates are the evaluated points",
          not problems, "; ".join(problems[:4]))
    check("final GP row trained on max_evals points",
          bool(finals) and finals[-1]["n_train"] == a.max_evals,
          f"{[f['n_train'] for f in finals]}")
    post_bad = []
    for r in acq:
        X = torch.tensor(r["candidates_unit"], dtype=torch.float64)
        post = bo.load_gp_posterior(gp_dir / r["state_file"], X)
        if (np.abs(post["mean_loss"].detach().numpy() - r["candidates_post_mean_loss"]).max() > 1e-6
                or np.abs(post["std_loss"].detach().numpy()
                          - r["candidates_post_std_loss"]).max() > 1e-6):
            post_bad.append(r["batch"])
    check("load_gp_posterior rebuilds every batch's posterior at its candidates", not post_bad,
          f"batches {post_bad}")

    if a.sobol_seed is not None:
        from torch.quasirandom import SobolEngine
        ps = bo.ParamSpace(SEARCH)
        warm = bo.load_warm_start_from_trajectory(run / "bo_trajectory.csv", ps)
        X = warm[0][: a.batch_size]
        expected = torch.cat([ps.init_unit().unsqueeze(0),
                              SobolEngine(ps.d, scramble=True, seed=a.sobol_seed)
                              .draw(a.batch_size - 1, dtype=torch.float64)])
        check(f"first batch = initial point + {a.batch_size - 1} scrambled Sobol points "
              f"(seed {a.sobol_seed})", torch.allclose(X, expected, atol=1e-9),
              f"max diff {float((X - expected).abs().max()):.2e}")


def cmd_cut_gp_log(a):
    gp = Path(a.dir) / "gp_log" / "gp_log.jsonl"
    data = gp.read_bytes()
    shutil.copy2(gp, gp.with_name("gp_log.jsonl.before_cut"))
    body = data.rstrip(b"\n")
    start = body.rfind(b"\n") + 1
    last = body[start:]
    fragment = last[: len(last) // 2]
    gp.write_bytes(data[:start] + fragment)          # as a kill mid-write leaves it
    (Path(a.dir) / "cut_fragment.txt").write_bytes(fragment)
    line = data[:start].count(b"\n") + 1
    print(f"[smoke] cut line {line} of {gp} from {len(last)} to {len(fragment)} bytes "
          "(no newline); copy in gp_log.jsonl.before_cut")


def cmd_check_cut(a):
    bo = _import_bo(NEW_REPO)
    gp_dir = Path(a.dir) / "gp_log"
    fragment = (Path(a.dir) / "cut_fragment.txt").read_bytes().decode()
    moved = [json.loads(s) for s in (gp_dir / "gp_log_truncated.jsonl").read_text().splitlines()]
    check("the cut line is in gp_log_truncated.jsonl, text exactly as cut",
          len(moved) == 1 and moved[0]["text"] == fragment and moved[0]["had_newline"] is False,
          f"{len(moved)} record(s)")
    data = (gp_dir / "gp_log.jsonl").read_bytes()
    check("gp_log.jsonl ends with a newline and every line parses",
          data.endswith(b"\n")
          and all(json.loads(s) or True for s in data.splitlines() if s.strip()))
    rows = bo.read_gp_log(gp_dir)
    acq = [r["batch"] for r in rows if r["stage"] == "acquisition"]
    check("batch numbers continue after the resume", acq == list(range(1, len(acq) + 1)), f"{acq}")
    before = (gp_dir / "gp_log.jsonl.before_cut").read_bytes().splitlines()
    check("every complete line from before the cut is still there, in place",
          data.splitlines()[: len(before) - 1] == before[:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-evals", type=int, required=True)
    r.add_argument("--batch-size", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    k = sub.add_parser("check-run")
    k.add_argument("dir")
    k.add_argument("--max-evals", type=int, required=True)
    k.add_argument("--batch-size", type=int, required=True)
    k.add_argument("--sobol-seed", type=int, default=None)
    k.add_argument("--gp-log", default=None, help="default DIR/gp_log")
    u = sub.add_parser("cut-gp-log")
    u.add_argument("dir")
    v = sub.add_parser("check-cut")
    v.add_argument("dir")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "check-run": cmd_check_run,
     "cut-gp-log": cmd_cut_gp_log, "check-cut": cmd_check_cut}[a.cmd](a)
    if FAILED:
        print(f"\n{len(FAILED)} check(s) FAILED")
        return 1
    if a.cmd not in ("run", "cut-gp-log"):
        print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
