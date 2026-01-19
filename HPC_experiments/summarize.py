import argparse
import json
import csv
import re, math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
try:
    from scipy import stats as _stats  # type: ignore
except Exception:
    _stats = None

try:
    import mpmath as _mp  # type: ignore
except Exception:
    _mp = None

RUNS_ROOT = "nanogpt-runs"
SUMMARY = f"{RUNS_ROOT}/summary.csv"

ARGS = None


# =======================================================

def ensure_dir(path:str | Path, die:bool = True) -> Path:
    path = Path(path)
    
    if not path.exists() or not path.is_dir():
        if die:
            raise Exception(f"No such directory: {path}")
        path.mkdir(parents=True)
            
    return path

def ensure_file(path: str | Path, die:bool = False) -> Path:
    path = Path(path)
    
    if not path.exists() or not path.is_file():
        if die:
            raise Exception(f"No such file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    
    return path 


def _to_int(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _to_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


# =======================================================


def check_arguments() -> None:
    global RUNS_ROOT
    global SUMMARY
    global ARGS
    
    ap = argparse.ArgumentParser()    
    ap.add_argument("--runs-root", type=str, default=RUNS_ROOT)
    ap.add_argument("--summary", type=str, default=SUMMARY)
    
    ap.add_argument("--mode", type=str, choices=["summarize", "filter"], default="summarize")
    ap.add_argument("--baseline", type=str, default="", help="Baseline row value (by --baseline-col)")
    ap.add_argument("--baseline-col", type=str, default="key")
    ap.add_argument("--n-col", type=str, default="cnt")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="", help="Output path for filter mode")
    ap.add_argument("--no-test-std", action="store_true")
    ap.add_argument("--correction", type=str, choices=["none", "bh"], default="none")
    
    args = ap.parse_args()
    ARGS = args
    
    RUNS_ROOT = args.runs_root
    SUMMARY = args.summary
    
    if args.mode == "summarize":
        ensure_dir(RUNS_ROOT, die=True)
        ensure_file(SUMMARY)
    else:
        ensure_file(SUMMARY, die=True)


# =======================================================

SUMMARY_HEADERS = [
    # ---- main info ----
    "key",              
    "name",
    "desc",               
    "cnt",              
    "run_dir",          
    "dataset",
    "config",
    "git_commit",
    "git_diff_path",
    
    # ---- tecnical----
    "gpus",
    "gpu",
    "nvidia-smi",

    # ---- quality aggregated ----
    "best_val_loss_mean",
    "best_val_loss_std",
    "best_val_loss_min",
    "best_val_loss_max",
    "final_val_loss_mean",
    "final_val_loss_std",

    "best_val_ppl_mean",
    "best_val_ppl_std",
    "final_val_ppl_mean",
    "final_val_ppl_std",

    "best_step_median",
    "gen_gap_final_mean",
    "gen_gap_final_std",

    # ---- performance aggregated ----
    "tokens_per_sec_mean",
    "tokens_per_sec_std",
    "mfu_mean",
    "mfu_std",
    "iter_time_ms_mean",
    "iter_time_ms_std",
]

_INT_FIELDS = {"cnt", "gpus"}
_STR_FIELDS = {"key", "name", "desc", "git_commit", "gpu"}
_PATH_FIELDS = {"run_dir", "dataset", "config", "git_diff_path", "nvidia-smi"}
# else - float

@dataclass
class RunSummary:
    data: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load_from_csv_row(cls, row:dict[str, str]) -> "RunSummary":
        d: dict[str, Any] = {}

        # normalize row values
        norm = {k: (v or "") for k, v in row.items()}

        for h in SUMMARY_HEADERS:
            raw = (norm.get(h) or "").strip()

            if h in _INT_FIELDS:
                d[h] = _to_int(raw)
            elif h in _STR_FIELDS:
                d[h] = raw
            elif h in _PATH_FIELDS:
                d[h] = Path(raw)
            else:
                d[h] = _to_float(raw)

        # если в CSV вдруг есть дополнительные колонки — сохраним тоже
        for k, v in norm.items():
            if k not in d:
                d[k] = (v or "").strip()

        return cls(data=d)
    
    def to_csv_row(self, headers: list[str]) -> dict[str, str]:
        return {h: _fmt(self.data.get(h)) for h in headers}
    

@dataclass
class Summary:
    items: dict[str, RunSummary] = field(default_factory=dict)
    headers: list[str] = field(default_factory=lambda: list(SUMMARY_HEADERS))
    
    def load_from_path(self, path:str | Path):
        path = Path(path)
        
        self.items.clear()
        self.headers = list(SUMMARY_HEADERS)
        
        if not path.exists():
            return
        
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            file_headers = list(reader.fieldnames or [])
            if file_headers:
                merged = []
                seen = set()
                for h in file_headers:
                    if h and h not in seen:
                        merged.append(h)
                        seen.add(h)
                for h in SUMMARY_HEADERS:
                    if h not in seen:
                        merged.append(h)
                        seen.add(h)
                self.headers = merged

            for row in reader:
                rs = RunSummary.load_from_csv_row(row)
                key = (rs.data.get("key") or "").strip()
                if not key:
                    continue
                self.items[key] = rs
    
    def save_to(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        all_cols = set(self.headers)
        for rs in self.items.values():
            all_cols |= set(rs.data.keys())
            
        headers = list(self.headers)
        for h in sorted(all_cols):
            if h not in headers:
                headers.append(h)
                
        rows = list(self.items.values())
        rows.sort(key=lambda rs: ((rs.data.get("timestamp") or ""), (rs.data.get("key") or "")))

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for rs in rows:
                w.writerow(rs.to_csv_row(headers))
    
    def check_presents(self, key:str):
        return key in self.items
    
    def add(self, run:RunSummary):
        key = (run.data.get("key") or "").strip()
        if not key:
            raise ValueError("RunSummary has empty 'key'")
        self.items[key] = run


# =======================================================
    
    
def collect_runs() -> list[tuple[str, list[Path]]]:
    root = ensure_dir(RUNS_ROOT, die=True)

    def read_meta(run_dir: Path) -> dict[str, Any] | None:
        mp = run_dir / "meta" / "run_meta.json"
        if not mp.is_file():
            return None
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None

    def parse_from_dirname(name: str) -> tuple[str, str, str, str]:
        """
        Expected by template.sbatch:
          RUN_NAME = "{JOBID}_{TASKID}_{EXP_NAME}_{TS}"
          TS = "%Y-%m-%d_%H-%M-%S"
        EXP_NAME may contain underscores -> parse from the end.
        Returns: (job_id, task_id, exp_name, timestamp)
        """
        parts = name.split("_")
        if len(parts) < 5:
            return ("", "", "", "")
        job_id = parts[0]
        task_id = parts[1]
        timestamp = f"{parts[-2]}_{parts[-1]}"
        exp_name = "_".join(parts[2:-2])
        return (job_id, task_id, exp_name, timestamp)

    def get_group_key_and_ts(run_dir: Path) -> tuple[str, str]:
        meta = read_meta(run_dir)
        if meta:
            exp_name = str(meta.get("exp_name") or "").strip()
            timestamp = str(meta.get("timestamp") or "").strip()
            slurm = meta.get("slurm") if isinstance(meta.get("slurm"), dict) else {}
            array_job_id = str((slurm or {}).get("array_job_id") or (slurm or {}).get("job_id") or "").strip()

            if exp_name and array_job_id:
                return (f"{array_job_id}_{exp_name}".strip("_"), timestamp)

        # fallback: parse directory name
        job_id, _task_id, exp_name, timestamp = parse_from_dirname(run_dir.name)
        if exp_name and job_id:
            return (f"{job_id}_{exp_name}".strip("_"), timestamp)

        # last resort: just use folder name as key (rare)
        return (run_dir.name, "")

    def sort_key_inside_group(run_dir: Path) -> tuple[int, int, str]:
        meta = read_meta(run_dir) or {}
        seed_idx = meta.get("seed_idx", None)
        slurm = meta.get("slurm") if isinstance(meta.get("slurm"), dict) else {}
        task_id = (slurm or {}).get("task_id", None)

        def to_int(x: Any, default: int) -> int:
            try:
                if x is None:
                    return default
                return int(x)
            except Exception:
                return default

        return (to_int(seed_idx, 10**9), to_int(task_id, 10**9), run_dir.name)

    def parse_dt(ts: str) -> datetime:
        try:
            return datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
        except Exception:
            return datetime.max

    # --- group folders ---
    groups: dict[str, dict[str, Any]] = {}  # key -> {"folders": [Path], "timestamps":[str]}
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue

        key, ts = get_group_key_and_ts(p)
        bucket = groups.setdefault(key, {"folders": [], "timestamps": []})
        bucket["folders"].append(p)
        bucket["timestamps"].append(ts)

    # --- build output ---
    out: list[tuple[str, list[Path], datetime]] = []
    for key, v in groups.items():
        folders: list[Path] = v["folders"]
        folders.sort(key=sort_key_inside_group)

        # group sort timestamp: take earliest parseable
        dts = [parse_dt(t) for t in v["timestamps"] if t]
        group_dt = min(dts) if dts else datetime.max

        out.append((key, folders, group_dt))

    out.sort(key=lambda x: (x[2], x[0]))
    return [(key, folders) for key, folders, _dt in out]

# =======================================================

RE_STEP = re.compile(
    r"^step\s+(\d+):\s*train loss\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*val loss\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE | re.MULTILINE,
)
RE_ITER = re.compile(
    r"^iter\s+(\d+):.*?time\s*([0-9]+(?:\.[0-9]+)?)ms,\s*mfu\s*(-?[0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE | re.MULTILINE,
)
RE_TOKENS = re.compile(
    r"tokens per iteration will be:\s*([0-9,\s]+)",
    re.IGNORECASE,
)
RE_GPU_NAME = re.compile(r"^\|\s*\d+\s+(.+?)\s{2,}On\b", re.MULTILINE)


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None

def _std(xs: list[float]) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)  # sample std
    return math.sqrt(var)

def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return 0.5 * (ys[mid - 1] + ys[mid])

def _safe_exp(x: float | None) -> float | None:
    if x is None:
        return None
    # loss обычно маленький, но на всякий
    if x > 50:
        return float("inf")
    return math.exp(x)


def read_meta(run_dir: Path) -> dict[str, Any] | None:
    mp = run_dir / "meta" / "run_meta.json"
    if not mp.is_file():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None
    
def parse_gpu_from_nvidia_smi(path: Path) -> tuple[int | None, str]:
    """
    Returns (gpus_count, gpu_name).
    gpu_name: one name or ';' joined unique names.
    """
    if not path.is_file():
        return None, ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = [ " ".join(m.group(1).split()) for m in RE_GPU_NAME.finditer(text) ]
    if not names:
        return None, ""
    uniq = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return len(names), (uniq[0] if len(uniq) == 1 else ";".join(uniq))


def parse_stdout_metrics(stdout_path: Path) -> dict[str, Any]:
    """
    Parse per-seed metrics from logs/stdout.log.
    """
    out: dict[str, Any] = {
        "best_val_loss": None,
        "best_step": None,
        "final_val_loss": None,
        "final_train_loss": None,
        "tokens_per_iter": None,
        "iter_time_ms_median": None,
        "tokens_per_sec_median": None,
        "mfu_median": None,
    }

    if not stdout_path.is_file():
        return out

    text = stdout_path.read_text(encoding="utf-8", errors="ignore")

    # tokens per iteration (may break line -> allow whitespace)
    m = RE_TOKENS.search(text)
    if m:
        tok_raw = re.sub(r"\s+", "", m.group(1))  # remove spaces/newlines
        tok_raw = re.match(r"[0-9,]+", tok_raw).group(0) if re.match(r"[0-9,]+", tok_raw) else ""
        if tok_raw:
            try:
                out["tokens_per_iter"] = int(tok_raw.replace(",", ""))
            except Exception:
                pass

    # step metrics
    steps: list[tuple[int, float, float]] = []
    for sm in RE_STEP.finditer(text):
        step = int(sm.group(1))
        tr = float(sm.group(2))
        va = float(sm.group(3))
        steps.append((step, tr, va))

    if steps:
        # final = last seen
        last_step, last_tr, last_va = steps[-1]
        out["final_val_loss"] = last_va
        out["final_train_loss"] = last_tr

        # best by val
        best = min(steps, key=lambda t: t[2])
        out["best_step"] = best[0]
        out["best_val_loss"] = best[2]

    # iter perf (exclude very long eval iters by cutoff)
    times: list[float] = []
    mfus: list[float] = []
    for im in RE_ITER.finditer(text):
        it = int(im.group(1))
        tms = float(im.group(2))
        mfu = float(im.group(3))

        # drop iter 0 (часто компиляция/прогрев) и отрицательный mfu
        if it == 0:
            continue

        # cut eval / outliers (eval в nanoGPT часто ~20s)
        if tms <= 0 or tms > 5000.0:
            continue

        times.append(tms)
        if mfu >= 0:
            mfus.append(mfu)

    if times:
        out["iter_time_ms_median"] = _median(times)
    if mfus:
        out["mfu_median"] = _median(mfus)

    if out["tokens_per_iter"] is not None and out["iter_time_ms_median"] is not None and out["iter_time_ms_median"] > 0:
        out["tokens_per_sec_median"] = out["tokens_per_iter"] / (out["iter_time_ms_median"] / 1000.0)

    return out


def summarize_run(folders:list[Path]) -> RunSummary:
    if not folders:
        return RunSummary(data={"key": "", "cnt": 0})
    
    real_cnt = 0
    
    # ---------- collect per-run info ----------
    per_best_val: list[float] = []
    per_final_val: list[float] = []
    per_best_step: list[float] = []
    per_gen_gap_final: list[float] = []

    per_tok_s: list[float] = []
    per_mfu: list[float] = []
    per_it_ms: list[float] = []

    metas: list[dict[str, Any]] = []
    
    run_infos: list[dict[str, Any]] = []
    
    for rd in folders:
        meta = read_meta(rd) or {}
        if meta:
            metas.append(meta)
            real_cnt += 1
        else:
            continue
            
        stdout_path = rd / "logs" / "stdout.log"
        m = parse_stdout_metrics(stdout_path)
        run_infos.append({"run_dir": rd, **m})
        
        if m["best_val_loss"] is not None:
            per_best_val.append(float(m["best_val_loss"]))
        if m["final_val_loss"] is not None:
            per_final_val.append(float(m["final_val_loss"]))
        if m["best_step"] is not None:
            per_best_step.append(float(m["best_step"]))

        if m["final_val_loss"] is not None and m["final_train_loss"] is not None:
            per_gen_gap_final.append(float(m["final_val_loss"]) - float(m["final_train_loss"]))

        if m["tokens_per_sec_median"] is not None:
            per_tok_s.append(float(m["tokens_per_sec_median"]))
        if m["mfu_median"] is not None:
            per_mfu.append(float(m["mfu_median"]))
        if m["iter_time_ms_median"] is not None:
            per_it_ms.append(float(m["iter_time_ms_median"]))
        
    print(f'{real_cnt}/{len(folders)}')    
    
    if not len(metas):
        return RunSummary(data={"key": "", "cnt": 0})
    
    first_meta = metas[0] if metas else {}
    exp_name = str(first_meta.get("exp_name") or "").strip()
    exp_desc = str(first_meta.get("exp_desc") or "").strip()
    git_commit = str(first_meta.get("git_commit") or "").strip()

    slurm = first_meta.get("slurm") if isinstance(first_meta.get("slurm"), dict) else {}
    array_job_id = str((slurm or {}).get("array_job_id") or (slurm or {}).get("job_id") or "").strip()
    
    key = f"{array_job_id}_{exp_name}".strip("_") if array_job_id and exp_name else (exp_name or folders[0].name)

    # repo_dir = str(first_meta.get("repo_dir") or "").strip()
    dataset_name = str(first_meta.get("dataset") or "").strip()
    # dataset_path = (Path(repo_dir) / "data" / dataset_name) if (repo_dir and dataset_name) else (Path(dataset_name) if dataset_name else None)

    config_path = str(first_meta.get("config") or "").strip()
    config_path = Path(config_path) if config_path else None

    # pick "representative" run_dir = run with minimal best_val_loss, else first
    best_run_dir = folders[0]
    if run_infos:
        with_best = [ri for ri in run_infos if ri.get("best_val_loss") is not None]
        if with_best:
            best_run_dir = min(with_best, key=lambda ri: ri["best_val_loss"])["run_dir"]

    # file paths from representative run
    git_diff_p = best_run_dir / "meta" / "git_diff.patch"
    git_diff_path = git_diff_p if git_diff_p.is_file() else None
    if git_diff_path:
        try:
            git_diff_path = git_diff_path if git_diff_path.stat().st_size > 0 else None
        except Exception:
            git_diff_path = None

    nvsmi_p = best_run_dir / "logs" / "nvidia-smi.txt"
    nvidia_smi_path = nvsmi_p if nvsmi_p.is_file() else None

    gpus_cnt, gpu_name = parse_gpu_from_nvidia_smi(nvsmi_p)
    
    
    # ---------- aggregate ----------
    best_val_mean = _mean(per_best_val)
    best_val_std = _std(per_best_val)
    best_val_min = min(per_best_val) if per_best_val else None
    best_val_max = max(per_best_val) if per_best_val else None

    final_val_mean = _mean(per_final_val)
    final_val_std = _std(per_final_val)

    best_ppl = [x for x in (_safe_exp(v) for v in per_best_val) if x is not None and math.isfinite(x)]
    final_ppl = [x for x in (_safe_exp(v) for v in per_final_val) if x is not None and math.isfinite(x)]

    best_ppl_mean = _mean(best_ppl)
    best_ppl_std = _std(best_ppl)
    final_ppl_mean = _mean(final_ppl)
    final_ppl_std = _std(final_ppl)

    best_step_median = _median(per_best_step)

    gen_gap_mean = _mean(per_gen_gap_final)
    gen_gap_std = _std(per_gen_gap_final)

    tok_mean = _mean(per_tok_s)
    tok_std = _std(per_tok_s)

    mfu_mean = _mean(per_mfu)
    mfu_std = _std(per_mfu)

    it_mean = _mean(per_it_ms)
    it_std = _std(per_it_ms)

    data: dict[str, Any] = {
        # ---- main info ----
        "key": key,
        "name": exp_name,
        "desc": exp_desc,
        "cnt": real_cnt,
        "run_dir": best_run_dir,
        "dataset": dataset_name,
        "config": config_path,
        "git_commit": git_commit,
        "git_diff_path": git_diff_path,

        # ---- technical ----
        "gpus": gpus_cnt,
        "gpu": gpu_name,
        "nvidia-smi": nvidia_smi_path,

        # ---- quality aggregated ----
        "best_val_loss_mean": best_val_mean,
        "best_val_loss_std": best_val_std,
        "best_val_loss_min": best_val_min,
        "best_val_loss_max": best_val_max,
        "final_val_loss_mean": final_val_mean,
        "final_val_loss_std": final_val_std,

        "best_val_ppl_mean": best_ppl_mean,
        "best_val_ppl_std": best_ppl_std,
        "final_val_ppl_mean": final_ppl_mean,
        "final_val_ppl_std": final_ppl_std,

        "best_step_median": best_step_median,
        "gen_gap_final_mean": gen_gap_mean,
        "gen_gap_final_std": gen_gap_std,

        # ---- performance aggregated ----
        "tokens_per_sec_mean": tok_mean,
        "tokens_per_sec_std": tok_std,
        "mfu_mean": mfu_mean,
        "mfu_std": mfu_std,
        "iter_time_ms_mean": it_mean,
        "iter_time_ms_std": it_std,
    }

    return RunSummary(data=data)


# =======================================================
# Significance filter (--mode filter)
# Uses only *_mean / *_std + n (cnt) to test deviations vs baseline.
# Mean: Welch t-test (two-sided)
# Backend: scipy.stats if available, else mpmath; otherwise error.
# =======================================================

def _need_stats_backend() -> None:
    if _stats is None and _mp is None:
        raise RuntimeError(
            "No scipy/mpmath available for --mode filter.\n"
            "Install one:\n"
            "  pip install scipy\n"
            "or\n"
            "  pip install mpmath"
        )

def _t_cdf(t: float, df: float) -> float:
    """Student-t CDF."""
    if _stats is not None:
        return float(_stats.t.cdf(t, df))
    _need_stats_backend()
    mp = _mp
    tt = mp.mpf(t)
    dff = mp.mpf(df)
    x = dff / (dff + tt * tt)
    ib = mp.betainc(dff / 2, mp.mpf("0.5"), 0, x, regularized=True)
    return float(1 - 0.5 * ib) if tt >= 0 else float(0.5 * ib)


def welch_t_pvalue(m1: float, s1: float, n1: int, m0: float, s0: float, n0: int) -> float:
    """Two-sided Welch t-test p-value using only summary stats."""
    if n1 < 2 or n0 < 2:
        return float("nan")
    if any(x is None for x in [m1, s1, m0, s0]):
        return float("nan")
    if any(isinstance(x, float) and math.isnan(x) for x in [m1, s1, m0, s0]):
        return float("nan")
    if s1 < 0 or s0 < 0:
        return float("nan")

    v1 = (s1 ** 2) / n1
    v0 = (s0 ** 2) / n0
    denom = math.sqrt(v1 + v0)
    if denom == 0:
        return 0.0 if m1 != m0 else 1.0

    t = (m1 - m0) / denom

    # Welch–Satterthwaite df
    num = (v1 + v0) ** 2
    den = 0.0
    if n1 > 1:
        den += (v1 ** 2) / (n1 - 1)
    if n0 > 1:
        den += (v0 ** 2) / (n0 - 1)
    df = (num / den) if den != 0 else (n1 + n0 - 2)

    cdf = _t_cdf(abs(t), df)
    p = 2.0 * (1.0 - cdf)
    return max(0.0, min(1.0, p))


def _bh_reject_mask(pvals: list[float], alpha: float) -> list[bool]:
    """
    Benjamini–Hochberg (FDR). NaN -> False.
    Returns boolean mask of rejections (significant).
    """
    m = len(pvals)
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None and not (isinstance(p, float) and math.isnan(p))]
    valid.sort(key=lambda x: x[1])
    mask = [False] * m
    if not valid:
        return mask

    kmax = -1
    M = len(valid)
    for rank, (i, p) in enumerate(valid, start=1):
        if p <= alpha * rank / M:
            kmax = rank

    if kmax < 0:
        return mask

    for rank, (i, p) in enumerate(valid, start=1):
        if rank <= kmax:
            mask[i] = True
    return mask

def _metric_groups(cols: list[str]) -> list[tuple[str, str, str]]:
    """
    Returns (base, mean_col, std_col) for columns like base_mean + base_std present.
    """
    means = [c for c in cols if c.endswith("_mean")]
    out: list[tuple[str, str, str]] = []
    for mc in means:
        base = mc[:-5]
        sc = base + "_std"
        if sc in cols:
            out.append((base, mc, sc))
    return out

# TODO пересмотреть 
def _get_n(rs: RunSummary, n_col: str, default_n: int = 5) -> int:
    v = rs.data.get(n_col, None)
    try:
        n = int(v)
        return n if n >= 2 else default_n
    except Exception:
        return default_n

def _find_baseline_key(summary: Summary, baseline_value: str, baseline_col: str) -> str:
    """
    Returns the key in summary.items corresponding to baseline_value in baseline_col.
    Special-case: baseline_col == 'key' means dictionary key.
    """
    if baseline_col == "key" and baseline_value in summary.items:
        return baseline_value
    for k, rs in summary.items.items():
        if str(rs.data.get(baseline_col, "")).strip() == str(baseline_value).strip():
            return k
    raise ValueError(f"Baseline '{baseline_value}' not found in column '{baseline_col}'")


def filter_summary_significant(
    summary: "Summary",
    baseline_value: str,
    baseline_col: str = "key",
    n_col: str = "cnt",
    alpha: float = 0.05,
    correction: str = "none",  # "none" | "bh"
) -> "Summary":
    """
    Builds new Summary:
      - baseline row kept intact
      - for each other row: metric cells (*_mean / *_std) are kept only if
        mean difference vs baseline is significant by Welch t-test on *_mean.
      - non-metric columns kept intact
    """
    base_key = _find_baseline_key(summary, baseline_value, baseline_col)
    base = summary.items[base_key]

    # columns universe (use saved headers + any columns appearing in data)
    cols = list(summary.headers)
    for rs in summary.items.values():
        for c in rs.data.keys():
            if c not in cols:
                cols.append(c)

    groups = _metric_groups(cols)
    metric_cols = set()
    for _b, mc, sc in groups:
        metric_cols.add(mc)
        metric_cols.add(sc)

    out = Summary()
    out.headers = cols
    out.items = {}

    # baseline as-is
    out.add(RunSummary(data=dict(base.data)))

    n0 = _get_n(base, n_col)

    for k, rs in summary.items.items():
        if k == base_key:
            continue

        n1 = _get_n(rs, n_col)
        rs_out = RunSummary(data=dict(rs.data))

        # blank all metric cells by default
        for c in metric_cols:
            if c in rs_out.data:
                rs_out.data[c] = None

        pvals: list[float] = []
        idx_map: list[tuple[str, str, str]] = []

        for base_name, mc, sc in groups:
            m1 = rs.data.get(mc, None)
            s1 = rs.data.get(sc, None)
            m0 = base.data.get(mc, None)
            s0 = base.data.get(sc, None)

            if m1 is None or s1 is None or m0 is None or s0 is None:
                p = float("nan")
            else:
                p = welch_t_pvalue(float(m1), float(s1), n1, float(m0), float(s0), n0)

            pvals.append(p)
            idx_map.append((base_name, mc, sc))

        if correction == "bh":
            sig_mask = _bh_reject_mask(pvals, alpha)
        else:
            sig_mask = [(p is not None and not (isinstance(p, float) and math.isnan(p)) and p < alpha) for p in pvals]

        for i, (base_name, mc, sc) in enumerate(idx_map):
            if sig_mask[i]:
                rs_out.data[mc] = rs.data.get(mc, None)
                rs_out.data[sc] = rs.data.get(sc, None)
                
                
        
        out.add(rs_out)

    def _is_col_needed(key: str) -> bool:
        _key = str(key);
        needed = [
            "key",
            "name",
            baseline_col,
            n_col,
            "desc",
            "dataset",
        ]
        return _key in needed or _key.endswith("_mean") or _key.endswith("_std")

    res_cols: list[str] = [c for c in cols if _is_col_needed(c)]

    out.headers = res_cols
    for _k, rs in out.items.items():
        rs.data = {c: rs.data.get(c, None) for c in res_cols}

    return out


# =======================================================

def main() -> int:
    check_arguments()
    
    if ARGS.mode == "summarize":
        print(f'RUNS_ROOT: {RUNS_ROOT}')
        print(f'SUMMARY:   {SUMMARY}')
        
        summary = Summary()
        summary.load_from_path(SUMMARY)
        
        runs = collect_runs() 
        
        for key, run in runs:
            print(key, end=" ")
            if not summary.check_presents(key) or summary.items[key].data.get("cnt", 0) != len(run):
                summary.add(summarize_run(run))
            else:
                print("skipped")
        
        summary.save_to(SUMMARY)
        
    elif ARGS.mode == "filter":
        if not ARGS.baseline:
            raise Exception("--baseline is required in --mode filter")

        summary = Summary()
        summary.load_from_path(SUMMARY)

        out_path = ARGS.out.strip()
        if not out_path:
            out_path = str(Path(SUMMARY).with_name(f"{Path(SUMMARY).name[:-4]}_significant.csv"))
        
        sig = filter_summary_significant(
            summary=summary,
            baseline_value=ARGS.baseline,
            baseline_col=ARGS.baseline_col,
            n_col=ARGS.n_col,
            alpha=float(ARGS.alpha),
            correction=str(ARGS.correction),
        )
        sig.save_to(out_path)

        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())