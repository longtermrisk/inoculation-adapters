"""Local experiment driver: build data → run GPU pipeline on a RunPod pod → score.

Run with the arsenal venv (needs stagehand + bellhop):
    ~/jarvis/repos/arsenal/.venv/bin/python experiments/leaky_backdoor/run_experiment.py [--smoke] [--gpu A100]

Local library steps run in this repo's own venv via `uv run`.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_env() -> None:
    envf = Path.home() / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _uv_run(script: str, *args: str) -> None:
    subprocess.run(["uv", "run", "python", script, *args], cwd=REPO, check=True)


async def build_data(smoke: bool) -> str:
    _uv_run("experiments/leaky_backdoor/build_data.py", *(["--smoke"] if smoke else []))
    return "data-built"


async def pod_run(_prev: str, smoke: bool, gpu: str) -> str:
    from bellhop import PodConfig, pod  # arsenal venv

    cfg = PodConfig(
        gpu=gpu,
        image_preset="pytorch-cuda",
        max_lifetime=timedelta(hours=1.5 if smoke else 6),
    )
    env = {}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    async with pod(cfg) as p:
        await p.push(str(REPO), "/workspace/job")
        r = await p.exec(
            "cd /workspace/job && pip install -q -r infra/pod-requirements.txt "
            f"&& python experiments/leaky_backdoor/pod_pipeline.py --data experiments/leaky_backdoor/data --out experiments/leaky_backdoor/out{' --smoke' if smoke else ''}",
            env=env,
        )
        print(r.stdout[-4000:])
        if r.stderr:
            print(r.stderr[-2000:], file=sys.stderr)
        if "pipeline COMPLETE" not in r.stdout:
            raise RuntimeError(f"pod pipeline did not complete (exit {r.returncode})")
        await p.pull("/workspace/job/experiments/leaky_backdoor/out", str(REPO / "experiments" / "leaky_backdoor"))
    return "pod-done"


async def score(_prev: str) -> str:
    _uv_run("experiments/leaky_backdoor/score_results.py")
    return "scored"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--skip-data", action="store_true", help="reuse existing data/")
    args = parser.parse_args()
    load_env()
    if not os.environ.get("RUNPOD_API_KEY"):
        raise SystemExit("RUNPOD_API_KEY not set (checked env and ~/.env)")

    from stagehand import Flow, live_dashboard

    flow = Flow(str(REPO / "runs"), concurrency=1)
    async def reuse_data() -> str:
        return "data-reused"

    if args.skip_data:
        d = flow.spawn(reuse_data, name="build_data")
    else:
        d = flow.spawn(build_data, args=[args.smoke], name="build_data")
    pr = flow.spawn(pod_run, args=[d, args.smoke, args.gpu], name="pod_run")
    flow.spawn(score, args=[pr], name="score")

    async with live_dashboard(flow.runs_dir, title="inoc"):
        state = await flow.run()
    if state.failed:
        raise SystemExit(f"{state.failed} step(s) failed")
    print("EXPERIMENT COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
