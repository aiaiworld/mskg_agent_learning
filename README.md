# MSKG Agent Learning

This repository contains the reproducibility artifact for:

**MSKG: Morphology--Semantic--Knowledge Graph Learning for Lifelong Agentic 5G Core Operations**

The artifact is intentionally lightweight. The main prototype is dependency-free Python and evaluates a morphology-semantic-knowledge-graph agent on synthetic 5G Core RCA scenarios plus a small public-log OOD/distractor check.

## Contents

```text
src/mskg_agent_learning_demo.py        Main MSKG RCA prototype, CLI, HTTP demo, and synthetic benchmark
benchmarks/open_log_ood_benchmark.py   Public-log OOD/distractor benchmark
results/                              Saved benchmark outputs used in the paper
paper/                                LaTeX source and compiled PDF
k8s/                                  Optional Kubernetes deployment example
scripts/                              Optional deployment helper
```

## Requirements

- Python 3.10 or newer.
- No external Python packages are required for the core artifact.

## Reproduce the Main Benchmark

```bash
python src/mskg_agent_learning_demo.py --self-test
python src/mskg_agent_learning_demo.py --benchmark --iterations 100 --json > results/benchmark_mskg_learning_latest_reproduced.json
```

Expected pattern:

- Semantic-only and Vector-RAG underperform morphology-aware scoring.
- Full MSKG reaches perfect Top-1 on the 10-case base split.
- The stress split reported in the paper remains the harder evidence, with non-saturated Top-1.

## Reproduce the Public-Log OOD/Distractor Benchmark

```bash
python benchmarks/open_log_ood_benchmark.py > results/open_log_ood_results_reproduced.json
```

The public-log rows are infrastructure log examples, not 5G Core incidents. They are used for:

- OOD abstention: standalone non-5GC logs should not be forced into a 5GC RCA label.
- Distractor robustness: open infrastructure log strings are injected into 5GC RCA tickets.

## Run the Browser Demo Locally

```bash
python src/mskg_agent_learning_demo.py --serve --host 0.0.0.0 --port 8090
```

Then open:

```text
http://localhost:8090/
```

## Docker

```bash
docker build -t mskg-agent-learning:repro .
docker run --rm -p 8090:8090 mskg-agent-learning:repro
```

## Kubernetes

The files under `k8s/` and `scripts/` are optional examples. They use a public Python base image and do not depend on a private registry.

```bash
bash scripts/deploy_mskg_demo.sh
```

