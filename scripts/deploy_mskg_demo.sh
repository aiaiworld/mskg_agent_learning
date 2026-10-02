#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"

cd "$(dirname "$0")/.."

python3 -m py_compile src/mskg_agent_learning_demo.py
python3 src/mskg_agent_learning_demo.py --self-test

kubectl -n "$NAMESPACE" create configmap mskg-agent-learning-code \
  --from-file=mskg_agent_learning_demo.py=src/mskg_agent_learning_demo.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NAMESPACE" apply -f k8s/mskg-agent-learning.yaml
kubectl -n "$NAMESPACE" rollout status deployment/mskg-agent-learning --timeout=180s
kubectl -n "$NAMESPACE" get deploy,svc,pod -l app=mskg-agent-learning -o wide

