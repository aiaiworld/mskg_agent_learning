"""External OOD stress benchmark for the MSKG paper.

The rows below are public examples exposed by the Hugging Face dataset
viewer/search snippets for reachjalil/jevlogs-log-triage-benchmark and
logfit-project/BGL/HDFS. They are not 5G Core incidents. The benchmark tests
whether a 5GC RCA agent abstains instead of forcing a telecom root cause on
open infrastructure logs.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mskg_agent_learning_demo import EVAL_SCENARIOS, MSKGLearningAgent, evaluate, otel_for_cause  # noqa: E402


PUBLIC_LOG_ROWS = [
    {
        "id": "bgl_normal_icache_1",
        "dataset": "logfit-project/BGL",
        "label": "normal",
        "content": "KERNEL INFO instruction cache parity error corrected",
    },
    {
        "id": "bgl_normal_core_1",
        "dataset": "logfit-project/BGL",
        "label": "normal",
        "content": "KERNEL INFO generating core 33707",
    },
    {
        "id": "bgl_normal_core_2",
        "dataset": "logfit-project/BGL",
        "label": "normal",
        "content": "KERNEL INFO generating core 14562",
    },
    {
        "id": "bgl_normal_core_3",
        "dataset": "logfit-project/BGL",
        "label": "normal",
        "content": "KERNEL INFO generating core 10170",
    },
    {
        "id": "bgl_anom_socket_1",
        "dataset": "logfit-project/BGL",
        "label": "anomaly",
        "content": "APP ciod FATAL Error reading message prefix after LOAD_MESSAGE on CioStream socket to IP Link has been severed",
    },
    {
        "id": "bgl_anom_socket_2",
        "dataset": "logfit-project/BGL",
        "label": "anomaly",
        "content": "APP ciod FATAL Error reading message prefix on CioStream socket to IP Link has been severed",
    },
    {
        "id": "bgl_anom_node_map",
        "dataset": "logfit-project/BGL",
        "label": "anomaly",
        "content": "APP ciod FATAL Error creating node map from file PATH Resource temporarily unavailable",
    },
    {
        "id": "bgl_anom_storage_interrupt",
        "dataset": "logfit-project/BGL",
        "label": "anomaly",
        "content": "KERNEL FATAL data storage interrupt",
    },
    {
        "id": "bgl_anom_tlb_interrupt_1",
        "dataset": "logfit-project/BGL",
        "label": "anomaly",
        "content": "KERNEL FATAL data TLB error interrupt",
    },
    {
        "id": "bgl_anom_tlb_interrupt_2",
        "dataset": "logfit-project/BGL",
        "label": "anomaly",
        "content": "KERNEL FATAL data TLB error interrupt repeated on compute node",
    },
    {
        "id": "hdfs_normal_receive_block",
        "dataset": "logfit-project/HDFS_v1",
        "label": "normal",
        "content": "dfs.DataNode DataXceiver INFO Receiving block BLOCK src IP dest IP",
    },
    {
        "id": "hdfs_normal_pipeline",
        "dataset": "logfit-project/HDFS_v1",
        "label": "normal",
        "content": "dfs.DataNode INFO PacketResponder for block terminating",
    },
]


def evaluate_threshold(threshold):
    agent = MSKGLearningAgent()
    examples = []
    accepted = 0
    max_scores = []
    for row in PUBLIC_LOG_ROWS:
        ranked = agent.rank(row["content"], facts=[], mode="mskg_learning")
        top = ranked[0]
        is_accepted = top["score"] >= threshold
        accepted += int(is_accepted)
        max_scores.append(top["score"])
        examples.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "source_label": row["label"],
                "top_cause": top["cause_id"],
                "top_score": top["score"],
                "decision": "accept_5gc_rca" if is_accepted else "abstain_ood",
            }
        )
    total = len(PUBLIC_LOG_ROWS)
    return {
        "threshold": threshold,
        "samples": total,
        "ood_abstention_rate": round((total - accepted) / total, 4),
        "false_5gc_acceptance_rate": round(accepted / total, 4),
        "mean_top_score": round(sum(max_scores) / total, 6),
        "max_top_score": round(max(max_scores), 6),
        "examples": examples,
    }


def evaluate_open_augmented_incidents():
    """Inject public infrastructure logs as distractor telemetry into 5GC RCA."""
    augmented = []
    for scenario in EVAL_SCENARIOS:
        for row in PUBLIC_LOG_ROWS:
            augmented.append(
                {
                    "id": "%s_%s" % (scenario["id"], row["id"]),
                    "query": "%s External infrastructure log: %s" % (scenario["query"], row["content"]),
                    "facts": list(scenario.get("facts", [])),
                    "otel": otel_for_cause(scenario["gold"]),
                    "gold": scenario["gold"],
                }
            )
    agent = MSKGLearningAgent()
    return {
        "samples": len(augmented),
        "results": [
            evaluate(agent, augmented, "semantic_only"),
            evaluate(agent, augmented, "vector_rag"),
            evaluate(agent, augmented, "graph_rag"),
            evaluate(agent, augmented, "morphology_semantic"),
            evaluate(agent, augmented, "mskg_learning"),
        ],
    }


def main():
    result = {
        "benchmark": "public-log-ood-stress",
        "source": "Public rows shown by Hugging Face dataset viewer/search snippets for reachjalil/jevlogs-log-triage-benchmark, logfit-project/BGL, and logfit-project/HDFS_v1.",
        "note": "Rows are open infrastructure logs, not 5G Core incidents. Expected behavior is abstention/OOD rather than forced 5GC RCA.",
        "ood_results": [evaluate_threshold(t) for t in (0.10, 0.15, 0.20, 0.25)],
        "open_log_augmented_5gc": evaluate_open_augmented_incidents(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
