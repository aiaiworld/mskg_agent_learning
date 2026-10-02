# Public-Log OOD/Distractor Benchmark Summary

Timestamp: 20260930-112613

Source scope: 12 public infrastructure log examples from BGL/HDFS previews derived from LogHub-style datasets. These are not 5G Core incidents, so they are used only for OOD abstention and distractor robustness checks.

## Standalone OOD Check

| Threshold | Samples | OOD abstention | False 5GC accept | Mean top score | Max top score |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 12 | 1.000 | 0.000 | 0.066667 | 0.066667 |
| 0.15 | 12 | 1.000 | 0.000 | 0.066667 | 0.066667 |
| 0.20 | 12 | 1.000 | 0.000 | 0.066667 | 0.066667 |
| 0.25 | 12 | 1.000 | 0.000 | 0.066667 | 0.066667 |

## 5GC Incidents With Open-Log Distractors

| Method | Samples | Top-1 | Top-3 | MRR | Mean rank |
|---|---:|---:|---:|---:|---:|
| Semantic-only | 120 | 0.600 | 0.800 | 0.7092 | 2.5750 |
| Vector-RAG | 120 | 0.525 | 0.750 | 0.6672 | 2.8667 |
| GraphRAG | 120 | 0.7583 | 0.975 | 0.8531 | 1.4167 |
| Morphology+semantic | 120 | 0.775 | 1.000 | 0.8708 | 1.3250 |
| MSKG-learned | 120 | 1.000 | 1.000 | 1.0000 | 1.0000 |

