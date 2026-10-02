"""Morphology-Semantic-Knowledge Graph learning demo for 5G Core agents.

This demo models agent learning as three coupled representations:

1. Morphology: the operational shape of the environment, including NF topology,
   interfaces, tool affordances, and action constraints.
2. Semantics: normalized concepts extracted from operator text and observations.
3. Knowledge Graph: typed facts and causal/evidence links that support
   multi-hop reasoning and online updates.

It is intentionally dependency-free so it can run in the Lab 5G cluster from a
plain Python image.
"""

import argparse
import json
import math
import re
import time
import unicodedata
from collections import defaultdict, deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlsplit


PORT = 8090
MAX_BODY_BYTES = 1024 * 1024


ALIASES = {
    "pdu": "pdu_session",
    "session": "pdu_session",
    "pfcp": "pfcp",
    "n4": "n4",
    "smf": "smf",
    "upf": "upf",
    "gtpu": "gtpu",
    "gtp": "gtpu",
    "n3": "n3",
    "n6": "n6",
    "drop": "drop",
    "loss": "drop",
    "timeout": "timeout",
    "timeouts": "timeout",
    "degraded": "degraded",
    "fail": "failed",
    "failed": "failed",
    "failure": "failed",
    "reject": "rejected",
    "rejected": "rejected",
    "crash": "crashloop",
    "crashloop": "crashloop",
    "restart": "restart",
    "pod": "pod",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "nrf": "nrf",
    "sba": "sba",
    "discovery": "discovery",
    "register": "registration",
    "registration": "registration",
    "auth": "authentication",
    "authentication": "authentication",
    "ausf": "ausf",
    "udm": "udm",
    "udr": "udr",
    "db": "database",
    "database": "database",
    "subscriber": "subscriber",
    "subscription": "subscriber",
    "profile": "subscriber",
    "pcf": "pcf",
    "policy": "policy",
    "dnn": "dnn",
    "qos": "qos",
    "slice": "slice",
    "slices": "slice",
    "tac": "tac",
    "nssf": "nssf",
    "sst": "sst",
    "sd": "sd",
    "amf": "amf",
    "n2": "n2",
    "sctp": "sctp",
    "ngap": "ngap",
    "gnb": "gnb",
    "qwen": "qwen",
    "llm": "llm",
    "model": "llm",
    "endpoint": "endpoint",
    "route": "route",
    "routing": "route",
    "data": "user_plane",
    "plane": "user_plane",
    "packet": "drop",
    "goi": "drop",
    "mat": "drop",
    "thap": "throughput",
    "latency": "latency",
    "slow": "latency",
    "healthy": "healthy",
}


GENERIC_CONCEPTS = set(
    [
        "pdu_session",
        "timeout",
        "failed",
        "degraded",
        "healthy",
        "rejected",
        "latency",
        "route",
        "nf",
    ]
)


CAUSES = {
    "n4_pfcp_path": {
        "label": "N4/PFCP path SMF-UPF",
        "tags": ["pdu_session", "pfcp", "n4", "smf", "upf", "timeout", "degraded"],
        "nodes": ["smf", "upf", "n4", "pfcp"],
        "tools": ["pfcp_trace", "smf_logs", "upf_logs", "kpi_window"],
        "actions": ["check_n4_reachability", "validate_upf_route", "restart_pfcp_association"],
    },
    "pcf_policy_mismatch": {
        "label": "PCF/DNN policy mismatch",
        "tags": ["pcf", "policy", "dnn", "qos", "slice", "rejected", "pdu_session"],
        "nodes": ["pcf", "smf", "dnn", "slice"],
        "tools": ["query_pcf_policy", "smf_logs", "subscriber_profile"],
        "actions": ["compare_policy_snapshot", "validate_dnn_slice_rule"],
    },
    "nrf_registration": {
        "label": "NRF registration or discovery fault",
        "tags": ["nrf", "sba", "discovery", "registration", "nf", "failed"],
        "nodes": ["nrf", "sba", "amf", "smf", "udm", "pcf"],
        "tools": ["nrf_registry", "service_health", "sba_trace"],
        "actions": ["refresh_nf_registration", "check_sba_dns"],
    },
    "ausf_udm_auth": {
        "label": "AUSF/UDM authentication data fault",
        "tags": ["ausf", "udm", "authentication", "subscriber", "supi", "suci", "failed"],
        "nodes": ["ausf", "udm", "udr", "subscriber"],
        "tools": ["auth_trace", "subscriber_profile", "udm_logs"],
        "actions": ["validate_auth_vector", "check_subscriber_profile"],
    },
    "upf_user_plane_drop": {
        "label": "UPF N3/N6 user-plane packet drop",
        "tags": ["upf", "gtpu", "n3", "n6", "drop", "latency", "throughput", "user_plane"],
        "nodes": ["upf", "n3", "n6", "gtpu"],
        "tools": ["upf_counters", "gtpu_trace", "node_network"],
        "actions": ["check_gtpu_drop", "check_n6_route"],
    },
    "llm_endpoint_route": {
        "label": "LLM endpoint route or model serving fault",
        "tags": ["qwen", "llm", "endpoint", "route", "timeout", "model"],
        "nodes": ["qwen", "llm_endpoint", "l40", "vdi"],
        "tools": ["curl_llm_health", "route_trace", "model_list"],
        "actions": ["select_alternate_endpoint", "patch_endpoint_priority"],
    },
    "smf_pod_runtime": {
        "label": "SMF pod runtime fault",
        "tags": ["smf", "pod", "kubernetes", "crashloop", "restart", "runtime"],
        "nodes": ["smf", "smf_pod", "kubernetes"],
        "tools": ["kubectl_get_pods", "kubectl_logs", "kubectl_events"],
        "actions": ["inspect_crashloop", "rollback_smf_config"],
    },
    "amf_n2_sctp": {
        "label": "AMF N2/SCTP or NGAP association fault",
        "tags": ["amf", "n2", "sctp", "ngap", "gnb", "reset", "failed"],
        "nodes": ["amf", "n2", "sctp", "gnb"],
        "tools": ["amf_logs", "sctp_trace", "gnb_status"],
        "actions": ["check_sctp_association", "validate_tac_plmn"],
    },
    "nssf_slice_selection": {
        "label": "NSSF slice selection mismatch",
        "tags": ["nssf", "slice", "tac", "sst", "sd", "selection", "rejected"],
        "nodes": ["nssf", "slice", "tac", "amf"],
        "tools": ["nssf_rules", "amf_logs", "slice_catalog"],
        "actions": ["compare_slice_catalog", "validate_tac_mapping"],
    },
    "udr_database_latency": {
        "label": "UDR database latency or stale subscriber state",
        "tags": ["udr", "database", "udm", "subscriber", "latency", "profile"],
        "nodes": ["udr", "database", "udm", "subscriber"],
        "tools": ["db_latency", "udr_logs", "subscriber_profile"],
        "actions": ["check_db_latency", "verify_subscription_cache"],
    },
}


TAG_FREQUENCY = defaultdict(int)
for _cause_id, _spec in CAUSES.items():
    for _tag in set(_spec["tags"] + _spec["nodes"]):
        TAG_FREQUENCY[_tag] += 1


TRAIN_SCENARIOS = [
    {
        "id": "train_n4_1",
        "query": "UE tao PDU session bi timeout, SMF bao PFCP retransmission tang tren N4.",
        "facts": [{"entity": "smf", "attribute": "n4_state", "value": "degraded", "confidence": 0.94}],
        "gold": "n4_pfcp_path",
    },
    {
        "id": "train_policy_1",
        "query": "PDU session bi reject khi dung DNN ims va slice enterprise.",
        "facts": [{"entity": "pcf", "attribute": "dnn_policy", "value": "mismatch", "confidence": 0.92}],
        "gold": "pcf_policy_mismatch",
    },
    {
        "id": "train_nrf_1",
        "query": "SMF khong discover duoc UDM qua SBA, nhieu NF mat registration tren NRF.",
        "facts": [{"entity": "nrf", "attribute": "registry_state", "value": "stale", "confidence": 0.9}],
        "gold": "nrf_registration",
    },
    {
        "id": "train_auth_1",
        "query": "UE authentication fail, auth vector khong khop subscriber profile.",
        "facts": [{"entity": "udm", "attribute": "auth_vector", "value": "invalid", "confidence": 0.91}],
        "gold": "ausf_udm_auth",
    },
    {
        "id": "train_upf_1",
        "query": "Throughput giam manh, GTPU drop cao o UPF tren duong N3/N6.",
        "facts": [{"entity": "upf", "attribute": "gtpu_drop", "value": "high", "confidence": 0.93}],
        "gold": "upf_user_plane_drop",
    },
    {
        "id": "train_llm_1",
        "query": "Agent goi Qwen timeout, model endpoint tu L40 khong tra /v1/models.",
        "facts": [{"entity": "llm_endpoint", "attribute": "route", "value": "timeout", "confidence": 0.89}],
        "gold": "llm_endpoint_route",
    },
    {
        "id": "train_smf_pod_1",
        "query": "SMF pod crashloop sau khi thay config, kubernetes events bao restart lien tuc.",
        "facts": [{"entity": "smf_pod", "attribute": "phase", "value": "CrashLoopBackOff", "confidence": 0.96}],
        "gold": "smf_pod_runtime",
    },
    {
        "id": "train_amf_1",
        "query": "gNB mat ket noi, AMF ghi N2 SCTP reset va NGAP setup fail.",
        "facts": [{"entity": "amf", "attribute": "n2_state", "value": "sctp_reset", "confidence": 0.95}],
        "gold": "amf_n2_sctp",
    },
    {
        "id": "train_nssf_1",
        "query": "UE trong TAC moi khong chon duoc slice, NSSF reject SST SD.",
        "facts": [{"entity": "nssf", "attribute": "slice_rule", "value": "missing_tac", "confidence": 0.9}],
        "gold": "nssf_slice_selection",
    },
    {
        "id": "train_udr_1",
        "query": "UDM doc subscriber profile rat cham, UDR database latency tang.",
        "facts": [{"entity": "udr", "attribute": "db_latency", "value": "high", "confidence": 0.93}],
        "gold": "udr_database_latency",
    },
]


EVAL_SCENARIOS = [
    {
        "id": "eval_n4_ambiguous",
        "query": "Khach hang tao session khong xong sau migration user-plane; service khong crash.",
        "facts": [{"entity": "smf", "attribute": "n4_state", "value": "degraded", "confidence": 0.93}],
        "gold": "n4_pfcp_path",
    },
    {
        "id": "eval_policy_sparse",
        "query": "Goi data moi khong duoc cap cho thue bao enterprise, log chi bao reject chung.",
        "facts": [{"entity": "pcf", "attribute": "policy_delta", "value": "new_slice_rule", "confidence": 0.88}],
        "gold": "pcf_policy_mismatch",
    },
    {
        "id": "eval_nrf_noisy",
        "query": "Nhieu service trong core khong tim thay nhau sau maintenance window.",
        "facts": [{"entity": "nrf", "attribute": "registration_count", "value": "dropped", "confidence": 0.86}],
        "gold": "nrf_registration",
    },
    {
        "id": "eval_auth",
        "query": "Mot nhom thue bao khong vao mang, chua den buoc tao session.",
        "facts": [{"entity": "ausf", "attribute": "auth_status", "value": "failed", "confidence": 0.9}],
        "gold": "ausf_udm_auth",
    },
    {
        "id": "eval_upf",
        "query": "UE da co session nhung dich vu data rat kem va mat goi.",
        "facts": [{"entity": "upf", "attribute": "n6_drop_rate", "value": "high", "confidence": 0.92}],
        "gold": "upf_user_plane_drop",
    },
    {
        "id": "eval_llm",
        "query": "Planner chay binh thuong nhung hoi model AI thi treo khi kiem tra service.",
        "facts": [{"entity": "llm_endpoint", "attribute": "health", "value": "timeout", "confidence": 0.9}],
        "gold": "llm_endpoint_route",
    },
    {
        "id": "eval_smf_pod",
        "query": "Sau rollout, service xu ly session restart lien tuc, cac service khac van ok.",
        "facts": [{"entity": "smf_pod", "attribute": "restarts", "value": "high", "confidence": 0.94}],
        "gold": "smf_pod_runtime",
    },
    {
        "id": "eval_amf",
        "query": "RAN attach luc duoc luc khong, association voi core reset theo chu ky.",
        "facts": [{"entity": "amf", "attribute": "sctp_assoc", "value": "flapping", "confidence": 0.91}],
        "gold": "amf_n2_sctp",
    },
    {
        "id": "eval_nssf",
        "query": "Thue bao o khu vuc moi khong lay dung dich vu theo slice mong muon.",
        "facts": [{"entity": "nssf", "attribute": "tac_mapping", "value": "missing", "confidence": 0.88}],
        "gold": "nssf_slice_selection",
    },
    {
        "id": "eval_udr",
        "query": "Tra cuu ho so thue bao cham bat thuong, cac NF khac healthy.",
        "facts": [{"entity": "udr", "attribute": "read_latency_ms", "value": "1800", "confidence": 0.89}],
        "gold": "udr_database_latency",
    },
]


OTEL_SIGNAL_TEMPLATES = {
    "n4_pfcp_path": {
        "trace_id": "tr-n4-pfcp",
        "spans": [
            {
                "service": "smf",
                "name": "pfcp.session.establish",
                "status": "error",
                "attributes": {"interface": "n4", "peer": "upf", "protocol": "pfcp"},
            },
            {
                "service": "upf",
                "name": "pfcp.association.heartbeat",
                "status": "degraded",
                "attributes": {"interface": "n4", "retransmission": "high"},
            },
        ],
        "metrics": [
            {"service": "smf", "name": "pfcp_retransmission_rate", "value": 5.4, "unit": "%"},
            {"service": "smf", "name": "pdu_session_success_rate", "value": 94.2, "unit": "%"},
        ],
        "logs": [
            {
                "service": "smf",
                "level": "WARN",
                "message": "PFCP Session Establishment Request timeout on N4 peer upf",
            }
        ],
    },
    "pcf_policy_mismatch": {
        "trace_id": "tr-pcf-policy",
        "spans": [
            {
                "service": "smf",
                "name": "pdu.session.policy.authorization",
                "status": "error",
                "attributes": {"pcf": "pcf", "dnn": "enterprise", "slice": "1-010203"},
            }
        ],
        "metrics": [{"service": "pcf", "name": "policy_reject_rate", "value": 7.9, "unit": "%"}],
        "logs": [
            {"service": "pcf", "level": "WARN", "message": "DNN policy mismatch for enterprise slice qos rule"},
            {"service": "smf", "level": "WARN", "message": "PDU session rejected by PCF policy decision"},
        ],
    },
    "nrf_registration": {
        "trace_id": "tr-nrf-reg",
        "spans": [
            {
                "service": "smf",
                "name": "sba.discovery.nrf",
                "status": "error",
                "attributes": {"target": "nrf", "operation": "discover"},
            },
            {
                "service": "nrf",
                "name": "nf.registration.lookup",
                "status": "degraded",
                "attributes": {"registry_state": "stale"},
            },
        ],
        "metrics": [{"service": "nrf", "name": "nf_registration_count", "value": 43, "unit": "count"}],
        "logs": [{"service": "nrf", "level": "ERROR", "message": "NF discovery failed because registry is stale"}],
    },
    "ausf_udm_auth": {
        "trace_id": "tr-auth-udm",
        "spans": [
            {
                "service": "ausf",
                "name": "ue.authentication",
                "status": "error",
                "attributes": {"udm": "udm", "subscriber": "supi"},
            },
            {
                "service": "udm",
                "name": "auth.vector.lookup",
                "status": "error",
                "attributes": {"udr": "udr", "profile": "subscriber"},
            },
        ],
        "metrics": [{"service": "ausf", "name": "authentication_failure_rate", "value": 8.8, "unit": "%"}],
        "logs": [{"service": "udm", "level": "ERROR", "message": "Authentication vector invalid for subscriber profile"}],
    },
    "upf_user_plane_drop": {
        "trace_id": "tr-upf-drop",
        "spans": [
            {
                "service": "upf",
                "name": "gtpu.forward.n3",
                "status": "degraded",
                "attributes": {"interface": "n3", "protocol": "gtpu", "drop": "high"},
            },
            {
                "service": "upf",
                "name": "n6.egress",
                "status": "degraded",
                "attributes": {"interface": "n6", "throughput": "low"},
            },
        ],
        "metrics": [
            {"service": "upf", "name": "gtpu_drop_rate", "value": 3.1, "unit": "%"},
            {"service": "upf", "name": "n6_egress_throughput", "value": 1.8, "unit": "Gbps"},
        ],
        "logs": [{"service": "upf", "level": "WARN", "message": "No matching PDR for uplink GTPU TEID on N3"}],
    },
    "llm_endpoint_route": {
        "trace_id": "tr-llm-route",
        "spans": [
            {
                "service": "planner-agent",
                "name": "llm.chat.completions",
                "status": "error",
                "attributes": {"endpoint": "qwen", "route": "l40"},
            },
            {
                "service": "llm_endpoint",
                "name": "model.list",
                "status": "timeout",
                "attributes": {"model": "qwen", "path": "/v1/models"},
            },
        ],
        "metrics": [{"service": "llm_endpoint", "name": "request_timeout_rate", "value": 19.0, "unit": "%"}],
        "logs": [{"service": "planner-agent", "level": "ERROR", "message": "Qwen model endpoint route timeout"}],
    },
    "smf_pod_runtime": {
        "trace_id": "tr-smf-pod",
        "spans": [
            {
                "service": "kubernetes",
                "name": "pod.restart",
                "status": "error",
                "attributes": {"pod": "smf_pod", "component": "smf"},
            }
        ],
        "metrics": [{"service": "smf_pod", "name": "container_restart_count", "value": 7, "unit": "count"}],
        "logs": [{"service": "smf_pod", "level": "ERROR", "message": "CrashLoopBackOff after SMF config rollout"}],
    },
    "amf_n2_sctp": {
        "trace_id": "tr-amf-n2",
        "spans": [
            {
                "service": "amf",
                "name": "ngap.setup",
                "status": "error",
                "attributes": {"interface": "n2", "protocol": "sctp", "gnb": "gnb"},
            }
        ],
        "metrics": [{"service": "amf", "name": "sctp_reset_count", "value": 18, "unit": "count"}],
        "logs": [{"service": "amf", "level": "WARN", "message": "N2 SCTP association reset and NGAP setup failed"}],
    },
    "nssf_slice_selection": {
        "trace_id": "tr-nssf-slice",
        "spans": [
            {
                "service": "nssf",
                "name": "slice.selection",
                "status": "error",
                "attributes": {"slice": "sst_sd", "tac": "missing"},
            }
        ],
        "metrics": [{"service": "nssf", "name": "slice_selection_reject_rate", "value": 6.7, "unit": "%"}],
        "logs": [{"service": "nssf", "level": "WARN", "message": "Slice selection rejected due to TAC SST SD mapping"}],
    },
    "udr_database_latency": {
        "trace_id": "tr-udr-db",
        "spans": [
            {
                "service": "udm",
                "name": "subscriber.profile.lookup",
                "status": "degraded",
                "attributes": {"udr": "udr", "database": "subscriber"},
            },
            {
                "service": "udr",
                "name": "database.read",
                "status": "degraded",
                "attributes": {"latency": "high", "profile": "subscriber"},
            },
        ],
        "metrics": [{"service": "udr", "name": "database_read_latency_ms", "value": 1800, "unit": "ms"}],
        "logs": [{"service": "udr", "level": "WARN", "message": "Subscriber profile read latency high in UDR database"}],
    },
}


def otel_for_cause(cause_id, distractor_id=None):
    bundle = deepcopy(OTEL_SIGNAL_TEMPLATES[cause_id])
    if distractor_id:
        distractor = OTEL_SIGNAL_TEMPLATES[distractor_id]
        bundle.setdefault("logs", []).append(
            {
                "service": "legacy-note",
                "level": "INFO",
                "message": "Stale historical incident note mentioned %s"
                % CAUSES[distractor_id]["label"],
            }
        )
        bundle.setdefault("spans", []).append(
            {
                "service": "legacy-note",
                "name": "stale.operator.note",
                "status": "ok",
                "attributes": {"stale_cause": distractor["trace_id"]},
            }
        )
    return bundle


def otel_text(otel):
    if not otel:
        return ""
    parts = [str(otel.get("trace_id", ""))]
    for span in otel.get("spans", []):
        parts.extend(
            [
                span.get("service", ""),
                span.get("name", ""),
                span.get("status", ""),
                " ".join("%s %s" % (key, value) for key, value in span.get("attributes", {}).items()),
            ]
        )
    for metric in otel.get("metrics", []):
        parts.extend([metric.get("service", ""), metric.get("name", ""), str(metric.get("value", "")), metric.get("unit", "")])
    for log in otel.get("logs", []):
        parts.extend([log.get("service", ""), log.get("level", ""), log.get("message", "")])
    return " ".join(parts)


def facts_from_otel(otel):
    facts = []
    if not otel:
        return facts
    for span in otel.get("spans", []):
        service = ALIASES.get(str(span.get("service", "")).lower(), str(span.get("service", "")).lower())
        if service:
            facts.append(
                {
                    "entity": service,
                    "attribute": span.get("name", "span"),
                    "value": span.get("status", "unknown"),
                    "confidence": 0.82 if span.get("status") in {"error", "degraded", "timeout"} else 0.35,
                }
            )
    for metric in otel.get("metrics", []):
        service = ALIASES.get(str(metric.get("service", "")).lower(), str(metric.get("service", "")).lower())
        if service:
            facts.append(
                {
                    "entity": service,
                    "attribute": metric.get("name", "metric"),
                    "value": str(metric.get("value", "")) + str(metric.get("unit", "")),
                    "confidence": 0.88,
                }
            )
    for log in otel.get("logs", []):
        service = ALIASES.get(str(log.get("service", "")).lower(), str(log.get("service", "")).lower())
        if service and service != "legacy-note":
            facts.append(
                {
                    "entity": service,
                    "attribute": "log_" + str(log.get("level", "info")).lower(),
                    "value": log.get("message", ""),
                    "confidence": 0.86 if log.get("level") in {"WARN", "ERROR"} else 0.45,
                }
            )
    return facts


def enrich_with_otel(scenarios):
    enriched = []
    for scenario in scenarios:
        item = deepcopy(scenario)
        item["otel"] = otel_for_cause(item["gold"])
        enriched.append(item)
    return enriched


def normalize_text(text):
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9_.:-]+", text)
    tokens = []
    for token in raw_tokens:
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens


def concepts_from_text(text):
    concepts = []
    for token in normalize_text(text):
        concepts.append(ALIASES.get(token, token))
    return set(concepts)


def fact_text(facts):
    return " ".join(
        "%s %s %s" % (fact.get("entity", ""), fact.get("attribute", ""), fact.get("value", ""))
        for fact in facts
    )


def severity(value):
    tokens = concepts_from_text(value)
    if tokens & set(["degraded", "timeout", "failed", "rejected", "crashloop", "drop", "latency"]):
        return 1.0
    if tokens & set(["high", "stale", "missing", "invalid", "reset", "flapping", "mismatch"]):
        return 0.9
    if tokens & set(["healthy", "normal", "ok"]):
        return 0.15
    return 0.45


class TypedGraph(object):
    def __init__(self):
        self.edges = defaultdict(list)
        self.edge_weights = {}

    def add(self, source, relation, target, weight=1.0):
        self.edges[source].append((target, relation, weight))
        self.edges[target].append((source, "rev_" + relation, weight))
        self.edge_weights[(source, relation, target)] = weight

    def shortest_path(self, sources, target, max_depth=5):
        queue = deque()
        seen = set()
        for source in sources:
            queue.append((source, [source]))
            seen.add(source)
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            if len(path) > max_depth:
                continue
            for nxt, relation, _weight in self.edges.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [relation, nxt]))
        return []

    def distance(self, sources, target, max_depth=5):
        path = self.shortest_path(sources, target, max_depth=max_depth)
        if not path:
            return None
        return int((len(path) - 1) / 2)


class MSKGLearningAgent(object):
    def __init__(self):
        self.graph = TypedGraph()
        self.learned = defaultdict(float)
        self._build_graph()

    def _build_graph(self):
        base_edges = [
            ("ue", "accesses", "gnb"),
            ("gnb", "n2", "amf"),
            ("gnb", "n3", "upf"),
            ("amf", "sba", "nrf"),
            ("amf", "uses", "ausf"),
            ("ausf", "uses", "udm"),
            ("udm", "stores_in", "udr"),
            ("smf", "sba", "nrf"),
            ("smf", "uses", "pcf"),
            ("smf", "n4", "upf"),
            ("upf", "n6", "dn"),
            ("pcf", "controls", "policy"),
            ("policy", "binds", "dnn"),
            ("policy", "binds", "slice"),
            ("nssf", "selects", "slice"),
            ("amf", "uses", "nssf"),
            ("slice", "scoped_by", "tac"),
            ("smf_pod", "runs", "smf"),
            ("smf_pod", "managed_by", "kubernetes"),
            ("llm_endpoint", "serves", "qwen"),
            ("l40", "routes_to", "llm_endpoint"),
            ("vdi", "routes_to", "l40"),
            ("n2", "protocol", "sctp"),
            ("n2", "protocol", "ngap"),
            ("n3", "protocol", "gtpu"),
            ("n6", "protocol", "gtpu"),
            ("n4", "protocol", "pfcp"),
        ]
        for source, relation, target in base_edges:
            self.graph.add(source, relation, target, 1.0)

        for cause_id, spec in CAUSES.items():
            cause_node = "cause:" + cause_id
            for node in spec["nodes"]:
                self.graph.add(cause_node, "explained_by", node, 1.0)
            for tool in spec["tools"]:
                self.graph.add(cause_node, "observable_by", "tool:" + tool, 0.8)
            for action in spec["actions"]:
                self.graph.add(cause_node, "remediated_by", "action:" + action, 0.7)

    def observed_nodes(self, query, facts):
        concepts = concepts_from_text(query + " " + fact_text(facts))
        nodes = set()
        known_nodes = set(self.graph.edges.keys())
        for concept in concepts:
            if concept in known_nodes:
                nodes.add(concept)
        for fact in facts:
            entity = ALIASES.get(str(fact.get("entity", "")).lower(), str(fact.get("entity", "")).lower())
            if entity in known_nodes:
                nodes.add(entity)
        return nodes or set(["ue"])

    def semantic_score(self, cause_id, query):
        concepts = concepts_from_text(query)
        tags = set(CAUSES[cause_id]["tags"])
        if not concepts:
            return 0.0, []
        hits = sorted(concepts & tags)
        score = len(hits) / max(1.0, math.sqrt(len(tags)))
        return score, hits

    def morphology_score(self, cause_id, query, facts):
        sources = self.observed_nodes(query, facts)
        cause_node = "cause:" + cause_id
        distances = []
        best_path = []
        for source in sorted(sources):
            path = self.graph.shortest_path([source], cause_node, max_depth=4)
            if path:
                dist = int((len(path) - 1) / 2)
                distances.append(dist)
                if not best_path or len(path) < len(best_path):
                    best_path = path
        if not distances:
            return 0.0, []
        score = sum(1.0 / (1.0 + dist) for dist in distances) / len(distances)
        return score, best_path

    def kg_score(self, cause_id, facts):
        if not facts:
            return 0.0, []
        cause_node = "cause:" + cause_id
        evidence = []
        total = 0.0
        for fact in facts:
            entity = ALIASES.get(str(fact.get("entity", "")).lower(), str(fact.get("entity", "")).lower())
            value = str(fact.get("value", ""))
            conf = float(fact.get("confidence", 0.8))
            if entity not in self.graph.edges:
                continue
            dist = self.graph.distance([entity], cause_node, max_depth=4)
            if dist is None:
                continue
            fact_concepts = concepts_from_text(
                "%s %s %s" % (entity, fact.get("attribute", ""), fact.get("value", ""))
            )
            overlap = fact_concepts & set(CAUSES[cause_id]["tags"])
            contribution = conf * severity(value) / (1.0 + dist) + 0.08 * len(overlap)
            total += contribution
            evidence.append(
                {
                    "fact": "%s.%s=%s" % (entity, fact.get("attribute", ""), value),
                    "distance": dist,
                    "contribution": round(contribution, 4),
                    "overlap": sorted(overlap),
                }
            )
        return total / max(1, len(facts)), evidence

    def vector_rag_score(self, cause_id, query, facts, otel=None):
        evidence_text = query
        evidence_concepts = concepts_from_text(evidence_text)
        cause_text = " ".join(
            [
                cause_id,
                CAUSES[cause_id]["label"],
                " ".join(CAUSES[cause_id]["tags"]),
                " ".join(CAUSES[cause_id]["nodes"]),
                otel_text(OTEL_SIGNAL_TEMPLATES[cause_id]),
            ]
        )
        cause_concepts = concepts_from_text(cause_text)
        if not evidence_concepts or not cause_concepts:
            return 0.0, []
        hits = sorted(evidence_concepts & cause_concepts)
        score = len(hits) / math.sqrt(len(cause_concepts))
        return score, hits

    def graph_rag_score(self, cause_id, query, facts, otel=None):
        vector, hits = self.vector_rag_score(cause_id, query, facts, otel=otel)
        morph, path = self.morphology_score(cause_id, query + " " + otel_text(otel), facts)
        return 0.55 * vector + 0.45 * morph, {
            "hits": hits,
            "path": path,
            "evidence": [],
        }

    def learned_score(self, cause_id, query):
        concepts = concepts_from_text(query)
        if not concepts:
            return 0.0
        raw = sum(self.learned[(concept, cause_id)] for concept in concepts) / math.sqrt(len(concepts))
        return math.tanh(raw)

    def rank(self, query, facts=None, mode="mskg_learning", otel=None):
        facts = list(facts or [])
        all_facts = facts + facts_from_otel(otel)
        rows = []
        for cause_id in sorted(CAUSES):
            sem, hits = self.semantic_score(cause_id, query)
            morph, path = self.morphology_score(cause_id, query, all_facts)
            kg, evidence = self.kg_score(cause_id, all_facts)
            mem = self.learned_score(cause_id, query)
            vector, vector_hits = self.vector_rag_score(cause_id, query, all_facts, otel=otel)
            graph_rag, graph_rag_details = self.graph_rag_score(cause_id, query, all_facts, otel=otel)
            if mode == "semantic_only":
                score = sem
            elif mode == "vector_rag":
                score = vector
            elif mode == "graph_rag":
                score = graph_rag
            elif mode == "morphology_semantic":
                score = 0.58 * sem + 0.42 * morph
            else:
                score = 0.18 * sem + 0.20 * morph + 0.60 * kg + 0.02 * mem
            rows.append(
                {
                    "cause_id": cause_id,
                    "label": CAUSES[cause_id]["label"],
                    "score": round(score, 6),
                    "semantic": round(sem, 6),
                    "morphology": round(morph, 6),
                    "knowledge_graph": round(kg, 6),
                    "learned_memory": round(mem, 6),
                    "vector_rag": round(vector, 6),
                    "graph_rag": round(graph_rag, 6),
                    "semantic_hits": hits,
                    "vector_hits": vector_hits,
                    "morphology_path": path,
                    "kg_evidence": evidence,
                    "graph_rag_evidence": graph_rag_details,
                    "suggested_tools": CAUSES[cause_id]["tools"],
                    "suggested_actions": CAUSES[cause_id]["actions"],
                }
            )
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows

    def learn(self, scenario, predicted=None):
        query = scenario["query"]
        gold = scenario["gold"]
        facts = list(scenario.get("facts", [])) + facts_from_otel(scenario.get("otel"))
        pred = predicted or self.rank(query, facts, mode="mskg_learning", otel=scenario.get("otel"))[0]["cause_id"]
        reward = 1.0 if pred == gold else 1.4
        concepts = concepts_from_text(query + " " + fact_text(facts) + " " + otel_text(scenario.get("otel")))
        gold_specific = set(CAUSES[gold]["tags"] + CAUSES[gold]["nodes"])
        pred_specific = set(CAUSES[pred]["tags"] + CAUSES[pred]["nodes"]) if pred in CAUSES else set()
        for concept in concepts:
            if concept in GENERIC_CONCEPTS:
                continue
            if TAG_FREQUENCY[concept] > 1:
                continue
            if concept not in gold_specific:
                continue
            self.learned[(concept, gold)] += 0.12 * reward
            if pred != gold and concept in pred_specific:
                self.learned[(concept, pred)] -= 0.06

    def ask(self, query, facts=None, mode="mskg_learning", top_k=3, otel=None):
        ranked = self.rank(query, facts or [], mode=mode, otel=otel)
        best = ranked[0]
        answer = [
            "Most likely cause: %s (%s)" % (best["label"], best["cause_id"]),
            "Score: %.4f | semantic %.4f | morphology %.4f | KG %.4f | learned %.4f"
            % (
                best["score"],
                best["semantic"],
                best["morphology"],
                best["knowledge_graph"],
                best["learned_memory"],
            ),
            "Tools: %s" % ", ".join(best["suggested_tools"][:4]),
            "Actions: %s" % ", ".join(best["suggested_actions"][:3]),
        ]
        if best["morphology_path"]:
            answer.append("Morphology path: %s" % " -> ".join(best["morphology_path"]))
        if best["kg_evidence"]:
            answer.append("KG evidence: %s" % "; ".join(item["fact"] for item in best["kg_evidence"][:3]))
        return {"answer": "\n".join(answer), "ranking": ranked[:top_k], "mode": mode}


def evaluate(agent, scenarios, mode):
    ranks = []
    top1 = 0
    top3 = 0
    examples = []
    for scenario in scenarios:
        ranked = agent.rank(scenario["query"], scenario.get("facts", []), mode=mode, otel=scenario.get("otel"))
        ordered = [item["cause_id"] for item in ranked]
        rank = ordered.index(scenario["gold"]) + 1
        ranks.append(rank)
        top1 += 1 if rank == 1 else 0
        top3 += 1 if rank <= 3 else 0
        examples.append(
            {
                "id": scenario["id"],
                "gold": scenario["gold"],
                "predicted": ordered[0],
                "rank": rank,
                "top_score": ranked[0]["score"],
                "gold_score": next(item["score"] for item in ranked if item["cause_id"] == scenario["gold"]),
            }
        )
    total = len(scenarios)
    return {
        "mode": mode,
        "top1": round(top1 / total, 4),
        "top3": round(top3 / total, 4),
        "mrr": round(sum(1.0 / rank for rank in ranks) / total, 4),
        "mean_rank": round(sum(ranks) / total, 4),
        "examples": examples,
    }


def stress_scenarios():
    variants = []
    distractor_ids = sorted(CAUSES)
    for index, scenario in enumerate(EVAL_SCENARIOS):
        distractor_id = distractor_ids[(index + 3) % len(distractor_ids)]
        distractor = CAUSES[distractor_id]["label"]
        base_facts = list(scenario.get("facts", []))
        variants.extend(
            [
                {
                    "id": scenario["id"] + "_noise",
                    "query": scenario["query"]
                    + " NOC ghi chu: node CPU binh thuong, mot alarm cu noi ve %s." % distractor,
                    "facts": base_facts
                    + [{"entity": "amf", "attribute": "status", "value": "healthy", "confidence": 0.25}],
                    "otel": otel_for_cause(scenario["gold"], distractor_id=distractor_id),
                    "gold": scenario["gold"],
                },
                {
                    "id": scenario["id"] + "_sparse",
                    "query": "Ticket van hanh: dich vu bat thuong sau maintenance, can RCA khong thao tac write.",
                    "facts": base_facts,
                    "otel": otel_for_cause(scenario["gold"]),
                    "gold": scenario["gold"],
                },
                {
                    "id": scenario["id"] + "_stale_note",
                    "query": scenario["query"]
                    + " Tai lieu cu trong wiki goi y %s, nhung telemetry hien tai can duoc uu tien." % distractor,
                    "facts": base_facts,
                    "otel": otel_for_cause(scenario["gold"], distractor_id=distractor_id),
                    "gold": scenario["gold"],
                },
                {
                    "id": scenario["id"] + "_mixed_lang",
                    "query": "Mixed EN/VN incident after rollout: " + scenario["query"] + " Need bounded RCA with evidence.",
                    "facts": base_facts,
                    "otel": otel_for_cause(scenario["gold"]),
                    "gold": scenario["gold"],
                },
                {
                    "id": scenario["id"] + "_fact_only",
                    "query": "Operator note is vague: thue bao bao loi dich vu, dashboard can doi chieu telemetry.",
                    "facts": base_facts,
                    "otel": otel_for_cause(scenario["gold"]),
                    "gold": scenario["gold"],
                },
            ]
        )
    return variants


def run_benchmark(iterations=100):
    started = time.perf_counter()
    train = enrich_with_otel(TRAIN_SCENARIOS)
    eval_scenarios = enrich_with_otel(EVAL_SCENARIOS)
    cold_agent = MSKGLearningAgent()
    semantic = evaluate(cold_agent, eval_scenarios, "semantic_only")
    vector_rag = evaluate(cold_agent, eval_scenarios, "vector_rag")
    graph_rag = evaluate(cold_agent, eval_scenarios, "graph_rag")
    morphology_semantic = evaluate(cold_agent, eval_scenarios, "morphology_semantic")
    mskg_before = evaluate(cold_agent, eval_scenarios, "mskg_learning")
    stress = stress_scenarios()
    stress_semantic = evaluate(cold_agent, stress, "semantic_only")
    stress_vector_rag = evaluate(cold_agent, stress, "vector_rag")
    stress_graph_rag = evaluate(cold_agent, stress, "graph_rag")
    stress_morphology_semantic = evaluate(cold_agent, stress, "morphology_semantic")
    stress_mskg_before = evaluate(cold_agent, stress, "mskg_learning")

    learned_agent = MSKGLearningAgent()
    for _ in range(max(1, int(iterations))):
        for scenario in train:
            pred = learned_agent.rank(
                scenario["query"], scenario.get("facts", []), mode="mskg_learning", otel=scenario.get("otel")
            )[0]["cause_id"]
            learned_agent.learn(scenario, predicted=pred)
    mskg_after = evaluate(learned_agent, eval_scenarios, "mskg_learning")
    stress_mskg_after = evaluate(learned_agent, stress, "mskg_learning")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "benchmark": "5gc-mskg-agent-learning-otel-baselines",
        "iterations": int(iterations),
        "train_scenarios": len(train),
        "eval_scenarios": len(eval_scenarios),
        "stress_eval_scenarios": len(stress),
        "elapsed_ms": round(elapsed_ms, 2),
        "results": [semantic, vector_rag, graph_rag, morphology_semantic, mskg_before, mskg_after],
        "stress_results": [
            stress_semantic,
            stress_vector_rag,
            stress_graph_rag,
            stress_morphology_semantic,
            stress_mskg_before,
            stress_mskg_after,
        ],
        "interpretation": {
            "semantic_only": "Intent/text matching only; no operational shape or dynamic facts.",
            "vector_rag": "Lexical chunk retrieval over simulated OpenTelemetry logs, metrics, and traces.",
            "graph_rag": "GraphRAG-style dependency/evidence scoring over simulated OpenTelemetry entities.",
            "morphology_semantic": "Adds NF/interface/tool-affordance graph paths.",
            "mskg_learning_before": "Adds KG evidence before episodic learning.",
            "mskg_learning_after": "Adds episode-derived concept-to-cause weights.",
            "stress": "Deterministic perturbations add sparse tickets, stale notes, distractor alarms, and mixed-language wording.",
        },
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MSKG Agent Learning Demo</title>
  <style>
    :root { --bg:#f6f7f9; --surface:#fff; --text:#17201f; --muted:#60716e; --line:#d7dfdd; --teal:#0f766e; --blue:#1f5fbf; --amber:#a15c07; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:var(--bg); color:var(--text); font-family:Arial,Helvetica,sans-serif; letter-spacing:0; }
    header { min-height:68px; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 22px; background:var(--surface); border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:20px; line-height:1.2; }
    header span { color:var(--muted); font-size:13px; }
    main { max-width:1380px; margin:0 auto; padding:18px 22px 24px; display:grid; grid-template-columns:minmax(360px,0.9fr) minmax(420px,1.1fr); gap:16px; }
    section { background:var(--surface); border:1px solid var(--line); border-radius:8px; min-width:0; }
    .head { min-height:52px; display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px 14px; border-bottom:1px solid var(--line); }
    .head h2 { margin:0; font-size:15px; }
    .body { padding:14px; }
    textarea, input { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px; font:inherit; }
    textarea { min-height:126px; resize:vertical; line-height:1.45; }
    button { min-height:36px; border:1px solid var(--line); border-radius:6px; background:#fff; padding:0 12px; cursor:pointer; font:inherit; }
    button.primary { border-color:var(--teal); background:var(--teal); color:#fff; }
    .toolbar { display:flex; gap:8px; flex-wrap:wrap; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--line); }
    .metric { padding:12px 14px; border-right:1px solid var(--line); min-height:72px; }
    .metric:last-child { border-right:0; }
    .metric label { display:block; color:var(--muted); font-size:12px; margin-bottom:8px; }
    .metric strong { font-size:22px; }
    pre { min-height:220px; max-height:430px; overflow:auto; margin:0; padding:12px; border-radius:6px; background:#101918; color:#e8f5f2; white-space:pre-wrap; line-height:1.45; font-size:13px; }
    table { width:100%; border-collapse:collapse; font-size:13px; min-width:620px; }
    th, td { padding:9px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); }
    .scroll { overflow:auto; max-height:360px; }
    .pill { display:inline-flex; align-items:center; min-height:26px; border-radius:999px; padding:0 9px; background:#d9f2ee; color:#0c504c; font-size:12px; }
    .row { display:grid; gap:10px; }
    @media(max-width:980px){ header, main { padding-left:12px; padding-right:12px; } header { align-items:flex-start; flex-direction:column; } main { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body>
  <header>
    <div><h1>MSKG Agent Learning Demo</h1><span>Morphology + Semantic + Knowledge Graph cho 5G Core agent learning</span></div>
    <span id="status" class="pill">checking</span>
  </header>
  <main>
    <section>
      <div class="metrics">
        <div class="metric"><label>Semantic Top-1</label><strong id="m0">-</strong></div>
        <div class="metric"><label>Vector-RAG Top-1</label><strong id="m1">-</strong></div>
        <div class="metric"><label>GraphRAG Top-1</label><strong id="m2">-</strong></div>
        <div class="metric"><label>MSKG Top-1</label><strong id="m3">-</strong></div>
      </div>
      <div class="head"><h2>Benchmark</h2><div class="toolbar"><button id="benchBtn" class="primary">Run</button></div></div>
      <div class="body"><pre id="benchOut">No benchmark run.</pre></div>
    </section>
    <section>
      <div class="head"><h2>Ask MSKG Agent</h2><div class="toolbar"><button id="askBtn" class="primary">Ask</button></div></div>
      <div class="body row">
        <textarea id="question">Khach hang bao PDU bi timeout sau khi doi route UPF; pod SMF khong crash.</textarea>
        <textarea id="facts">[{"entity":"smf","attribute":"n4_state","value":"degraded","confidence":0.93}]</textarea>
        <pre id="askOut">No query.</pre>
      </div>
    </section>
    <section style="grid-column:1/-1">
      <div class="head"><h2>Latest Ranking</h2><span class="pill">explainable scoring</span></div>
      <div class="body"><div class="scroll"><table><thead><tr><th>Cause</th><th>Score</th><th>Semantic</th><th>Morphology</th><th>KG</th><th>Learned</th></tr></thead><tbody id="rankBody"></tbody></table></div></div>
    </section>
  </main>
  <script>
    const el = id => document.getElementById(id);
    async function jsonFetch(url, options = {}) {
      const r = await fetch(url, { ...options, headers: { "Content-Type":"application/json", ...(options.headers || {}) } });
      const p = await r.json();
      if (!r.ok) throw new Error(p.error || r.statusText);
      return p;
    }
    function renderRank(rows) {
      el("rankBody").innerHTML = rows.map(r => `<tr><td>${r.label}<br><span class="pill">${r.cause_id}</span></td><td>${r.score}</td><td>${r.semantic}</td><td>${r.morphology}</td><td>${r.knowledge_graph}</td><td>${r.learned_memory}</td></tr>`).join("");
    }
    async function runBenchmark() {
      el("benchOut").textContent = "running...";
      const data = await jsonFetch("/benchmark?iterations=100");
      const results = data.results;
      el("m0").textContent = results[0].top1;
      el("m1").textContent = results[1].top1;
      el("m2").textContent = results[2].top1;
      el("m3").textContent = results[5].top1;
      el("benchOut").textContent = JSON.stringify(data, null, 2);
    }
    async function ask() {
      let facts = [];
      try { facts = JSON.parse(el("facts").value || "[]"); } catch (e) { el("askOut").textContent = "Invalid facts JSON"; return; }
      const data = await jsonFetch("/ask", { method:"POST", body: JSON.stringify({ question: el("question").value, facts, mode:"mskg_learning", train_iterations:100 }) });
      el("askOut").textContent = data.answer;
      renderRank(data.ranking || []);
    }
    async function health() {
      const data = await jsonFetch("/health");
      el("status").textContent = data.status + " | causes " + data.causes;
    }
    el("benchBtn").addEventListener("click", runBenchmark);
    el("askBtn").addEventListener("click", ask);
    health().then(runBenchmark).then(ask).catch(e => { el("status").textContent = String(e.message || e); });
  </script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "MSKGAgentLearningDemo/0.1"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _send_headers(self, status, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def write_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8")
        self.wfile.write(data)

    def read_json(self):
        size = int(self.headers.get("Content-Length", "0") or "0")
        if size > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        data = self.rfile.read(size) if size else b"{}"
        return json.loads(data.decode("utf-8"))

    def do_OPTIONS(self):
        self._send_headers(204, "text/plain; charset=utf-8")

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path in ("/", "/index.html"):
            data = HTML.encode("utf-8")
            self._send_headers(200, "text/html; charset=utf-8")
            self.wfile.write(data)
            return
        if parsed.path == "/health":
            self.write_json(200, {"status": "ok", "service": "mskg-agent-learning", "causes": len(CAUSES)})
            return
        if parsed.path == "/benchmark":
            query = parse_qs(parsed.query)
            iterations = int(query.get("iterations", ["100"])[0])
            self.write_json(200, run_benchmark(iterations=iterations))
            return
        if parsed.path == "/graph":
            self.write_json(
                200,
                {
                    "nodes": sorted(MSKGLearningAgent().graph.edges.keys()),
                    "causes": CAUSES,
                    "train_scenarios": TRAIN_SCENARIOS,
                    "eval_scenarios": EVAL_SCENARIOS,
                },
            )
            return
        self.write_json(404, {"error": "not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/ask":
            self.write_json(404, {"error": "not found"})
            return
        try:
            payload = self.read_json()
            agent = MSKGLearningAgent()
            for _ in range(int(payload.get("train_iterations", 100))):
                for scenario in TRAIN_SCENARIOS:
                    agent.learn(scenario)
            result = agent.ask(
                payload.get("question", ""),
                facts=payload.get("facts", []),
                mode=payload.get("mode", "mskg_learning"),
                top_k=int(payload.get("top_k", 5)),
            )
            self.write_json(200, result)
        except Exception as exc:
            self.write_json(400, {"error": str(exc)})


def self_test():
    result = run_benchmark(iterations=5)
    assert result["eval_scenarios"] == len(EVAL_SCENARIOS)
    assert result["results"][-1]["top1"] >= result["results"][0]["top1"]
    agent = MSKGLearningAgent()
    reply = agent.ask("SMF PFCP timeout tren N4", [{"entity": "smf", "attribute": "n4_state", "value": "degraded"}])
    assert reply["ranking"][0]["cause_id"] == "n4_pfcp_path"
    print("MSKG agent learning self-test OK")


def main():
    parser = argparse.ArgumentParser(description="5G Core MSKG agent learning demo.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ask", default="")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.benchmark:
        result = run_benchmark(iterations=args.iterations)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for row in result["results"]:
                print("%s top1=%.3f top3=%.3f mrr=%.3f mean_rank=%.3f" % (
                    row["mode"], row["top1"], row["top3"], row["mrr"], row["mean_rank"]
                ))
        return 0
    if args.ask:
        agent = MSKGLearningAgent()
        for _ in range(args.iterations):
            for scenario in TRAIN_SCENARIOS:
                agent.learn(scenario)
        result = agent.ask(args.ask, facts=[])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.serve:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
        print("Serving MSKG demo on http://%s:%s" % (args.host, args.port), flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
