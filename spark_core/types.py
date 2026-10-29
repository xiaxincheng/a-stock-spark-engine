"""Shared dataclasses and enums for the Spark Engine system."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    DEBATING = "debating"
    VERIFIED = "verified"
    REJECTED = "rejected"
    ACTIVE = "active"
    SUSPENDED = "suspended"  # 数据不足，等待重审


class DebateVerdict(Enum):
    PASS = "pass"          # score >= 0.6, 进入验证
    REVISE = "revise"      # score 0.4-0.6, 修正后重辩
    REJECT = "reject"      # score < 0.2, 进入失败博物馆
    SUSPEND = "suspend"    # score 0.2-0.4, 数据不足, 30天后重审


class VerificationGate(Enum):
    SANDBOX = "sandbox"
    FULL_BACKTEST = "full_backtest"
    ROBUSTNESS = "robustness"


@dataclass
class KnowledgeNode:
    """A concept in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: str = ""          # 'strategy', 'factor', 'condition', 'group', 'analogy'
    name: str = ""
    description: str = ""
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KnowledgeEdge:
    """Relationship between two knowledge nodes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str = ""
    target_id: str = ""
    relation: str = ""          # 'categorized_as', 'strengthens', 'supports', 'contradicts', 'derives_from'
    weight: float = 1.0
    evidence: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FactorCondition:
    """An IF-THEN rule for the rules engine."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition_expr: str = ""     # e.g., "price_deviation_from_ma60 < -15% AND volume_ratio > 1.0"
    then_action: str = ""        # e.g., "STRONG_BUY", "FILTER_OUT", "BUY"
    confidence: float = 1.0
    evidence_source: str = ""
    falsification_count: int = 0
    verification_count: int = 0
    is_active: bool = True


@dataclass
class CrossDomainAnalogy:
    """Mapping from another discipline to finance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_domain: str = ""      # 'physics', 'chemistry', 'biology'
    source_pattern: str = ""     # e.g., "momentum conservation"
    finance_mapping: str = ""    # e.g., "trend continuation = inertia"
    confidence: float = 0.5
    examples: list[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """A testable strategy idea."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    direction: Direction = Direction.LONG
    name: str = ""               # human-readable name
    description: str = ""
    condition_expr: str = ""     # falsifiable IF condition
    signal_logic_desc: str = ""  # natural language description of scan logic
    source_chain: list[str] = field(default_factory=list)
    generation_layer: int = 1    # L1, L2, or L3
    combo_type: str = ""          # L1: 'AND', 'OR', '时序', '过滤'; L2/L3: ''
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    debate_score: float = 0.0
    verification: Optional[dict] = None
    reward_score: Optional[dict] = None
    rejection_reason: Optional[str] = None
    verified_at: Optional[str] = None
    backtested: bool = False     # 是否已通过全量回测（训练/测试分离验证）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SparkTask:
    """A unit of work in the task queue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: str = ""          # 'generate_l1', 'verify', 'debate', 'reward', 'reverse_mine'
    hypothesis_id: Optional[str] = None
    priority: int = 5            # 1-10, lower = higher priority
    payload: dict = field(default_factory=dict)
    status: str = "pending"      # pending, running, done, failed
    retries: int = 0
    max_retries: int = 3
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Checkpoint:
    """Engine state snapshot."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    hypothesis_queue: list[str] = field(default_factory=list)
    debate_round_counts: dict = field(default_factory=dict)    # hypothesis_id -> round
    evolution_generation: int = 0
    task_id_dedup: set = field(default_factory=set)
    completed_hypotheses: list[str] = field(default_factory=list)
    failed_hypotheses: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class DebateRecord:
    """One round of Agent A vs Agent B."""
    hypothesis_id: str = ""
    round_number: int = 1
    agent_a_claim: str = ""
    agent_b_attack: str = ""
    agent_b_weapon: str = ""     # which of the 8 weapons was used
    judge_ruling: str = ""       # 'a_wins', 'b_wins', 'tie'
    judge_reasoning: str = ""
    updated_score: float = 0.0


@dataclass
class VerificationResult:
    """Result of the 3-gate verification pipeline."""
    hypothesis_id: str = ""
    gate: VerificationGate = VerificationGate.SANDBOX
    passed: bool = False
    backtest_result: Optional[dict] = None
    failure_reason: Optional[str] = None
    metrics: dict = field(default_factory=dict)
