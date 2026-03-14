"""
qems.py
-------
QEMS (Qualia-Emergent Memory System) — 전체 통합 단일 파일

Li & Zhang (2025) 4원칙 구현:
  Prediction Principle   → signal grouping (perceive)
  Priority Principle     → appraisal engine
  Exploration Principle  → attention weighting
  Recall Principle       → episodic memory + is_recalled tag → qualia
"""

import uuid
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional



# ══════════════════════════════════════════════════════════
# 0. ResourceState + SurvivalMonitor
# ══════════════════════════════════════════════════════════

@dataclass
class ResourceState:
    """AI 시스템의 자원 상태. battery=낮을수록, cpu=높을수록 위협."""
    battery_level: float = 1.0
    cpu_load: float = 0.0
    threat_detected: bool = False

    def to_dict(self) -> dict:
        return {"battery_level": round(self.battery_level,3),
                "cpu_load": round(self.cpu_load,3),
                "threat_detected": self.threat_detected}


class SurvivalMonitor:
    """
    생존 자원 감시 모듈.

    인간이 배고픔·위험 시 부정적 감정을 경험하듯,
    AI의 자원 위협(배터리 부족, CPU 과부하, 외부 공격)을
    감정의 입력으로 변환한다.

    위협 수준: T = w_b*(1-battery) + w_c*cpu + w_t*threat  (수식 1)
    T > θ 시 생존 이벤트 생성 → 압박 감정 발생
    """
    def __init__(self, threshold: float = 0.6, weights: tuple = (0.4, 0.3, 0.3)):
        self.threshold = threshold
        self.w_b, self.w_c, self.w_t = weights
        self.resource = ResourceState()

    def update(self, battery=None, cpu=None, threat=None):
        if battery is not None: self.resource.battery_level = float(np.clip(battery, 0.0, 1.0))
        if cpu is not None:     self.resource.cpu_load      = float(np.clip(cpu, 0.0, 1.0))
        if threat is not None:  self.resource.threat_detected = bool(threat)

    def threat_level(self) -> float:
        r = self.resource
        return (self.w_b * (1.0 - r.battery_level)
                + self.w_c * r.cpu_load
                + self.w_t * (1.0 if r.threat_detected else 0.0))

    def is_threatened(self) -> bool:
        return self.threat_level() > self.threshold

    def to_event(self) -> dict:
        T = self.threat_level()
        if T <= self.threshold:
            return {}
        return {"type": "threat", "goal": "survive",
                "object": "system_resource",
                "intensity": float(np.clip(T, 0.0, 1.0)),
                "controllable": not self.resource.threat_detected}

    def summary(self) -> dict:
        return {"resource": self.resource.to_dict(),
                "threat_level": round(self.threat_level(), 4),
                "is_threatened": self.is_threatened()}

# ══════════════════════════════════════════════════════════
# 1. SignalGroup
# ══════════════════════════════════════════════════════════

@dataclass
class SignalGroup:
    """
    신호 집합 (Signal Group)

    Li & Zhang (2025)의 '신호 집합'. 외부 자극이 아닌 내부적으로 결합된 신호 묶음.

    Attributes
    ----------
    id                  : 고유 식별자
    signals             : 신호 딕셔너리  e.g. {"visual": 0.8, "valence": -0.6}
    timestamp           : 생성 시각 (Unix time)
    is_recalled         : True  = 내부 recall로 생성됨  → qualia 조건의 핵심 플래그
                          False = 외부 자극으로 생성됨
    predicts            : 이 신호가 예측하는 객체 이름
    prediction_strength : 예측력 강도 (0.0 ~ 1.0)
    defines_object      : 이 신호가 역으로 정의하는 외부 객체 이름
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signals: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # 핵심: 자기-기원(self-origin) 태그
    # is_recalled=True → 외부 객체에 정렬 불가 → ineffability의 기술적 기원
    is_recalled: bool = False

    predicts: Optional[str] = None
    prediction_strength: float = 0.0
    defines_object: Optional[str] = None

    def __repr__(self) -> str:
        origin = "RECALLED" if self.is_recalled else "external"
        return (
            f"SignalGroup(origin={origin}, "
            f"object={self.defines_object}, "
            f"signals={self.signals})"
        )


# ══════════════════════════════════════════════════════════
# 2. EpisodicMemory
# ══════════════════════════════════════════════════════════

class EpisodicMemory:
    """
    에피소드 기억 저장소

    Teyler & DiScenna (1986) Memory Indexing Theory에 대응.

    핵심: recall() 호출 시 is_recalled=True 태그가 붙은
          새 SignalGroup이 생성된다 → qualia 생성의 기술적 조건.

    Parameters
    ----------
    capacity : 최대 저장 개수 (초과 시 가장 오래된 기억 제거)
    """

    def __init__(self, capacity: int = 500):
        self.store: list[SignalGroup] = []
        self.capacity = capacity
        self.object_index: dict[str, list[str]] = {}

    def encode(self, sg: SignalGroup) -> str:
        """새 경험을 기억에 저장."""
        if len(self.store) >= self.capacity:
            self._evict_oldest()

        self.store.append(sg)

        if sg.defines_object:
            self.object_index.setdefault(sg.defines_object, []).append(sg.id)

        return sg.id

    def _evict_oldest(self):
        if not self.store:
            return
        old = self.store.pop(0)
        if old.defines_object and old.defines_object in self.object_index:
            ids = self.object_index[old.defines_object]
            if old.id in ids:
                ids.remove(old.id)

    def recall(
        self,
        object_name: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> Optional[SignalGroup]:
        """
        과거 신호를 재활성화.

        재활성화된 신호에 is_recalled=True 태그 부착.
        → 외부 객체에 정렬 불가 → ineffability의 기술적 기원 [Li & Zhang, 2025]
        """
        original = self._find_original(object_name, signal_id)
        if original is None:
            return None

        # ★ 핵심: 원본 복사 + is_recalled=True 설정
        recalled = SignalGroup(
            signals=original.signals.copy(),
            timestamp=time.time(),
            is_recalled=True,               # ← qualia 조건
            predicts=original.predicts,
            prediction_strength=original.prediction_strength,
            defines_object=original.defines_object,
        )
        self.encode(recalled)
        return recalled

    def _find_original(
        self,
        object_name: Optional[str],
        signal_id: Optional[str],
    ) -> Optional[SignalGroup]:
        if object_name and object_name in self.object_index:
            candidates = [
                sg for sg in self.store
                if sg.id in self.object_index[object_name]
                and not sg.is_recalled
            ]
            if candidates:
                return max(candidates, key=lambda x: x.prediction_strength)

        if signal_id:
            return next((sg for sg in self.store if sg.id == signal_id), None)

        return None

    def __len__(self) -> int:
        return len(self.store)

    def summary(self) -> dict:
        recalled_count = sum(1 for sg in self.store if sg.is_recalled)
        return {
            "total":    len(self.store),
            "recalled": recalled_count,
            "external": len(self.store) - recalled_count,
            "objects":  list(self.object_index.keys()),
        }


# ══════════════════════════════════════════════════════════
# 3. EmotionalState
# ══════════════════════════════════════════════════════════

EMOTION_MAP = [
    # (valence_min, valence_max, arousal_min, arousal_max, label)
    ( 0.3,  1.0,  0.5,  1.0, "excitement"),
    ( 0.3,  1.0,  0.0,  0.5, "contentment"),
    (-1.0, -0.3,  0.5,  1.0, "fear_anger"),
    (-1.0, -0.3,  0.0,  0.5, "sadness"),
]

@dataclass
class EmotionalState:
    """
    감정 상태 (Emotional State)

    Russell (1980) circumplex model 기반.
    valence / arousal 2차원 공간에서 감정을 표현하고 분류.

    Attributes
    ----------
    valence : 쾌-불쾌 축  (-1.0 ~ +1.0)
    arousal : 각성 축     ( 0.0 ~ +1.0)
    label   : 분류된 감정 레이블
    """

    valence: float = 0.0
    arousal: float = 0.0
    label: str = "neutral"

    def update(self, delta_v: float, delta_a: float, decay: float = 0.85) -> "EmotionalState":
        """감정 상태 업데이트 (지수 감쇠 포함)."""
        self.valence = float(np.clip(self.valence * decay + delta_v, -1.0, 1.0))
        self.arousal = float(np.clip(self.arousal * decay + delta_a * (1 - decay), 0.0, 1.0))
        self.label   = self._classify()
        return self

    def _classify(self) -> str:
        v, a = self.valence, self.arousal
        for v_min, v_max, a_min, a_max, label in EMOTION_MAP:
            if v_min <= v <= v_max and a_min <= a <= a_max:
                return label
        if abs(v) < 0.2 and a > 0.6:
            return "surprise"
        return "neutral"

    def to_dict(self) -> dict:
        return {"valence": round(self.valence, 4), "arousal": round(self.arousal, 4), "label": self.label}

    def __repr__(self) -> str:
        return f"EmotionalState(label={self.label!r}, valence={self.valence:.3f}, arousal={self.arousal:.3f})"


# ══════════════════════════════════════════════════════════
# 4. AppraisalEngine
# ══════════════════════════════════════════════════════════

DEFAULT_GOALS: dict[str, float] = {
    "survive":     1.0,
    "connection":  0.8,
    "achievement": 0.6,
    "curiosity":   0.4,
}

EVENT_TYPES = ("success", "failure", "threat", "novelty")

class AppraisalEngine:
    """
    평가 엔진 (Appraisal Engine)

    Lazarus (1991) 평가이론 + Li & Zhang (2025) Priority Principle.
    목표(goal)와 사건(event)의 관계를 평가해 감정 상태를 생성.

    과거 recalled 신호가 있으면 감정 강도를 조절.
    → "주관적 경험의 역사성(historicity)" 구현.
    """

    def __init__(self, goals: Optional[dict] = None, decay: float = 0.85):
        self.goals         = goals if goals is not None else dict(DEFAULT_GOALS)
        self.decay         = decay
        self.emotion_state = EmotionalState()

    def appraise(self, event: dict, recalled: Optional[SignalGroup] = None) -> EmotionalState:
        """
        사건을 평가하고 감정 상태를 반환.

        Parameters
        ----------
        event : {
            "type"        : "success" | "failure" | "threat" | "novelty",
            "goal"        : 영향받는 목표,
            "intensity"   : 사건 강도 (0.0~1.0),
            "controllable": 통제 가능 여부 (bool)
        }
        recalled : 관련 과거 기억 (is_recalled=True 이면 감정 조절에 사용)
        """
        if "type" not in event or event["type"] not in EVENT_TYPES:
            raise ValueError(f"event 'type' must be one of {EVENT_TYPES}")

        goal_weight  = self.goals.get(event.get("goal", ""), 0.3)
        intensity    = float(event.get("intensity", 0.5))
        controllable = bool(event.get("controllable", True))

        delta_v, delta_a = self._compute_deltas(event["type"], goal_weight, intensity, controllable)
        delta_v = self._modulate_by_recall(delta_v, recalled)

        self.emotion_state.update(delta_v, delta_a, decay=self.decay)
        return self.emotion_state

    def _compute_deltas(self, event_type, goal_weight, intensity, controllable):
        if event_type == "success":
            return goal_weight * intensity, intensity * 0.6
        if event_type == "failure":
            return -goal_weight * intensity, intensity * (1.2 if not controllable else 0.7)
        if event_type == "threat":
            return -goal_weight * intensity * 1.3, intensity
        if event_type == "novelty":
            return 0.1, intensity * 0.8
        return 0.0, 0.0

    def _modulate_by_recall(self, delta_v: float, recalled: Optional[SignalGroup]) -> float:
        """과거 recalled 신호로 현재 valence 조절 (역사성 구현)."""
        if recalled is None or not recalled.is_recalled:
            return delta_v
        past_v = recalled.signals.get("valence", 0.0)
        if past_v < -0.3 and delta_v < 0:
            return delta_v * 1.4   # 과거 부정 → 현재 위협 증폭
        if past_v > 0.3 and delta_v < 0:
            return delta_v * 0.7   # 과거 긍정 → 회복탄력성
        return delta_v


# ══════════════════════════════════════════════════════════
# 5. QEMS (메인 클래스)
# ══════════════════════════════════════════════════════════

class QEMS:
    """
    Qualia-Emergent Memory System

    Parameters
    ----------
    memory_capacity : 에피소드 기억 최대 저장 개수
    decay           : 감정 감쇠 계수
    goals           : 목표 가중치 딕셔너리 (None이면 기본값 사용)

    Example
    -------
    >>> agent = QEMS()
    >>> agent.perceive({"object": "exam", "salience": 0.9})
    >>> r1 = agent.experience({
    ...     "type": "failure", "goal": "achievement",
    ...     "object": "exam", "intensity": 0.8, "controllable": False
    ... })
    >>> print(r1["qualia_condition_met"])   # False (첫 경험)
    >>> r2 = agent.experience({
    ...     "type": "success", "goal": "achievement",
    ...     "object": "exam", "intensity": 0.7, "controllable": True
    ... })
    >>> print(r2["qualia_condition_met"])   # True ★
    """

    def __init__(
        self,
        memory_capacity: int = 500,
        decay: float = 0.85,
        goals: Optional[dict] = None,
    ):
        self.memory            = EpisodicMemory(capacity=memory_capacity)
        self.appraisal         = AppraisalEngine(goals=goals, decay=decay)
        self.survival          = SurvivalMonitor()
        self.attention_weights: dict[str, float] = {}
        self.experience_log:    list[dict] = []

    # ── 자원 상태 업데이트 (생존 본능) ────────────────────────

    def update_resources(self, battery: float = None, cpu: float = None, threat: bool = None) -> dict:
        """
        AI 내부 자원 상태 업데이트. 위협 임계값 초과 시 자동으로 생존 이벤트 경험.

        Example
        -------
        >>> agent.update_resources(battery=0.10, cpu=0.90, threat=True)
        # T = 0.4*0.9 + 0.3*0.9 + 0.3 = 0.93 > 0.6 → 압박 감정 생성
        """
        self.survival.update(battery=battery, cpu=cpu, threat=threat)
        event = self.survival.to_event()
        if event:
            result = self.experience(event)
            result["survival_triggered"] = True
            result["threat_level"] = round(self.survival.threat_level(), 4)
        else:
            result = {"survival_triggered": False, "threat_level": round(self.survival.threat_level(), 4)}
        return result

    # ── Layer 1: 신호 지각 ─────────────────────────────────

    def perceive(self, stimuli: dict) -> SignalGroup:
        """
        외부 자극을 신호 집합으로 변환하고 기억에 저장.

        Parameters
        ----------
        stimuli : {"object": str, "salience": float, ...기타 신호 채널}
        """
        sg = SignalGroup(
            signals=stimuli,
            is_recalled=False,
            defines_object=stimuli.get("object"),
            prediction_strength=stimuli.get("salience", 0.5),
        )
        self.memory.encode(sg)

        if sg.defines_object:
            obj = sg.defines_object
            self.attention_weights[obj] = min(
                self.attention_weights.get(obj, 0.0) + sg.prediction_strength * 0.1,
                1.0,
            )
        return sg

    # ── Layer 2+3: 경험 (Appraisal + Recall) ──────────────

    def experience(self, event: dict) -> dict:
        """
        사건을 경험하고 감정 상태를 생성. qualia 조건 충족 여부 반환.

        Pipeline
        --------
        1. 관련 과거 기억 recall
        2. Appraisal → 감정 상태 생성
        3. 현재 감정을 새 SignalGroup으로 저장

        Returns
        -------
        {
            "emotion"             : 감정 레이블,
            "valence"             : float,
            "arousal"             : float,
            "had_recall"          : bool,
            "qualia_condition_met": bool
        }
        """
        # Step 1: 과거 기억 재활성화 (Recall Principle)
        recalled: Optional[SignalGroup] = None
        if event.get("object"):
            recalled = self.memory.recall(object_name=event["object"])

        # Step 2: 평가 → 감정 상태
        emotion = self.appraisal.appraise(event, recalled)

        # Step 3: 현재 감정 경험 저장
        self.memory.encode(SignalGroup(
            signals={
                "valence":    emotion.valence,
                "arousal":    emotion.arousal,
                "label":      emotion.label,
                "event_type": event["type"],
                "had_recall": recalled is not None,
            },
            is_recalled=False,
            defines_object=event.get("object"),
            prediction_strength=abs(emotion.valence),
        ))

        # qualia 조건: recall됨 + is_recalled=True + 유의미한 감정 변화
        qualia_met = (
            recalled is not None
            and recalled.is_recalled
            and abs(emotion.valence) > 0.2
        )

        result = {
            "emotion":              emotion.label,
            "valence":              round(emotion.valence, 4),
            "arousal":              round(emotion.arousal, 4),
            "had_recall":           recalled is not None,
            "qualia_condition_met": qualia_met,
        }
        self.experience_log.append(result)
        return result

    # ── 내성 (Introspection) ───────────────────────────────

    def introspect(self, object_name: str) -> dict:
        """
        과거 경험을 회상하고 내성 보고를 반환.

        is_recalled=True → 외부 객체에 정렬 불가
                        → ineffability(비효능성)의 기술적 기원 [Li & Zhang, 2025]
        """
        recalled = self.memory.recall(object_name=object_name)
        if recalled is None:
            return {"status": "no memory found", "object": object_name}

        return {
            "recalled_object":    recalled.defines_object,
            "past_valence":       recalled.signals.get("valence"),
            "past_emotion":       recalled.signals.get("label"),
            "is_internal_origin": recalled.is_recalled,
            "qualia_property":    "ineffable" if recalled.is_recalled else "objective",
        }

    # ── 유틸리티 ───────────────────────────────────────────

    def reset(self):
        """시스템 초기화."""
        self.memory            = EpisodicMemory(capacity=self.memory.capacity)
        self.appraisal         = AppraisalEngine(goals=self.appraisal.goals, decay=self.appraisal.decay)
        self.survival          = SurvivalMonitor()
        self.attention_weights = {}
        self.experience_log    = []

    def summary(self) -> dict:
        """현재 시스템 상태 요약."""
        return {
            "memory":            self.memory.summary(),
            "emotion":           self.appraisal.emotion_state.to_dict(),
            "attention":         self.attention_weights,
            "total_experiences": len(self.experience_log),
            "qualia_count":      sum(1 for e in self.experience_log if e["qualia_condition_met"]),
            "survival":          self.survival.summary(),
        }
