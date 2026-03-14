"""
qems.py
-------
QEMS (Qualia-Emergent Memory System) — 전체 통합 클래스.

Li & Zhang (2025) 4원칙 구현:
  Prediction Principle  → signal grouping (perceive)
  Priority Principle    → appraisal engine
  Exploration Principle → attention weighting
  Recall Principle      → episodic memory + is_recalled tag → qualia
"""

from typing import Optional

from qems.signal import SignalGroup
from qems.memory import EpisodicMemory
from qems.appraisal import AppraisalEngine


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
    >>> result = agent.experience({
    ...     "type": "failure", "goal": "achievement",
    ...     "object": "exam", "intensity": 0.8, "controllable": False
    ... })
    >>> print(result["qualia_condition_met"])   # False (첫 경험)
    >>> result2 = agent.experience({
    ...     "type": "success", "goal": "achievement",
    ...     "object": "exam", "intensity": 0.7, "controllable": True
    ... })
    >>> print(result2["qualia_condition_met"])  # True (과거 기억 recall됨)
    """

    def __init__(
        self,
        memory_capacity: int = 500,
        decay: float = 0.85,
        goals: Optional[dict[str, float]] = None,
    ):
        self.memory    = EpisodicMemory(capacity=memory_capacity)
        self.appraisal = AppraisalEngine(goals=goals, decay=decay)
        self.attention_weights: dict[str, float] = {}
        self.experience_log: list[dict] = []

    # ──────────────────────────────────────────
    # Layer 1: 신호 지각 (Signal Perception)
    # ──────────────────────────────────────────

    def perceive(self, stimuli: dict) -> SignalGroup:
        """
        외부 자극을 신호 집합으로 변환하고 기억에 저장.

        Parameters
        ----------
        stimuli : {
            "object"   : 자극 객체 이름 (선택),
            "salience" : 현출성 (0.0~1.0, 기본 0.5),
            ...        : 임의의 신호 채널 값
        }

        Returns
        -------
        SignalGroup : is_recalled=False 인 새 신호 집합
        """
        sg = SignalGroup(
            signals=stimuli,
            is_recalled=False,
            defines_object=stimuli.get("object"),
            prediction_strength=stimuli.get("salience", 0.5),
        )
        self.memory.encode(sg)

        # 주의(attention) 가중치 업데이트 (Exploration Principle)
        if sg.defines_object:
            obj = sg.defines_object
            self.attention_weights[obj] = min(
                self.attention_weights.get(obj, 0.0)
                + sg.prediction_strength * 0.1,
                1.0,
            )

        return sg

    # ──────────────────────────────────────────
    # Layer 2 + 3: 경험 (Appraisal + Recall)
    # ──────────────────────────────────────────

    def experience(self, event: dict) -> dict:
        """
        사건을 경험하고 감정 상태를 생성. qualia 조건 충족 여부를 반환.

        Pipeline
        --------
        1. 관련 과거 기억 recall
        2. Appraisal → 감정 상태 생성
        3. 현재 감정을 새 SignalGroup으로 저장 (미래 recall 대비)

        Parameters
        ----------
        event : {
            "type"        : "success" | "failure" | "threat" | "novelty",
            "goal"        : 영향받는 목표,
            "object"      : 관련 객체 이름 (recall에 사용),
            "intensity"   : 사건 강도 (0.0~1.0),
            "controllable": 통제 가능 여부 (bool)
        }

        Returns
        -------
        dict : {
            "emotion"             : 감정 레이블,
            "valence"             : float,
            "arousal"             : float,
            "had_recall"          : bool (과거 기억 recall 여부),
            "qualia_condition_met": bool (qualia 조건 충족 여부)
        }
        """
        # Step 1: 관련 과거 기억 재활성화 (Recall Principle)
        recalled: Optional[SignalGroup] = None
        if event.get("object"):
            recalled = self.memory.recall(object_name=event["object"])

        # Step 2: 평가 → 감정 상태 생성
        emotion = self.appraisal.appraise(event, recalled)

        # Step 3: 현재 감정 경험을 새 신호로 저장
        experience_sg = SignalGroup(
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
        )
        self.memory.encode(experience_sg)

        # qualia 조건:
        #   (1) 과거 기억이 recall됨
        #   (2) recall된 신호에 is_recalled=True 태그가 있음
        #   (3) 현재 감정 변화가 유의미함 (|valence| > 0.2)
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

    # ──────────────────────────────────────────
    # 내성 (Introspection)
    # ──────────────────────────────────────────

    def introspect(self, object_name: str) -> dict:
        """
        과거 경험을 회상하고 내성 보고를 반환.

        is_recalled=True → 외부 객체에 정렬 불가
                        → ineffability(비효능성)의 기술적 기원
                           [Li & Zhang, 2025]

        Parameters
        ----------
        object_name : 회상할 객체 이름

        Returns
        -------
        dict : {
            "recalled_object"  : str,
            "past_valence"     : float,
            "past_emotion"     : str,
            "is_internal_origin": bool,
            "qualia_property"  : "ineffable" | "objective"
        }
        """
        recalled = self.memory.recall(object_name=object_name)
        if recalled is None:
            return {"status": "no memory found", "object": object_name}

        return {
            "recalled_object":   recalled.defines_object,
            "past_valence":      recalled.signals.get("valence"),
            "past_emotion":      recalled.signals.get("label"),
            # is_recalled=True → 내부에서 온 신호 → 외부 설명 불가
            "is_internal_origin": recalled.is_recalled,
            "qualia_property": (
                "ineffable" if recalled.is_recalled else "objective"
            ),
        }

    # ──────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────

    def reset(self):
        """시스템 초기화."""
        self.memory = EpisodicMemory(capacity=self.memory.capacity)
        self.appraisal = AppraisalEngine(
            goals=self.appraisal.goals,
            decay=self.appraisal.decay,
        )
        self.attention_weights.clear()
        self.experience_log.clear()

    def summary(self) -> dict:
        """현재 시스템 상태 요약."""
        return {
            "memory":           self.memory.summary(),
            "emotion":          self.appraisal.emotion_state.to_dict(),
            "attention":        self.attention_weights,
            "total_experiences": len(self.experience_log),
            "qualia_count": sum(
                1 for e in self.experience_log
                if e["qualia_condition_met"]
            ),
        }
