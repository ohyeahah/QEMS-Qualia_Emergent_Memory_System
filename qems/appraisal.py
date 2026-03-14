"""
appraisal.py
------------
Lazarus (1991) 평가이론 + Li & Zhang (2025) Priority Principle 구현.
목표(goal)와 사건(event)의 관계를 평가해 감정 상태를 생성한다.
"""

from typing import Optional

from qems.signal import SignalGroup
from qems.emotion import EmotionalState


# 목표 중요도 가중치 (Priority Principle)
DEFAULT_GOALS: dict[str, float] = {
    "survive":     1.0,   # 생존
    "connection":  0.8,   # 사회적 연결
    "achievement": 0.6,   # 성취
    "curiosity":   0.4,   # 탐구
}

# 지원되는 사건 유형
EVENT_TYPES = ("success", "failure", "threat", "novelty")


class AppraisalEngine:
    """
    평가 엔진 (Appraisal Engine)

    사건을 평가해 valence / arousal 변화량을 계산하고
    EmotionalState를 업데이트한다.

    과거 recalled 신호가 있으면 감정 강도를 조절한다.
    → 이것이 "주관적 경험의 역사성(historicity)" 구현.

    Parameters
    ----------
    goals : 목표 가중치 딕셔너리. None이면 기본값 사용.
    decay : 감정 감쇠 계수 (0~1)
    """

    def __init__(
        self,
        goals: Optional[dict[str, float]] = None,
        decay: float = 0.85,
    ):
        self.goals = goals if goals is not None else dict(DEFAULT_GOALS)
        self.decay = decay
        self.emotion_state = EmotionalState()

    # ──────────────────────────────────────────
    # 핵심 메서드
    # ──────────────────────────────────────────

    def appraise(
        self,
        event: dict,
        recalled: Optional[SignalGroup] = None,
    ) -> EmotionalState:
        """
        사건을 평가하고 감정 상태를 반환한다.

        Parameters
        ----------
        event : {
            "type"        : "success" | "failure" | "threat" | "novelty",
            "goal"        : 영향받는 목표 이름 (DEFAULT_GOALS 키),
            "intensity"   : 사건 강도 (0.0 ~ 1.0),
            "controllable": 통제 가능 여부 (bool)
        }
        recalled : 관련 과거 기억 (is_recalled=True 이면 감정 조절에 사용)

        Returns
        -------
        EmotionalState : 업데이트된 현재 감정 상태
        """
        self._validate_event(event)

        goal_weight  = self.goals.get(event.get("goal", ""), 0.3)
        intensity    = float(event.get("intensity", 0.5))
        controllable = bool(event.get("controllable", True))

        delta_v, delta_a = self._compute_deltas(
            event["type"], goal_weight, intensity, controllable
        )

        # ★ 과거 recalled 신호로 감정 강도 조절 (역사성 구현)
        delta_v = self._modulate_by_recall(delta_v, recalled)

        self.emotion_state.update(delta_v, delta_a, decay=self.decay)
        return self.emotion_state

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    def _compute_deltas(
        self,
        event_type: str,
        goal_weight: float,
        intensity: float,
        controllable: bool,
    ) -> tuple[float, float]:
        """사건 유형별 valence / arousal 변화량 계산."""
        if event_type == "success":
            return (
                goal_weight * intensity,
                intensity * 0.6,
            )
        if event_type == "failure":
            return (
                -goal_weight * intensity,
                intensity * (1.2 if not controllable else 0.7),
            )
        if event_type == "threat":
            return (
                -goal_weight * intensity * 1.3,
                intensity,
            )
        if event_type == "novelty":
            return (0.1, intensity * 0.8)

        return (0.0, 0.0)

    def _modulate_by_recall(
        self,
        delta_v: float,
        recalled: Optional[SignalGroup],
    ) -> float:
        """
        과거 recalled 신호가 현재 valence 변화량을 조절.

        - 과거 부정 경험 + 현재 부정 자극 → 위협 증폭 (×1.4)
        - 과거 긍정 경험 + 현재 부정 자극 → 회복탄력성 (×0.7)
        """
        if recalled is None or not recalled.is_recalled:
            return delta_v

        past_valence = recalled.signals.get("valence", 0.0)

        if past_valence < -0.3 and delta_v < 0:
            # 과거 부정 경험 → 현재 위협 더 강하게 느낌
            return delta_v * 1.4

        if past_valence > 0.3 and delta_v < 0:
            # 과거 긍정 경험 → 현재 부정 자극 완충
            return delta_v * 0.7

        return delta_v

    @staticmethod
    def _validate_event(event: dict):
        """사건 딕셔너리 유효성 검사."""
        if "type" not in event:
            raise ValueError("event must have a 'type' key.")
        if event["type"] not in EVENT_TYPES:
            raise ValueError(
                f"event 'type' must be one of {EVENT_TYPES}, "
                f"got {event['type']!r}"
            )
