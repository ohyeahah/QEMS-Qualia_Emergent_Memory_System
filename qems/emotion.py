"""
emotion.py
----------
Russell (1980) circumplex model 기반 감정 상태.
valence / arousal 2차원 공간에서 감정을 표현하고 분류한다.
"""

import numpy as np
from dataclasses import dataclass


# 감정 레이블 분류 기준 (valence × arousal)
EMOTION_MAP = [
    # (valence_min, valence_max, arousal_min, arousal_max, label)
    ( 0.3,  1.0,  0.5,  1.0, "excitement"),   # 높은 쾌감 + 높은 각성
    ( 0.3,  1.0,  0.0,  0.5, "contentment"),  # 높은 쾌감 + 낮은 각성
    (-1.0, -0.3,  0.5,  1.0, "fear_anger"),   # 낮은 쾌감 + 높은 각성
    (-1.0, -0.3,  0.0,  0.5, "sadness"),      # 낮은 쾌감 + 낮은 각성
]


@dataclass
class EmotionalState:
    """
    감정 상태 (Emotional State)

    Attributes
    ----------
    valence : 쾌-불쾌 축  (-1.0 ~ +1.0)
    arousal : 각성 축     ( 0.0 ~ +1.0)
    label   : 분류된 감정 레이블
    """

    valence: float = 0.0
    arousal: float = 0.0
    label: str = "neutral"

    def update(
        self,
        delta_v: float,
        delta_a: float,
        decay: float = 0.85,
    ) -> "EmotionalState":
        """
        감정 상태 업데이트 (지수 감쇠 포함).

        Parameters
        ----------
        delta_v : valence 변화량
        delta_a : arousal 변화량
        decay   : 이전 상태가 유지되는 비율 (0~1)
        """
        self.valence = float(np.clip(
            self.valence * decay + delta_v,
            -1.0, 1.0
        ))
        self.arousal = float(np.clip(
            self.arousal * decay + delta_a * (1 - decay),
            0.0, 1.0
        ))
        self.label = self._classify()
        return self

    def _classify(self) -> str:
        """valence × arousal → 감정 레이블."""
        v, a = self.valence, self.arousal
        for v_min, v_max, a_min, a_max, label in EMOTION_MAP:
            if v_min <= v <= v_max and a_min <= a <= a_max:
                return label
        if abs(v) < 0.2 and a > 0.6:
            return "surprise"
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "label":   self.label,
        }

    def __repr__(self) -> str:
        return (
            f"EmotionalState(label={self.label!r}, "
            f"valence={self.valence:.3f}, arousal={self.arousal:.3f})"
        )
