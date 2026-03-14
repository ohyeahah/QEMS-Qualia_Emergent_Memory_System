"""
signal.py
---------
Li & Zhang (2025)의 '신호 집합(signal group)' 구현.
외부 자극이 아닌 내부적으로 결합된 신호 묶음.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalGroup:
    """
    신호 집합 (Signal Group)

    Attributes
    ----------
    id              : 고유 식별자
    signals         : 신호 딕셔너리  e.g. {"visual": 0.8, "valence": -0.6}
    timestamp       : 생성 시각 (Unix time)
    is_recalled     : True = 내부 recall로 생성됨  →  qualia 조건의 핵심 플래그
                      False = 외부 자극으로 생성됨
    predicts        : 이 신호가 예측하는 객체 이름
    prediction_strength : 예측력 강도 (0.0 ~ 1.0)
    defines_object  : 이 신호가 역으로 정의하는 외부 객체 이름
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
