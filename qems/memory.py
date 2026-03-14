"""
memory.py
---------
에피소드 기억 시스템.
Teyler & DiScenna (1986) Memory Indexing Theory에 대응.

핵심: recall() 메서드가 호출되면 is_recalled=True 태그가 붙은
      새 SignalGroup이 생성된다 → qualia 생성의 기술적 조건.
"""

import time
from typing import Optional

from qems.signal import SignalGroup


class EpisodicMemory:
    """
    에피소드 기억 저장소

    Parameters
    ----------
    capacity : 최대 저장 개수 (초과 시 가장 오래된 기억 제거)
    """

    def __init__(self, capacity: int = 500):
        self.store: list[SignalGroup] = []
        self.capacity = capacity
        # 인덱스: 객체명 → 관련 signal group id 목록
        self.object_index: dict[str, list[str]] = {}

    # ──────────────────────────────────────────
    # Encode (저장)
    # ──────────────────────────────────────────

    def encode(self, sg: SignalGroup) -> str:
        """새 경험을 기억에 저장. 해마의 인코딩 과정에 대응."""
        if len(self.store) >= self.capacity:
            self._evict_oldest()

        self.store.append(sg)

        if sg.defines_object:
            obj = sg.defines_object
            self.object_index.setdefault(obj, []).append(sg.id)

        return sg.id

    def _evict_oldest(self):
        """가장 오래된 기억 제거 (FIFO)."""
        if not self.store:
            return
        old = self.store.pop(0)
        if old.defines_object and old.defines_object in self.object_index:
            ids = self.object_index[old.defines_object]
            if old.id in ids:
                ids.remove(old.id)

    # ──────────────────────────────────────────
    # Recall (재활성화)
    # ──────────────────────────────────────────

    def recall(
        self,
        object_name: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> Optional[SignalGroup]:
        """
        과거 신호를 재활성화.

        재활성화된 신호에는 is_recalled=True 태그가 붙는다.
        → 이 태그가 붙은 신호는 외부 객체에 정렬될 수 없음
        → ineffability(비효능성)의 기술적 기원  [Li & Zhang, 2025]

        Returns
        -------
        recalled : is_recalled=True 인 새 SignalGroup, 또는 None
        """
        original = self._find_original(object_name, signal_id)
        if original is None:
            return None

        # ★ 핵심: 원본을 복사하되 is_recalled=True 로 설정
        recalled = SignalGroup(
            signals=original.signals.copy(),
            timestamp=time.time(),
            is_recalled=True,            # ← qualia 조건
            predicts=original.predicts,
            prediction_strength=original.prediction_strength,
            defines_object=original.defines_object,
        )

        # 재활성화된 것도 새 경험으로 저장
        self.encode(recalled)
        return recalled

    def _find_original(
        self,
        object_name: Optional[str],
        signal_id: Optional[str],
    ) -> Optional[SignalGroup]:
        """원본 SignalGroup 검색 (외부 자극으로 생성된 것만)."""
        if object_name and object_name in self.object_index:
            candidates = [
                sg for sg in self.store
                if sg.id in self.object_index[object_name]
                and not sg.is_recalled          # 외부 자극 원본만 후보
            ]
            if candidates:
                # 예측력이 가장 강한 기억 선택
                return max(candidates, key=lambda x: x.prediction_strength)

        if signal_id:
            return next(
                (sg for sg in self.store if sg.id == signal_id), None
            )

        return None

    # ──────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.store)

    def summary(self) -> dict:
        """기억 통계 요약."""
        recalled_count = sum(1 for sg in self.store if sg.is_recalled)
        return {
            "total": len(self.store),
            "recalled": recalled_count,
            "external": len(self.store) - recalled_count,
            "objects": list(self.object_index.keys()),
        }
