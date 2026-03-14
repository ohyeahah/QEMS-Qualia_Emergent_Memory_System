"""
examples/demo.py
----------------
QEMS 빠른 실행 예제.
VS Code 터미널에서: python examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qems import QEMS


def main():
    print("=" * 55)
    print("  QEMS — Qualia-Emergent Memory System  Demo")
    print("=" * 55)

    agent = QEMS()

    # ── 1. 첫 번째 경험: 시험 실패 ──
    print("\n[Step 1] 시험 실패 경험 (과거 기억 없음)")
    agent.perceive({"object": "exam", "salience": 0.9})
    r1 = agent.experience({
        "type":         "failure",
        "goal":         "achievement",
        "object":       "exam",
        "intensity":    0.8,
        "controllable": False,
    })
    print(f"  감정    : {r1['emotion']}")
    print(f"  valence : {r1['valence']}")
    print(f"  qualia? : {r1['qualia_condition_met']}")   # → False

    # ── 2. 두 번째 경험: 같은 시험, 이번엔 성공 ──
    print("\n[Step 2] 시험 성공 경험 (과거 실패 기억 recall됨)")
    r2 = agent.experience({
        "type":         "success",
        "goal":         "achievement",
        "object":       "exam",
        "intensity":    0.7,
        "controllable": True,
    })
    print(f"  감정    : {r2['emotion']}")
    print(f"  valence : {r2['valence']}")
    print(f"  recall? : {r2['had_recall']}")
    print(f"  qualia? : {r2['qualia_condition_met']}")   # → True ★

    # ── 3. 내성 (introspection) ──
    print("\n[Step 3] 내성 — 시험에 대한 회상")
    intro = agent.introspect("exam")
    for k, v in intro.items():
        print(f"  {k}: {v}")
    # is_internal_origin=True, qualia_property='ineffable' ★

    # ── 4. 시스템 요약 ──
    print("\n[Summary]")
    s = agent.summary()
    print(f"  총 경험 횟수  : {s['total_experiences']}")
    print(f"  qualia 발생   : {s['qualia_count']}")
    print(f"  기억 저장 수  : {s['memory']['total']}")
    print(f"  현재 감정     : {s['emotion']['label']}")


if __name__ == "__main__":
    main()
