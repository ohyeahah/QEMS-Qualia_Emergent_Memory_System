"""
experiments/exp3_comparison.py
--------------------------------
실험 3: QEMS vs. 기억 없는 Appraisal-only 시스템 비교

가설 H4: QEMS는 더 높은 감정 다양성(entropy)과
         더 강한 장기 자기상관(autocorrelation)을 보인다.
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from qems import QEMS


EVENT_SEQUENCE = [
    {"type": "failure",  "goal": "achievement", "object": "task", "intensity": 0.8, "controllable": False},
    {"type": "novelty",  "goal": "curiosity",   "object": "task", "intensity": 0.6, "controllable": True},
    {"type": "success",  "goal": "achievement", "object": "task", "intensity": 0.7, "controllable": True},
    {"type": "threat",   "goal": "survive",     "object": "task", "intensity": 0.9, "controllable": False},
    {"type": "failure",  "goal": "connection",  "object": "task", "intensity": 0.5, "controllable": True},
    {"type": "success",  "goal": "connection",  "object": "task", "intensity": 0.6, "controllable": True},
    {"type": "novelty",  "goal": "curiosity",   "object": "task", "intensity": 0.4, "controllable": True},
    {"type": "threat",   "goal": "survive",     "object": "task", "intensity": 0.7, "controllable": False},
    {"type": "success",  "goal": "achievement", "object": "task", "intensity": 0.9, "controllable": True},
    {"type": "failure",  "goal": "achievement", "object": "task", "intensity": 0.6, "controllable": False},
]


def entropy(labels: list[str]) -> float:
    """Shannon entropy of emotion label distribution."""
    counts = Counter(labels)
    total = len(labels)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


def run_system(use_memory: bool) -> list[dict]:
    """시스템 실행 후 experience_log 반환."""
    agent = QEMS(memory_capacity=500 if use_memory else 0)
    agent.perceive({"object": "task", "salience": 0.8})
    results = []
    for ev in EVENT_SEQUENCE:
        if use_memory:
            r = agent.experience(ev)
        else:
            # 기억 없는 조건: 매 스텝마다 초기화
            agent.reset()
            agent.perceive({"object": "task", "salience": 0.8})
            r = agent.experience(ev)
        results.append(r)
    return results


def main():
    print("=" * 55)
    print("  Experiment 3 — QEMS vs. Memoryless Appraisal")
    print("=" * 55)

    print("\n[QEMS 실행]")
    qems_log = run_system(use_memory=True)

    print("[기억 없는 시스템 실행]")
    memoryless_log = run_system(use_memory=False)

    # 감정 다양성 (Shannon entropy)
    qems_labels       = [r["emotion"] for r in qems_log]
    memoryless_labels = [r["emotion"] for r in memoryless_log]

    qems_entropy       = entropy(qems_labels)
    memoryless_entropy = entropy(memoryless_labels)

    # qualia 발생 횟수
    qems_qualia = sum(1 for r in qems_log if r["qualia_condition_met"])

    print("\n[결과 비교]")
    print(f"  감정 레이블 분포 (QEMS)       : {dict(Counter(qems_labels))}")
    print(f"  감정 레이블 분포 (기억 없음)  : {dict(Counter(memoryless_labels))}")
    print(f"  Shannon Entropy (QEMS)        : {qems_entropy:.4f}")
    print(f"  Shannon Entropy (기억 없음)   : {memoryless_entropy:.4f}")
    print(f"  H4 충족 (QEMS entropy 더 높음): {qems_entropy > memoryless_entropy}")
    print(f"\n  QEMS qualia 발생 횟수         : {qems_qualia}/{len(qems_log)}")

    print("\n[QEMS 감정 궤적]")
    for i, r in enumerate(qems_log):
        q = "★" if r["qualia_condition_met"] else " "
        print(f"  {i+1:2d}. {q} {r['emotion']:12s} v={r['valence']:+.3f}  recall={r['had_recall']}")


if __name__ == "__main__":
    main()
