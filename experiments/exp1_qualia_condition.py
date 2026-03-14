"""
experiments/exp1_qualia_condition.py
-------------------------------------
실험 1: Qualia 조건 충족 검증

가설 H1: 에피소드 기억이 있는 조건에서만 qualia_condition_met=True
가설 H2: 과거 부정 기억이 있으면 |Δvalence|가 유의미하게 더 큼
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qems import QEMS


def run_trial(label: str, has_prior: bool) -> dict:
    """단일 트라이얼 실행."""
    agent = QEMS()

    if has_prior: 
        agent.perceive({"object": "exam", "salience": 0.9})
        agent.experience({
            "type": "failure", "goal": "achievement",
            "object": "exam", "intensity": 0.8, "controllable": False,
        })

    # 테스트 사건: 위협
    result = agent.experience({
        "type": "threat", "goal": "survive",
        "object": "exam", "intensity": 0.7, "controllable": False,
    })

    print(f"\n  [{label}]")
    print(f"    qualia_condition_met : {result['qualia_condition_met']}")
    print(f"    had_recall           : {result['had_recall']}")
    print(f"    valence              : {result['valence']}")
    print(f"    emotion              : {result['emotion']}")
    return result


def main():
    print("=" * 55)
    print("  Experiment 1 — Qualia Condition Validation")
    print("=" * 55)

    r_no_mem  = run_trial("기억 없음  (Trial 1)", has_prior=False)
    r_has_mem = run_trial("기억 있음  (Trial N)", has_prior=True)

    print("\n[결과 비교]")
    print(f"  H1 검증: qualia(기억無)={r_no_mem['qualia_condition_met']}, "
          f"qualia(기억有)={r_has_mem['qualia_condition_met']}")

    delta_no  = abs(r_no_mem["valence"])
    delta_has = abs(r_has_mem["valence"])
    print(f"  H2 검증: |valence|(기억無)={delta_no:.4f}, "
          f"|valence|(기억有)={delta_has:.4f}")
    print(f"  H2 충족: {delta_has > delta_no}")


if __name__ == "__main__":
    main()
