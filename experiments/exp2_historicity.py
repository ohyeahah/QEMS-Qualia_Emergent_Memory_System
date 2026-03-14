"""
experiments/exp2_historicity.py
--------------------------------
실험 2: 감정 역사성 검증 (종단 에피소드 추적)

가설 H3: 연속 실패 후 성공 사건의 valence 반응이
         실패 이력 없는 조건보다 더 높다 (회복탄력성 효과).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qems import QEMS


FAILURE_SEQ = [
    {"type": "failure", "goal": "achievement",
     "object": "project", "intensity": 0.7, "controllable": False},
    {"type": "failure", "goal": "achievement",
     "object": "project", "intensity": 0.8, "controllable": False},
    {"type": "failure", "goal": "connection",
     "object": "project", "intensity": 0.5, "controllable": True},
]

SUCCESS_EVENT = {
    "type": "success", "goal": "achievement",
    "object": "project", "intensity": 0.7, "controllable": True,
}


def run_scenario(label: str, include_failures: bool) -> float:
    agent = QEMS()
    agent.perceive({"object": "project", "salience": 0.8})

    if include_failures:
        for ev in FAILURE_SEQ:
            r = agent.experience(ev)
            print(f"    실패: emotion={r['emotion']}, "
                  f"valence={r['valence']:.4f}")

    result = agent.experience(SUCCESS_EVENT)
    print(f"\n  [{label}] 성공 후 valence={result['valence']:.4f}, "
          f"emotion={result['emotion']}, qualia={result['qualia_condition_met']}")
    return result["valence"]


def main():
    print("=" * 55)
    print("  Experiment 2 — Episodic Historicity")
    print("=" * 55)

    print("\n[조건 A] 실패 이력 없음")
    v_no_fail = run_scenario("실패 이력 없음", include_failures=False)

    print("\n[조건 B] 연속 실패 → 성공")
    v_after_fail = run_scenario("연속 실패 후 성공", include_failures=True)

    print("\n[결과]")
    print(f"  성공 valence (이력 없음)   : {v_no_fail:.4f}")
    print(f"  성공 valence (실패 후)     : {v_after_fail:.4f}")
    print(f"  H3 충족 (이력 있을 때 더 높음): {v_after_fail > v_no_fail}")


if __name__ == "__main__":
    main()
