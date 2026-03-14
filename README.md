# QEMS: Qualia-Emergent Memory System

> A computational framework for implementing functional emotion and qualia in artificial systems — including a survival-drive resource monitor.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-[YOUR_ARXIV_LINK]-red)]()

---

## Overview

QEMS is a three-layer architecture that integrates **appraisal-theoretic emotion generation**, **episodic memory reactivation**, and a **survival-drive resource monitor**, based on the Recall Principle proposed by Li & Zhang (2025).

**Core claims:**
- Signals reactivated with an `is_recalled=True` self-origin tag cannot be aligned to any external referent → technical origin of **ineffability** (qualia)
- Resource threats (low battery, high CPU load, external attack) activate the survival goal (weight 1.0) → **biological stress response** in AI

```
[SurvivalMonitor]  battery / cpu_load / threat_detected
        ↓ (when threat level T > θ)
[Layer 1] Signal Perception
        ↓
[Layer 2] Appraisal Engine + Emotional State (valence / arousal)
        ↓
[Layer 3] Episodic Memory → Recall (is_recalled=True) → modulates Layer 2
                                                          ↑ qualia generated here
```

> **⚠️ Simulation Note**
> The current `SurvivalMonitor` operates on **manually injected values** — `battery_level`, `cpu_load`, and `threat_detected` are passed in by the user, not read from real hardware sensors. This is standard practice for AI framework validation papers. For real-world deployment, connect the monitor to a system library such as `psutil` or an embedded hardware SDK. See [Connecting Real Sensors](#connecting-real-sensors) below.

---

## Repository Structure

```
QEMS/
│
├── qems.py                         # ResourceState, SurvivalMonitor
│                                   # SignalGroup, EpisodicMemory
│                                   # EmotionalState, AppraisalEngine, QEMS
│
├── experiments/
│   ├── exp1_qualia_condition.py    # Experiment 1: qualia_condition_met validation
│   ├── exp2_historicity.py         # Experiment 2: episodic historicity tracking
│   └── exp3_comparison.py          # Experiment 3: QEMS vs. memoryless appraisal
│
├── examples/
│   └── demo.py                     # Quick start demo
│
├── requirements.txt
├── LICENSE
└── README.md
```

> All classes live in a **single `qems.py`** file.

---

## Quick Start

### Installation

```bash
git clone https://github.com/[yourname]/QEMS.git
cd QEMS
pip install -r requirements.txt
```

### Basic Usage

```python
from qems import QEMS

agent = QEMS()

# 1. 외부 자극 지각
agent.perceive({"object": "exam", "salience": 0.9})

# 2. 첫 번째 경험 (과거 기억 없음 → qualia 조건 미충족)
r1 = agent.experience({
    "type": "failure", "goal": "achievement",
    "object": "exam", "intensity": 0.8, "controllable": False
})
print(r1["emotion"])                # "sadness"
print(r1["qualia_condition_met"])   # False

# 3. 두 번째 경험 (과거 기억 recall됨 → qualia 조건 충족)
r2 = agent.experience({
    "type": "success", "goal": "achievement",
    "object": "exam", "intensity": 0.7, "controllable": True
})
print(r2["qualia_condition_met"])   # True ★

# 4. 내성 (introspection)
intro = agent.introspect("exam")
print(intro["qualia_property"])     # "ineffable"
print(intro["is_internal_origin"])  # True
```

### Survival Drive (생존 본능)

```python
from qems import QEMS

agent = QEMS()

# 정상 상태 → 위협 수준 낮음, 감정 변화 없음
r = agent.update_resources(battery=0.9, cpu=0.2, threat=False)
print(r["threat_level"])          # 0.10  (임계값 0.6 미달)
print(r["survival_triggered"])    # False

# 위협 상황 → 압박 감정 자동 생성
r = agent.update_resources(battery=0.10, cpu=0.90, threat=True)
print(r["threat_level"])          # 0.93  (임계값 초과)
print(r["survival_triggered"])    # True
print(r["emotion"])               # "fear_anger" or "sadness"
print(r["valence"])               # 음수 (압박 감정)
```

---

## Key Concepts

| Concept | Implementation |
|---|---|
| Signal Group | `SignalGroup` dataclass — `is_recalled` flag is the qualia key |
| Survival Monitor | `SurvivalMonitor` — T = w_b·(1−battery) + w_c·cpu + w_t·threat |
| Appraisal | Goal-weighted: survive(1.0), connection(0.8), achievement(0.6), curiosity(0.4) |
| Emotional State | Valence/arousal with exponential decay (γ=0.85); 6 emotion labels |
| Qualia Condition | `is_recalled=True` ∧ prior memory exists ∧ \|valence\| > 0.2 |
| Ineffability | Recalled signals have no external referent → cannot be compared across systems |

---

## Survival Threat Formula

```
T = w_b × (1 − battery_level) + w_c × cpu_load + w_t × threat_detected

where: w_b=0.4, w_c=0.3, w_t=0.3,  threshold θ = 0.6

If T > θ:
    δv = −w_survive × T × intensity   (negative valence generated)
```

| Scenario | T | Triggered |
|---|---|---|
| battery=0.9, cpu=0.2, no threat | 0.10 | ✗ |
| battery=0.2, cpu=0.75, no threat | 0.55 | ✗ |
| battery=0.2, cpu=0.75, threat | 0.85 | ✓ |
| battery=0.1, cpu=0.9, threat | 0.93 | ✓ |

---

## Connecting Real Sensors

> The current implementation uses **simulated values** passed manually. To connect real hardware:

```python
# pip install psutil
import psutil
from qems import QEMS

agent = QEMS()

def run_with_real_sensors():
    battery = psutil.sensors_battery()
    agent.update_resources(
        battery=battery.percent / 100 if battery else 1.0,
        cpu=psutil.cpu_percent(interval=1) / 100,
        threat=False  # 외부 공격 감지는 별도 보안 모듈 필요
    )

run_with_real_sensors()
```

For embedded/robotic systems, replace `psutil` with the relevant hardware SDK (e.g., ROS, Arduino serial, Jetson GPIO).

**Why simulated in the paper?**
Scenario-based simulation is standard for AI framework validation. Real-sensor integration is left as future work (see paper §Ⅳ Limitations).

---

## Experiments

```bash
python experiments/exp1_qualia_condition.py   # H1, H2: qualia 조건 검증
python experiments/exp2_historicity.py         # H3: 감정 역사성
python experiments/exp3_comparison.py          # H4: QEMS vs. 기억 없는 시스템
```

---

## Limitations

1. **Hard problem**: Whether `is_recalled=True` generates genuine qualia cannot be philosophically verified.
2. **Linear threat formula**: The current resource monitor uses a simple weighted sum, which may not capture complex resource interactions.
3. **Simulated sensors**: Experiments use manually injected values, not real hardware. Real-world validation requires sensor integration (see [Connecting Real Sensors](#connecting-real-sensors)).

---

## Citation

```bibtex
@article{[yourname]2025qems,
  title   = {QEMS: A Qualia-Emergent Memory System for Emotionally Capable AI —
             Survival Drive, Episodic Memory, and the Technical Origin of Qualia},
  author  = {[Your Name]},
  year    = {2025},
  url     = {https://arxiv.org/abs/[YOUR_ARXIV_ID]}
}
```

---

## References

- Li, F., & Zhang, X. (2025). The Principles of Human-like Conscious Machine. *arXiv:2509.16859*
- Lazarus, R. S. (1991). *Emotion and Adaptation*. Oxford University Press.
- Scherer, K. R. (2001). Appraisal considered as a process of multilevel sequential checking. Oxford University Press.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6).
- Schuller, B., et al. (2026). Affective computing has changed: the foundation model disruption. *npj Artificial Intelligence, 2*, 16.
- Gratch, J. (2022). Emotion recognition is not the same as emotion understanding. *Emotion Researcher*.
- Picard, R. W. (1997). *Affective Computing*. MIT Press.
