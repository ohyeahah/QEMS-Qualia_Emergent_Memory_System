# QEMS: Qualia-Emergent Memory System

> A computational framework for implementing functional emotion and qualia in artificial systems.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-[YOUR_ARXIV_LINK]-red)]()

---

## Overview

QEMS is a three-layer architecture that integrates **appraisal-theoretic emotion generation** with an **episodic memory reactivation** mechanism, based on the Recall Principle proposed by Li & Zhang (2025).

The core claim: signals reactivated with a `is_recalled=True` self-origin tag cannot be aligned to any external referent — this is the technical origin of **ineffability**, the defining property of qualia.

```
[Layer 1] Signal Perception
      ↓
[Layer 2] Appraisal Engine + Emotional State (valence / arousal)
      ↓
[Layer 3] Episodic Memory → Recall → modulates Layer 2  ← qualia generated here
```

---

## Repository Structure

```
QEMS/
│
├── qems/
│   ├── __init__.py
│   ├── signal.py          # SignalGroup dataclass
│   ├── memory.py          # EpisodicMemory (store + recall)
│   ├── appraisal.py       # AppraisalEngine (goal-weighted evaluation)
│   ├── emotion.py         # EmotionalState (valence/arousal + decay)
│   └── qems.py            # QEMS main class (perceive / experience / introspect)
│
├── experiments/
│   ├── exp1_qualia_condition.py    # Experiment 1: qualia_condition_met validation
│   ├── exp2_historicity.py         # Experiment 2: episodic historicity tracking
│   └── exp3_comparison.py          # Experiment 3: QEMS vs. memoryless appraisal
│
├── examples/
│   └── demo.py            # Quick start demo
│
├── requirements.txt
├── LICENSE
└── README.md
```

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
from qems.qems import QEMS

# Initialize system
system = QEMS(memory_capacity=100)

# Perceive a stimulus
system.perceive({"visual": 0.8, "audio": 0.5})

# Experience an event
result = system.experience({
    "type": "exam_failure",
    "goal_impact": {"achievement": -0.9, "connection": -0.2}
})

print(result["emotion"])           # e.g. "sadness"
print(result["qualia_condition_met"])  # False (first experience, no prior memory)

# Experience again — qualia condition now met
result2 = system.experience({
    "type": "exam_success",
    "goal_impact": {"achievement": 0.9}
})

print(result2["qualia_condition_met"])  # True (past memory modulates current state)

# Introspect
introspection = system.introspect("exam_failure")
print(introspection["qualia_property"])    # "ineffable"
print(introspection["is_internal_origin"]) # True
```

---

## Key Concepts

| Concept | Implementation |
|---|---|
| Signal Group | `SignalGroup` dataclass with `is_recalled` flag |
| Appraisal | Goal-weighted evaluation: survive(1.0), connection(0.8), achievement(0.6), curiosity(0.4) |
| Emotional State | Valence/arousal with exponential decay; maps to 6 emotion labels |
| Qualia Condition | `is_recalled=True` + modulation of current state |
| Ineffability | Recalled signals cannot be aligned to external objects |

---

## Experiments

Run each experiment independently:

```bash
python experiments/exp1_qualia_condition.py
python experiments/exp2_historicity.py
python experiments/exp3_comparison.py
```

---

## Citation

If you use QEMS in your research, please cite:

```bibtex
@article{[yourname]2025qems,
  title   = {Toward Emotionally Capable AI: A Technical Framework for Implementing Functional Emotion in Artificial Systems},
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
