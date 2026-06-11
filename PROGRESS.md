# 📊 Project Progress — dPickleBall Bot

> **Single-page command center** for the team. Track overall progress at a glance.

---

## 📈 Overall Timeline

```
Phase 0: Repo Setup           ████████████████████████████████ 100%
Code Infrastructure (F1-F6)   ████████████████████████████████ 100%
Bug Fixes                     ████████████████████████████████ 100%  (4 found, 4 fixed)
────────── Team Experiments Start ──────────
Member 1: Architecture         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Member 2: Reward               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Member 3: Observation          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Member 4: Hyperparameters      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Member 5: Training Strategy    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Final Combined Training Run    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Competition Submission         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
```

---

## 🧑‍🤝‍🧑 Member Status

| Area | Member | Branch | Status | Best Win Rate (right) | Best Win Rate (left) | Current Opponent | Blockers |
|------|--------|--------|--------|----------------------|---------------------|------------------|----------|
| Architecture | **Member 1** | `team/arch-variants` | ⏳ Pending | — | — | — | B1 smoke test needed |
| Reward | **Member 2** | `team/reward-shaping` | ⏳ Pending | — | — | — | Repo access + B1 |
| Observation | **Member 3** | `team/obs-variants` | ⏳ Pending | — | — | — | Repo access + B1 |
| Hyperparameters | **Member 4** | `team/hyperparams` | ⏳ Pending | — | — | — | Repo access + B1 |
| Training Strategy | **Member 5** | `team/training-strategy` | ⏳ Pending | — | — | — | Repo access + B1 |

**🟢 Active** · **🟡 In Progress** · **⏳ Pending** · **🔴 Blocked** · **✅ Complete**

---

## 🚦 Critical Path

```
B1 Smoke Test (10k steps) ──gate──▶ All 5 Members Train in Parallel
                                       │
                                       ▼
                         Update RESULTS.md with findings
                                       │
                                       ▼
                         Team Discussion → Pick Winners
                                       │
                                       ▼
                         Final Combined Training (300k steps)
                                       │
                                       ▼
                         Competition Submission Package
```

### Current Gate: B1 (10k Smoke Test)

| Task | Status | Est. Time |
|------|--------|-----------|
| B1: Run 10k steps with left/right wrapper | ⏳ Pending | 5-10 min |
| Validate no crashes, shape match, curriculum | ⏳ Pending | — |

---

## ⚠️ Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | B1 reveals pipeline crash in Unity | Medium | High | Fix before member experiments start |
| 2 | Merge conflicts when members PR | Medium | Medium | Each owns separate files, no overlap |
| 3 | Member's machine can't run Unity | Low | High | Share checkpoints via Drive |
| 4 | Experiments produce incomparable results | Low | Medium | Fixed eval harness tests both sides |
| 5 | Final combined run underperforms | Medium | Medium | Use members' winning configs, do grid search |

---

## 📅 Milestones & Deadlines

| Milestone | Target Date | Status |
|-----------|------------|--------|
| All 5 members have first results in RESULTS.md | TBD | ⏳ |
| Team discussion + winning configs | TBD | ⏳ |
| Final combined training run | TBD | ⏳ |
| Competition submission | TBD | ⏳ |
