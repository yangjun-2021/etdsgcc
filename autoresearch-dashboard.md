# Autoresearch Dashboard: sgcc-f1-original

**Runs:** 11 | **Kept:** 1 | **Discarded:** 10 | **Crashed:** 0
**Baseline:** f1: 0.8657 (#6)
**Best:** f1: 0.8657 (#6, +0.00%)

| # | commit | f1 | status | description |
|---|--------|--------|--------|-------------|
| 6 | b22941b | 0.8657 (+0.00%) | keep | baseline on original labels |
| 7 | 7857c74 | 0.8657 (+0.00%) | discard | top_k=100 corr=0.9995 on original labels |
| 8 | 7857c74 | 0.8657 (+0.00%) | discard | add MLP meta-learner on original labels |
| 9 | 7857c74 | 0.8657 (+0.00%) | discard | aggressive corr pruning 0.99 on original labels |
| 10 | c794be1 | 0.8657 (+0.00%) | discard | confident-learning weighted meta-learners on original labels |
| 11 | c3731d9 | 0.8657 (+0.00%) | discard | consensus pseudo-label meta-learners on original labels |
| 12 | 4cb5091 | 0.8657 (+0.00%) | discard | add hard-negative rescue OOF (XGB on FN mask) |
| 13 | 2a5bec5 | 0.8657 (+0.00%) | discard | retrain Expert A on original labels and add its OOF |
| 14 | 8dd654f | 0.8657 (+0.00%) | discard | add co-teaching TCN OOF trained on original labels |
| 15 | e68870b | 0.8657 (+0.00%) | discard | add AMST original-label co-teaching OOF |
| 16 | 30fe81a | 0.8657 (+0.00%) | discard | add strong GBDT prior trained on original labels |
