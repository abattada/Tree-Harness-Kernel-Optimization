# STATE — addmm
status: in_progress
# best / budget 不在這裡手寫(會 drift):看 `python -m harness.task addmm` 任務卡,
# 由 journal 自動算。best.py 用 `python -m harness.task addmm --export-best` 還原。

## tried(每行一筆,新的加在最上面;格式: k<N> | <strategy> | <speedup 或 error 根因>)
(none yet)

## pitfalls(本 op 踩過的環境/API 坑,下一個 agent 不要再踩)
- Triton 3.6 沒有 tl.tanh / tl.math.tanh — 用 `from triton.language.extra
  import libdevice; libdevice.tanh(x)` 或 2*tl.sigmoid(2*x)-1

## next(給下一個 agent 的建議,按優先序)
1. 讀任務卡與 KB 範例,寫第一版 seed kernel
