# HARNESS.md — Claude agent 模式操作合約

這份文件是給 agentic LLM(如 Claude Code)在 pro_max harness 內**自主迭代優化 Triton kernel** 的規則。驗證與記錄由 harness 程式負責;你負責寫 kernel、讀回饋、迭代。

## QUICKSTART(30 秒版)

```bash
python -m harness.task --triage         # 總表:每個 op 的建議預算/目前最佳/狀態
                                        # → 決定這個 session 優化誰
python -m harness.task <op> --init      # 任務卡 + 建立/接續 workspace
# 讀 STATE.md(若有歷史)與任務卡列出的 kb/ 範例
# 寫 runs/agent_<op>/attempts/k<N>.py
python -m py_compile runs/agent_<op>/attempts/k<N>.py        # 免費 precheck
CUDA_VISIBLE_DEVICES=<gpu> python -m harness.eval_one --operator <op> \
    --candidate runs/agent_<op>/attempts/k<N>.py \
    --journal runs/agent_<op>/journal.jsonl --arm agent_<op> \
    --strategy autotune_grid                                   # 耗預算 1 次
# 迭代到預算盡/飽和 → python -m harness.task <op> --export-best、更新 STATE.md(收尾規範見下)
```

## 環境

```bash
conda activate pro_max        # torch cu128 + triton 3.6 + 此 repo 依賴
cd ~/para_final/pro_max
# GPU:nvidia-smi 查看;用 CUDA_VISIBLE_DEVICES 綁卡(每次驗證獨佔一張)
# 多 op 用 `&` 同時開多個 eval_one 時,先 `export EVAL_STARTUP_JITTER_S=2`,
# 錯開各進程的 CUDA-init/Triton 編譯,避免瞬間齊發把驅動打爆(對計時無影響)。
```

## 任務定義(從任務卡開始,不要自己翻 specs.py)

```bash
python -m harness.task <operator>     # 一次印出你需要的全部 context:
```

任務卡包含:簽名/shapes/dtypes、tolerance、PyTorch reference 原始碼、禁用
pattern、roofline 目標、KB 成功案例清單、workspace 現況(已用預算/最佳紀錄/
STATE.md 是否存在)、可直接複製的驗證指令。

目標:寫一個 Python module 實作 `triton_run(*inputs) -> torch.Tensor`,
輸出與 reference 在 tolerance 內一致,且越快越好。

## Kernel 合約(違反即自動 fail)

1. module 必須定義 `triton_run(*inputs)`,簽名順序照 spec 的 `signature_doc`。
2. 核心計算必須用你寫的 Triton kernel。直接呼叫同名 torch op(如 softmax 任務用
   `torch.softmax`、矩陣任務用 `@`/`torch.mm`)會被靜態檢查拒絕(`error_type=cheat`)。
   torch 只能用於 output 配置(`torch.empty*`)和瑣碎 glue。
3. module 內不准有 benchmark/計時/print。
4. 允許的 import:`torch`、`triton`、`triton.language as tl`、`math`、
   `from triton.language.extra import libdevice`。
5. Triton 3.6 陷阱:**沒有 `tl.tanh`、沒有 `tl.math.tanh`**——用 `libdevice.tanh(x)`
   或 `2*tl.sigmoid(2*x)-1`;erf 用 `tl.math.erf`。不確定 API 是否存在時,先
   `python -c "import triton.language as tl; print(hasattr(tl, 'xxx'))"` 查證。

## 參數搜尋:每個可調參數都要給一組 grid(LLM 決定範圍)

**行為合約**:不要把 launch / meta 參數寫死成單一值。每一個可調參數都必須暴露成
一組「候選值的 grid」,由你(LLM)依 shape / dtype / roofline 推理決定範圍,並用
`triton.autotune` 把這些 grid 的**交叉組合**掃過去、自動選最快的一組。harness 不
會幫你掃——掃描是寫在候選 code 裡的。

**哪些算可調參數**(有就要給 grid,沒用到的不必硬湊):
- tile / block:`BLOCK_SIZE`、2D 的 `BLOCK_M / BLOCK_N / BLOCK_K`、`ROWS_PER_PROG`、
  reduction 的 partial-sum 數。
- launch:`num_warps`、`num_stages`(、需要時 `num_ctas`)。
- 其他影響效能但**不影響正確性**的 constexpr 旋鈕(swizzle `GROUP_M`、split-K 的
  `SPLIT_K` 等)。

**寫法**(meta 參數放 Config 的 dict;`num_warps/num_stages` 是 Config 的 kwargs):

```python
import triton
configs = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
    for bs in (256, 512, 1024, 2048, 4096)   # ← 你決定的 BLOCK grid
    for w  in (2, 4, 8)                       # ← 你決定的 num_warps grid
    for s  in (2, 3)                          # ← 你決定的 num_stages grid
]

@triton.autotune(configs=configs, key=["n_elements"])  # key = 會觸發重調的尺寸參數
@triton.jit
def _kernel(..., n_elements, BLOCK_SIZE: tl.constexpr): ...
```

**規則**:
1. **每個參數一組 grid**,且在程式上方註解一句「為什麼是這個範圍」(occupancy vs
   register pressure、L2 reuse、tensor-core 對齊 16 倍數…),不要無理由地撒值。
2. **交叉組合要有上限**(建議 ≤ ~16 組):autotune 第一次呼叫會把每組 config 各
   benchmark 一次,而 eval_one 有 120s timeout——grid 太大會 timeout 直接判 fail。
   寧可分兩輪縮放範圍,也不要一次塞百組。
3. `key` 要列出「尺寸一變就該重調」的引數名(如 `n_elements`、`M, N, K`、`n_cols`),
   這樣 autotune 的 best-config 快取對被計時的那個 shape 才有效。
4. **grid 內每一組都必須正確**——只掃效能參數,不要把會改變數值結果的東西(dtype
   精度、是否 fuse 某步)塞進 autotune grid;那種要當成不同候選分開評。
5. autotune 的掃描發生在 eval_one 的 `do_bench` warmup 內,所以回報的 `speedup`
   **已經是最佳 config 的成績**,你不會被掃描時間算進去。
6. **跨輪迭代**:下一個候選依上一輪的結果**收窄或平移** grid(往贏的那格附近加密、
   把明顯爛的範圍砍掉),而不是每次都重撒同一片大網。

> **贏家會被自動記錄**:eval_one 會在計時後掃描候選 module 的全域 autotuner 實例,
> 把每個 kernel 選中的 config 寫進 journal 的 `winning_config` 欄位。接續時用
> `python -m harness.task <op>`(任務卡 workspace 區塊)或 `--export-best` 就看得到
> 上一輪贏的 `BLOCK_SIZE/num_warps/...`,據此把下一輪 grid 往那格收窄。
> 前提:autotuner 要是 **module 層級的全域變數**(`@triton.autotune` 直接掛在
> module 頂層的 `@triton.jit` 函式上)才掃得到;若把它藏在函式內就掃不到——那種
> 情況請自己在候選檔頂端註解寫下選中的 config 當 fallback。

## 驗證指令(唯一的評測入口,不准自己計時)

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m harness.eval_one \
    --operator <op> --candidate runs/agent_<op>/attempts/k<N>.py \
    --journal runs/agent_<op>/journal.jsonl --arm agent_<op> \
    --strategy <方向>        # 選填:這次試的方向標籤(如 autotune_grid),記進 journal
```

- stdout 的 `RESULT_JSON:` 行 = 本次結果(correct、speedup、achieved_gbps、
  bw_utilization、`winning_config`=autotune 選中的 grid config、error 尾段 traceback)。
- `--journal` 會自動記錄每次評測(與 ablation 比較管線相容)。**預算 = journal
  行數**:除非任務另有指示,單一 operator 最多 12 次評測,用完即停,回報最佳結果。

**沒有手動編譯這回事**:Triton 是 JIT,kernel 在首次呼叫時才編譯,而 eval_one
一個指令完成 import → 防作弊 → JIT 編譯 → 正確性 → benchmark → 記帳。每次
嘗試就是「寫檔 → 跑 eval_one」兩步。**免費前置檢查不消耗預算**(只有 eval_one
寫 journal):`python -m py_compile <file>` 抓語法錯、`python -c "import
triton.language as tl; print(hasattr(tl,'xxx'))"` 探測 API——先用它們把低級
錯誤擋掉,別拿寶貴的 eval 次數去試語法。

## Workspace 佈局(固定,跨 session 共用)

每個 operator 一個 workspace:`runs/agent_<operator>/`

```
runs/agent_<op>/
  journal.jsonl        # eval_one 自動寫;行數 = 已用預算(跨 session 累計)
  STATE.md             # 交接檔(本 session 結束前必須更新;只放人類判斷,格式見下)
  best.py              # 由 `python -m harness.task <op> --export-best` 從 journal
                       #   最佳正確 row 還原,不要手動複製(避免走樣)
  attempts/k<N>.py     # 每個候選一檔,N = 該次評測的 eval_index(journal 行號)
```

## 開場協議(每個新 agent session 的第一件事)

1. workspace 已存在 → **先讀 `STATE.md`、`best.py`、journal 末 5 行**,從現有
   eval_index 接續編號,不要重做 STATE.md 標記為無效的方向。
2. workspace 不存在 → `python -m harness.task <op> --init`(自動建目錄 +
   標準 STATE.md 模板;冪等,不會覆蓋既有檔案),從 k0 開始。
3. 任何時候預算(journal 行數)達上限 → 停止生成,直接進入收尾。

## 工作流

1. 讀 spec → 讀 `kb/`(若存在)同 operator 或同類別的成功案例(`kb/kb.jsonl` 索引)。
2. 寫候選 `attempts/k<N>.py` → 用上面的驗證指令評測 → 讀 traceback 尾行修錯。
3. 正確後看 `bw_utilization`:memory-bound op 達 ~85% 即近物理上限,別再浪費預算;
   compute-bound(gemm 類)看 speedup 持續迭代(tiling/swizzle/split-K/pipeline)。
4. 刷新最佳紀錄時:跑 `python -m harness.task <op> --export-best`(從 journal 還原
   `best.py`,不要手動複製)。autotune 候選的贏家 config 會印出來,用來收窄下一輪 grid。

## 收尾(交接給下一個 agent 的硬性規範)

Session 結束前**必須**把 `STATE.md` 更新成以下格式(下一個 agent 只靠這份檔案
+ journal + best.py 就能無縫接續)。**`best` 與 `budget_used` 不在 STATE 手寫**——
那兩個由 `python -m harness.task <op>` 任務卡從 journal 自動算,手寫只會 drift;
STATE.md 只放「機器算不出來的人類判斷」:

```markdown
# STATE — <operator>
status: in_progress | saturated | blocked      # saturated = 已達 roofline/物理上限

## tried(每行一筆,新的加在最上面;格式: k<N> | <strategy> | <speedup 或 error 根因>)
- k3 | autotune_grid | 1.82x (winning BLOCK=2048,num_warps=8)
- k2 | tune_block_size | wrong_output: mask 邊界少算一格
- ...

## pitfalls(本 op 踩過的環境/API 坑,下一個 agent 不要再踩)
- 例:tl.tanh 不存在,用 libdevice.tanh

## next(給下一個 agent 的建議,按優先序)
1. <具體可執行的方向,含理由(例:grid 往 BLOCK=2048 附近加密)>
2. ...
```

規則:`tried` 只增不刪(完整歷史),每行用 `k<N> | strategy | 結果` 結構化單行;
`next` 每次收尾重寫;若 status=saturated,在 next 寫「不建議續跑」並給證據(bw% 或
理論分析)。刷新最佳記得跑 `--export-best`。最後在對話中回報:最佳 speedup、
workspace 路徑、STATE.md 摘要。

## 禁止事項

- 不准修改 `harness/`、`operators/`、`configs/` 下任何檔案(評測公正性)。
- 不准繞過 eval_one 自行宣稱效能數字。
- 不准在 kernel module 裡讀寫檔案、起 subprocess。
