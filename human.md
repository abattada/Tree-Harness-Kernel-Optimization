# human.md — pipeline 操作手冊(給人看的)

這份是**人類操作者**的完整流程。系統有兩種跑法,分工原則:

| 模式 | 適合 | 你要做的事 |
|---|---|---|
| **API 迴圈**(runner/run_ablation) | 便宜題、批量、ablation 對比 | 下指令、等結果 |
| **Agent session**(Claude Code + HARNESS.md) | 難題(matmul 類)、需要查文件/多步推理 | 開新對話、貼一句 prompt |

> **目前狀態:已重置,從 0 開始**——`runs/` 與 `kb/` 已全清。下表 headroom 是 roofline
> 先驗(仍有效);標註的 pilot 數字(如 welford 2.46x)是重置前的歷史實測,需重跑重現。
> welford/rms_norm/geglu 已列為正式 ★ 目標,不因 pilot 跑過就跳過。
>
> **參數調優改成 grid + autotune**:候選現在把每個可調參數(`BLOCK_SIZE`、tile 形狀、
> `num_warps/num_stages`…)暴露成一組 grid,用 `triton.autotune` 內掃選最快;eval_one
> 評測後自動把贏的 config 記進 journal 的 `winning_config`。你看結果時多一個欄位可用,
> 操作流程不變。細節見 [HARNESS.md](HARNESS.md) 的「參數搜尋」。

---

## 0. 一次性設置(每次開終端機)

```bash
conda activate pro_max
cd ~/para_final/pro_max
export DEEPSEEK_API_KEY=sk-...        # API 迴圈需要;agent session 不需要
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
#                                      # 挑空卡;被占的卡跑出來的計時不可信
```

---

## 1. 聚焦 operator(headroom 實測,5090/GPU2,2026-06-11)

headroom = torch ref 實測時間 ÷ 頻寬理論下限。**優先做粗體標 ★ 的**(welford/rms_norm/geglu
是正式目標,不因 pilot 跑過就跳過——重置後一律重跑、進同一套比較管線):

| 優先 | operator | headroom | 預算 | 模式 | 為什麼有空間 |
|---|---|---|---|---|---|
| ★ | **rope** | 5.4x | 12 | API | ref 多次 materialize(slice/乘法/cat) |
| ★ | **kl_div** | 5.2x | 12 | API | ref 多個 elementwise pass + reduction |
| ★ | **cross_entropy** | 2.4x | 12 | API | ref materialize 1GB log_softmax |
| ★ | **int4_gemm** | ~2x(算力下限修正後) | 28 | **Agent** | ref 整塊 dequant 再 cuBLAS;fused W4A16 是文獻標準題 |
| ★ | **fused_linear_cross_entropy** | ~1.5–2x + 記憶體故事 | 28 | **Agent** | ref materialize 512MB logits;Liger 主打題 |
| ★ | **welford** | ~2.5x(pilot 2.46x) | 12 | API | reduction;兩 pass mean/var → 單 pass Welford |
| ★ | **rms_norm** | ~2.3x(pilot 2.33x) | 12 | API | normalization;ref 多 pass + materialize |
| ★ | **geglu** | ~1.6x(pilot 1.61x) | 12 | API | elementwise gated;ref materialize 中間 gate |
| 次 | embedding | 2.7x(隨機訪問折扣後 ~1.5–2x) | 12 | API | gather 低效 |
| 對照組 | vector_add/softmax/sum/layer_norm/vector_exp(~1.0–1.2x) | 6 | API | 各跑 6 次「確認無空間」,報告誠實性用 |

報告框架:這組 op ≈ Liger-Kernel 的 operator family(RMSNorm/RoPE/SwiGLU/CE/FLCE/KLDiv),加速論述同一套——消除中間張量 materialization,baseline 為 PyTorch eager。

---

## 2. 標準流程(按順序)

### Step 0:看現況,決定要跑什麼

```bash
python -m harness.task --triage     # 每 op:建議預算/最佳紀錄/agent 進度/建議動作
```

### Step 1:更新 KB(把歷史成功案例吸收進資料庫)

```bash
python -m harness.kb --build        # 掃所有 runs/ → kb/;每跑完一批就重跑一次
python -m harness.kb --show
```

> 已重置:第一次跑時 `runs/` 是空的,build 出來的 KB 也是空的(0 entries)——這正常。
> 跑完 Step 2 產生第一批成功案例後,回來重跑 `--build` 才會有東西。

### Step 2:API 迴圈跑便宜題(主力產分)

```bash
# 聚焦組(有空間的 memory-bound)
python runner.py --config configs/arms/minimal_kb.yaml \
    --operators rope kl_div cross_entropy welford rms_norm geglu embedding --gpus 2

# 對照組(確認無空間,各 6 次)
python runner.py --config configs/arms/minimal_kb.yaml \
    --operators vector_add softmax sum layer_norm vector_exp \
    --total-candidates 6 --gpus 2

# 多卡加速:所有 operator 共用一個 GPU pool、每張卡同時只跑一個候選。
# 給滿卡即可,卡多時把 llm.max_concurrent 調到 ≥ 卡數(否則卡會閒等生成):
python runner.py --config configs/arms/minimal_kb.yaml --operators all --gpus 0 1 2 3 4 5 6 7
```

跑完回 Step 1 重建 KB(新成功案例進資料庫)。

### Step 3:Agent session 跑難題(每題開一個新 Claude Code 對話)

開新對話,貼這一句(預算/GPU 自行代換):

```
讀 ~/para_final/pro_max/HARNESS.md,優化 int4_gemm,預算 28 次,用 GPU 2。
```

```
讀 ~/para_final/pro_max/HARNESS.md,優化 fused_linear_cross_entropy,預算 28 次,用 GPU 2。
```

- 中途斷了/想換人接手:**再開新對話貼同一句**(把「優化」改「接續優化」更明確)。
  交接靠 `runs/agent_<op>/STATE.md` + `best.py` + journal,不靠對話記憶。
- 檢查 agent 有沒有守規矩:`python -m harness.task <op>` 看 workspace 狀態
  (STATE.md 該存在、attempts/ 該有編號檔、預算 = journal 行數)。任務卡的 workspace
  區塊會直接印出最佳 speedup 與 **winning grid config**——你不用翻 STATE.md 找數字
  (best/預算改由 journal 自動算,STATE.md 只放 tried/pitfalls/next 等人類判斷)。
- 多卡:每個 operator 各派一張卡平行開 session(prompt 裡的「用 GPU N」自行代換),
  例如 GPU2 跑 int4_gemm、GPU3 跑 fused_linear_cross_entropy。
- 最佳 kernel 要存檔:`python -m harness.task <op> --export-best`(從 journal 還原
  `best.py`,不要手動複製);它也會印出贏的 autotune config,供下一輪收窄 grid。

### Step 4:Ablation(證明機制有用,報告的證據表)

```bash
# 核心對比:純分支 vs 分支+KB vs 單鏈 baseline(等預算,自動對齊)
python run_ablation.py --arms minimal minimal_kb pilot_seq \
    --operators rope kl_div cross_entropy --gpus 2

# 模型強度假設(可選):v4-pro vs v4-flash
python runner.py --config configs/arms/pilot_full_pro.yaml --operators int4_gemm --gpus 2
```

注意:ablation 期間**不要**在 arm 之間重建 KB(minimal_kb 各 arm 要用同一份 KB 快照才公平)。

### Step 5:彙整報告數據

```bash
# arm 對比表 + 等預算曲線圖(自動生成,別手填數字)
ls runs/ablation_*/comparison/      # compare_table.md, budget_curves.png, fast_p.csv

# token 成本與外插
python analysis/token_report.py --runs runs/ablation_*/minimal runs/ablation_*/minimal_kb

# 全域最佳總表
python -m harness.task --triage
```

---

## 3. 結果放在哪

```
runs/<arm>_<時間>/         # API 迴圈的一次 run
  journal.jsonl            #   每候選:正確性/speedup/頻寬/分數分解/winning_config/token 成本/完整 code
  metrics.csv              #   每輪 best-so-far(畫曲線用)
  summary.json             #   一眼結論
  candidates/*.py          #   生成的 kernel 原始碼
  transcripts/*.json       #   每次 LLM 互動逐字稿(system/prompt/response/reasoning),報告附錄用
runs/ablation_<時間>/comparison/   # 跨 arm 對比(報告直接用)
runs/agent_<op>/           # agent workspace
  journal.jsonl            #   行數 = 已用預算;含 winning_config 與 strategy 標籤
  STATE.md                 #   只放人類判斷(status/tried/pitfalls/next);best/預算不在這手寫
  best.py                  #   由 `task <op> --export-best` 從 journal 還原
  attempts/k<N>.py         #   每個候選一檔
kb/                        # 成功案例資料庫(kb.jsonl 索引 + 每 op 最佳 kernel)
```

看單一候選贏的 grid config(下一輪收窄 grid 用):

```bash
python -m harness.task <op>          # 任務卡 workspace 區塊直接印 best + winning grid config
# 或從 journal 撈:
python -c "import json;[print(r['winning_config']) for r in map(json.loads,open('runs/agent_<op>/journal.jsonl')) if r['correct']]"
```

---

## 4. 常見問題

- **GPU 被占**:先 `nvidia-smi` 挑空卡;計時用的卡必須獨佔,否則 speedup 不可信。
- **`--gpus 3,4,5` 或 `--gpus 3 4 5`** 都可以。給幾張就同時評幾個候選(每卡一個)。
  卡多時把 `llm.max_concurrent`(default.yaml,預設 10)調到 ≥ 卡數,否則卡會閒等 code。
- **多進程同時啟動別把驅動打爆**:runner.py 已內建 spawn stagger(`launch_stagger_s`,
  default.yaml 預設 0.75s),自動錯開子進程的 CUDA-init/Triton 編譯,不讓 N 個瞬間齊發;
  卡很多時可調大一點。**agent 模式手動 `&` 並行**多個 `eval_one` 時,設
  `export EVAL_STARTUP_JITTER_S=2` 給每個進程隨機 0–2s 啟動抖動(對計時無影響,
  do_bench 內部量測)。
- **autotune grid 超時**:單一候選的 grid 在它那張卡上序列掃,交叉組合太大會撞
  `candidate_timeout_s`(預設 120s)被判 timeout——多卡幫不上(只平行化不同候選)。
  grid 控制在 ≤ ~16 組,或分兩輪收窄(HARNESS.md「參數搜尋」有規範)。
- **候選被判 cheat**:看 journal 的 `error_msg`(列出命中的 regex)。檢查是 code
  真的呼叫 torch 同名 op,還是新的誤殺模式——誤殺請回報修 regex,不要直接放寬。
- **sum/reduction 類**:所有階段(含 partial 合併)都必須是 Triton kernel,
  torch.sum/.sum() 一律拒絕——這是規格,不是 bug。
- **OOM**:單一候選 OOM 只會 fail 那一個候選(獨立 subprocess),不影響整體;
  若 ref 本身 OOM(卡被占),換卡。
- **API 429/超時**:SDK 自動重試;若整輪很慢,降 `llm.max_concurrent`。
- **等預算原則**:要互相比較的 arm,`total_candidates` 必須相同;sequential 會
  自動對齊,不要手動改它的 rounds。

---

## 5. 報告素材對應

| 報告章節 | 來源 |
|---|---|
| 方法:op 選擇(roofline 先驗) | 上面 §1 headroom 表(可用 GPU2 重測) |
| 主結果:每 op 最佳 speedup | `task --triage` / `compare_table.md` |
| Ablation:分支 vs 單鏈、KB 有無 | `runs/ablation_*/comparison/`(表+曲線,重跑後生成) |
| 負結果:策略庫/bandit/scorer 增量≈0 | 重跑 ablation 的 `no_*` arm journal + 對話結論(原 pilot runs 已重置) |
| 回饋品質決定成敗的案例 | geglu 事件:tl.tanh ×24 連敗 → 修回饋截斷後一輪修通(重置前結論,需重現) |
| 誠實性:無空間 op 如實報 ~1.0x | 對照組 6 op 的 journal |
| 成本 | `token_report.py` 輸出(全程 < $5) |
