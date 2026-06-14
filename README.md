# pro_max — LLM Tree-Search Triton Kernel Optimization Harness

自動化 closed-loop:LLM(DeepSeek)生成 Triton kernel 候選 → 編譯/正確性/實測(獨立 subprocess、GPU pool)→ 有記憶的 LLM 評分(roofline 錨定)→ top-k beam 展開下一輪。每個機制都有 ablation 開關,baseline(單路徑 sequential refinement)是同一 harness 的一個模式,**等預算可比**。

**兩層搜尋**:外層由 LLM/樹搜尋探索 kernel 的「演算法結構」(融合、單 pass、persistent…);內層每個候選把可調參數(`BLOCK_SIZE`、tile 形狀、`num_warps/num_stages`…)各自暴露成一組 **LLM 決定範圍的 grid**,用 `triton.autotune` 掃交叉組合自動選最快——harness 不做 grid search,掃描寫在候選 code 裡。eval_one 評測後會自動把每個 kernel 選中的 config 記進 journal 的 `winning_config`,下一輪據此收窄 grid。詳見 [HARNESS.md](HARNESS.md) 的「參數搜尋」章節。

> **操作手冊**:[human.md](human.md)(人類流程 + 聚焦 operator 與預算)/
> [HARNESS.md](HARNESS.md)(agent session 合約)。

## 環境

```bash
conda activate pro_max          # python 3.11 + torch cu128 + triton + openai SDK
export DEEPSEEK_API_KEY=...
cd para_final/pro_max
```

LLM 用 DeepSeek API(OpenAI 相容,`deepseek-v4-flash`,$0.14/$0.28 per 1M tokens、
cache hit $0.0028):原生支援 `logprobs`(DeepConf 信心可用)、`thinking` 參數
(enabled/disabled + reasoning_effort)、自動 context caching(免管理)。
舊別名 deepseek-chat / deepseek-reasoner 於 2026-07-24 退役,本專案直接用新 ID。

## 快速驗證(按順序)

```bash
# 1. verifier 地基:手寫 vector_add 過編譯/正確性/計時
python -m harness.verifier --self-test

# 2. LLM 能力探測:async 生成 / thinking 參數 / logprobs /
#    JSON mode / 自動 cache 欄位(決定 confidence 模式)
python -m harness.llm --probe

# 3. 單 operator 煙霧測試(1 輪、少量候選)
python runner.py --config configs/arms/full.yaml \
    --operators vector_add --rounds 1 --total-candidates 4 --gpus 0

# 4. 小型 ablation 演練(baseline vs full,等預算)
python run_ablation.py --arms full sequential_baseline \
    --operators vector_add softmax --total-candidates 6 --gpus 0 1
```

## 正式跑法(12 小時切分)

```bash
# 主實驗:full arm、17 operator、4 卡(~7h 內)
python runner.py --config configs/arms/full.yaml --operators all

# 非對照組:12 個「有加速空間」的 operator(排除對照組 5 個:vector_add /
# vector_exp / sum / layer_norm / softmax)。對照組單獨用 --total-candidates 6 跑即可。
python runner.py --config configs/arms/full.yaml \
    --operators geglu swiglu welford rms_norm cross_entropy fused_linear_cross_entropy \
                kl_div gemm addmm int4_gemm rope embedding

# 多卡:GPUPool 每張卡同時只跑一個候選(計時乾淨),候選夠多時自動填滿所有卡。
# 8 卡時把 default.yaml 的 gpus 覆寫掉,並把 llm.max_concurrent 調到 ≥ 卡數:
python runner.py --config configs/arms/full.yaml --operators all --gpus 0 1 2 3 4 5 6 7

# ablation:小子集證明各機制有效
python run_ablation.py \
    --arms full no_bandit no_memory no_scorer no_roofline conf_in_score sequential_baseline \
    --operators vector_add softmax layer_norm sum gemm rope \
    --total-candidates 16
```

### 用 DeepSeek Pro(`deepseek-v4-pro`)

預設模型是 `deepseek-v4-flash`(`configs/default.yaml` 的 `llm.model`)。要換成更大的
`deepseek-v4-pro`,兩種方式:

```bash
# (a) --model 覆寫任何 arm 的 llm.model(最通用,可疊在 full / minimal_kb… 之上)
python runner.py --config configs/arms/full.yaml --operators all --model deepseek-v4-pro

# 非對照組 12 op + pro 模型:
python runner.py --config configs/arms/full.yaml --model deepseek-v4-pro \
    --operators geglu swiglu welford rms_norm cross_entropy fused_linear_cross_entropy \
                kl_div gemm addmm int4_gemm rope embedding

# (b) 內建的 pilot_full_pro arm:已把 llm.model 設成 deepseek-v4-pro,
#     搭配 pilot 搜尋設定(beam_k 2 / 12 候選),用於「模型太弱」假設對比。
python runner.py --config configs/arms/pilot_full_pro.yaml --operators all
```

> `--model` 沿用 `--rounds` / `--total-candidates` 的覆寫機制,單 run 內全程生效並寫進
> `runs/<arm>/config.yaml`、`run_meta.json`,事後可追。

## Ablation arms

**機制收斂(pilot 220 節點的結論)**:加速大頭來自 seed 與修錯迴圈,策略指派/bandit/scorer 增量未被證實——所以現在的主軸是 minimal 系列,其餘機制保留為開關供負結果 ablation。

| arm | 開關 | 驗證的假設 |
|---|---|---|
| **`minimal`** | 樹狀分支 + 自由修錯,無策略/bandit/scorer | 最小有效迴圈(主力) |
| **`minimal_kb`** | minimal + KB 成功案例進 prompt | **成功案例資料庫有用**(目前的核心假設) |
| **`pilot_full_pro`** | pilot 設定 + `deepseek-v4-pro` | 「deepseek 太弱」假設 |
| `pilot_seq` / `sequential_baseline` | 單鏈 refinement | 樹狀展開 > 單路徑(等預算) |
| `full` | 全機制開 | 原提案系統 |
| `no_bandit` / `no_memory` / `no_scorer` / `no_roofline` / `conf_in_score` | 個別關閉 | 各機制邊際貢獻(pilot 顯示趨近 0,留作報告證據) |

所有 arm 共用 input seed、verifier、prompt 模板;sequential 的 rounds 自動 = `total_candidates`(等預算)。

## KB:成功案例資料庫

```bash
# 1) 從既有 runs 萃取每 operator top-3 正確 kernel(code 已在 journal 裡)
python -m harness.kb --build            # -> kb/kb.jsonl + kb/<op>/<rank>_<speedup>x.py
python -m harness.kb --show

# 2) 對比「資料庫有沒有用」(同一 KB 快照、等預算)
python run_ablation.py --arms minimal minimal_kb pilot_seq \
    --operators sum welford geglu gemm --gpus 3,4,5
```

KB 檢索:同 operator 範例優先、不足補同類別,`kb.max_exemplars` 控制每 prompt 附幾個。跨 run 累積——每跑完一批,重跑 `--build` 即吸收新成功案例。

## Claude agent 模式

[HARNESS.md](HARNESS.md) 定義了讓 agentic Claude 直接在 harness 內迭代的合約:agent 自己寫 kernel、呼叫 `python -m harness.eval_one --journal ... --strategy <方向>` 驗證(預算 = journal 行數),結果進同一套比較管線。適合 gemm 類難題或檢驗「更強模型」上限。狀態交接:`STATE.md` 只放人類判斷(status/tried/pitfalls/next),best 與預算由任務卡從 journal 自動算,`best.py` 用 `python -m harness.task <op> --export-best` 從 journal 還原(不手動複製、不 drift)。

多卡:每個 `eval_one` 是「一個候選、一張卡」,所以 agent 模式靠**把不同 operator 分到不同卡平行開 session**(`CUDA_VISIBLE_DEVICES=<gpu>`)。注意單一候選內部的 autotune grid 是在它那張卡上序列掃,多卡只平行化「不同候選」。

決定下一個 session「優化誰、給多少預算」用 triage 總表:

```bash
python -m harness.task --triage   # 每 op:類型/建議預算(comp 28、多pass 12、已融合 6)
                                  # /跨所有 runs 的最佳紀錄/agent 進度/建議動作
```

## 輸出與比較

- `runs/<arm>/journal.jsonl` — 每候選一筆:正確性、speedup、達成頻寬、**分數分解**(`score_speedup/score_headroom/score_confidence/score_final`)、**`winning_config`**(autotune 選中的 grid config)、token/GPU 成本。
- `runs/<arm>/metrics.csv` — 每輪:`candidates_so_far, best_speedup_so_far, ...`。
- `runs/<arm>/transcripts/<eval_index>_<node_id>_<kind>.json` — **每次 LLM 互動的完整逐字稿**(報告附錄用):`kind=generation`(生成候選)或 `assessment`(scorer 評分),內含 `system`/`prompt`/`response` 原文、模型 `reasoning`(thinking 內容)、`usage`、以及失敗時的 `error`(連 genfail/parse 失敗的回覆也留)。由 `configs` 的 `transcripts.enabled` 開關(預設開),`include_reasoning` 控制是否一起存較佔空間的 reasoning。僅 API 迴圈(runner)產生——agent 模式的 prompt 即對話本身,在 harness 之外。
- `runs/ablation_*/comparison/` — `compare_table.md`(arm × operator 最佳 speedup + fast_p + 成本)、`budget_curves.csv/png`(best-so-far vs 已驗證候選數,x 軸等預算對齊)、`strategy_report.csv`(策略有效性)。全部由 journal 自動衍生。

## 設計要點 / 防作弊

- 候選只能定義 `triton_run(*inputs)`;計時與正確性由 harness 模板執行(`harness/eval_one.py`),候選碰不到時鐘。
- 參數調優:候選把每個可調參數暴露成 LLM 決定的 grid、用 `triton.autotune` 內掃;eval_one 計時後自動掃描候選的 autotuner 實例,把贏的 config 記進 `winning_config`(autotune 掃描發生在 `do_bench` warmup 內,計時即最佳 config,不含掃描成本)。
- 靜態檢查拒絕直接呼叫同名 torch op(`spec.forbidden_substrings`)。
- 每候選獨立 subprocess + timeout + `CUDA_VISIBLE_DEVICES` 綁卡;每張卡同時只跑一個驗證,計時乾淨。
- baseline 與候選用同一 `triton.testing.do_bench`(median),兩邊都含輸出 allocation,修正了 para_final/code 舊 benchmark 的不對稱。
- 信心訊號:DeepSeek 原生 `logprobs` → DeepConf 離線版(生成 token 滑動視窗最低平均 logprob);啟動時仍會探測,萬一被關閉自動降級 verbalized confidence,結果記在 `run_meta.json`。
- 成本:v4-flash 全 17 operator × 32 候選的主實驗,系統前綴自動命中 cache,總 API 成本約 $1–3 美元。
