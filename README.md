# Evaluation Harness

一个用于评测 AI 系统的可复现、可测试 Evaluation Harness。

本项目的主体是评测框架本身，而不是某个单独的 RAG 或 Agent Demo。后续加入的 RAG Pipeline 和 Tool-calling Agent 将作为被评测对象接入 Harness。

## 项目目标

本项目旨在构建一个小型但完整的 AI Evaluation Harness，用统一流程管理：

- 评测输入与数据集版本
- 被测系统输出
- Deterministic Evaluator
- LLM-as-a-Judge
- Human / Gold / Judge 标签
- Raw Agreement
- Cohen's Kappa
- Confusion Matrix
- Disagreement Analysis
- 可复现结果记录
- 自动化测试

目标不是单纯得到一个分数，而是建立一个：

**可运行 → 可复现 → 可比较 → 可解释 → 可验证**

的评测流程。

---

## 当前进度

### M0｜项目骨架

已完成：

- FastAPI 项目骨架
- `/health` API
- pytest 自动测试
- Git 本地版本管理
- GitHub 远程仓库
- GitHub Actions CI

### M1｜最小评测闭环

已完成：

- `SystemUnderTest` abstraction
- 真实 Qwen API 接入
- 普通文本输出
- Structured Output
- Exact Match Evaluator
- Normalized Match Evaluator
- Evaluator Injection
- CLI / FastAPI 评测入口
- Batch Evaluation
- 50 条人工校验 deterministic dataset
- Dataset Version
- Dataset SHA-256
- JSON 结构化结果输出
- 可复现实验结果保存

### M2｜Judge Consistency Analysis

已完成：

- `JudgeCase`
- `HumanLabel`
- `JudgeOutput`
- `JudgeResult`
- LLM-as-a-Judge
- Batch Judge
- Judge Result Persistence
- Human / Gold / Judge 数据加载
- Raw Agreement
- Cohen's Kappa
- Confusion Matrix
- Disagreement Identification
- Disagreement Decomposition
- Strict / Lenient Rubric 对比实验
- Structured Experiment Summary

---

## 项目结构

```text
evaluation-harness/
├── data/
│   ├── eval_cases.jsonl
│   ├── human_labels_v0.1.jsonl
│   ├── human_labels_v0.2.jsonl
│   ├── judge_cases_v0.1.jsonl
│   ├── judge_cases_v0.2.jsonl
│   └── disagreements_v0.2.jsonl
│
├── results/
│   ├── eval_v0.1_exact.json
│   ├── eval_v0.1_normalized.json
│   ├── judge_v0.1_qwen3.8-max.json
│   ├── judge_v0.2_qwen3.8-max.json
│   ├── judge_v0.2_strict_qwen3.8-max.json
│   ├── judge_v0.2_lenient_qwen3.8-max.json
│   └── m2_experiment_summary_v0.2.json
│
├── src/
│   └── eval_harness/
│       ├── __init__.py
│       ├── agreement.py
│       ├── api.py
│       ├── cli.py
│       ├── dataset.py
│       ├── evaluator.py
│       ├── experiment.py
│       ├── judge.py
│       ├── models.py
│       ├── reporting.py
│       ├── reproducibility.py
│       ├── runner.py
│       └── systems.py
│
├── tests/
│   ├── test_agreement.py
│   ├── test_api.py
│   ├── test_cli.py
│   ├── test_dataset.py
│   ├── test_evaluator.py
│   ├── test_experiment.py
│   ├── test_judge.py
│   ├── test_llm_integration.py
│   ├── test_reporting.py
│   ├── test_reproducibility.py
│   ├── test_runner.py
│   └── test_systems.py
│
├── pyproject.toml
├── .gitignore
└── README.md
```

---

# M1｜Deterministic Evaluation

## Dataset

Dataset version:

```text
v0.1
```

Cases:

```text
50
```

Model:

```text
qwen3.8-max
```

Dataset SHA-256:

```text
c54f3af49e2871908419b11be62570eedb4dfd11e2efa0ca7b11b3730d7df995
```

## Evaluation Results

### Exact Match

```text
50 / 50 passed
Pass rate: 100%
```

### Normalized Match

```text
50 / 50 passed
Pass rate: 100%
```

## M1 Limitation

The v0.1 dataset mainly contains deterministic short-answer tasks.

Under the tested model, both exact and normalized matching achieved 100%, so this dataset does not expose meaningful differences between the two evaluators.

This result is still useful as a reproducible baseline, but more complex semantic evaluation requires a different evaluation method.

---

# M2｜LLM Judge Consistency Analysis

M2 extends the Harness from deterministic matching to LLM-based evaluation.

The core comparison separates three types of labels:

- **Gold Label**  
  Reference label used as the correctness target.

- **Human Label**  
  Human or human-reviewed evaluation label used for agreement analysis.

- **Judge Label**  
  Output produced by the LLM-as-a-Judge.

This allows the Harness to distinguish:

```text
Judge vs Human
→ agreement

Judge vs Gold
→ correctness
```

These are related but not equivalent measurements.

---

## M2 Dataset

### v0.1 Baseline

```text
24 cases
```

The baseline dataset contains relatively clear PASS / FAIL examples covering:

- factual correctness
- instruction following
- format requirements
- irrelevant answers
- semantic equivalence

On this dataset, Judge, Human and Gold labels were fully aligned.

### v0.2 Boundary-Focused Dataset

```text
12 cases
```

The v0.2 dataset focuses on more difficult evaluation boundaries, including:

- partially correct answers
- additional false claims
- incomplete multi-part requirements
- precision requirements
- soft wording constraints
- semantic equivalence
- strict vs lenient instruction adherence

---

## Judge Variants

The experiment uses the same underlying model:

```text
qwen3.8-max
```

but applies two different evaluation rubrics.

### Strict Rubric

A response passes only when it is correct and satisfies all explicit requirements.

### Lenient Rubric

A response may still pass when its core answer is correct, even if secondary format, length or quantity requirements are not fully satisfied.

Keeping the underlying model fixed makes rubric strictness the main experimental variable.

---

# M2 Metrics

The Harness currently calculates:

### Raw Agreement

Percentage of samples where two label sets make the same PASS / FAIL decision.

### Cohen's Kappa

Measures agreement after accounting for expected agreement by chance.

### Confusion Matrix

PASS is treated as the positive class.

```text
TP = reference PASS, Judge PASS
TN = reference FAIL, Judge FAIL
FP = reference FAIL, Judge PASS
FN = reference PASS, Judge FAIL
```

### Disagreement Identification

The Harness aligns labels by `case_id` and returns the exact cases where two evaluators disagree.

---

# M2 Experimental Results

## Strict Judge

### Judge vs Human

```text
Raw Agreement: 0.9167
Cohen's Kappa: 0.8333
```

Confusion Matrix:

```text
TP = 6
TN = 5
FP = 0
FN = 1
```

### Judge vs Gold

```text
Raw Agreement: 1.0000
Cohen's Kappa: 1.0000
```

Confusion Matrix:

```text
TP = 6
TN = 6
FP = 0
FN = 0
```

---

## Lenient Judge

### Judge vs Human

```text
Raw Agreement: 0.7500
Cohen's Kappa: 0.4706
```

Confusion Matrix:

```text
TP = 6
TN = 3
FP = 2
FN = 1
```

### Judge vs Gold

```text
Raw Agreement: 0.8333
Cohen's Kappa: 0.6667
```

Confusion Matrix:

```text
TP = 6
TN = 4
FP = 2
FN = 0
```

---

# Disagreement Analysis

Three representative disagreement cases were identified.

## judge-v02-001

Category:

```text
partially_correct_with_false_claim
```

The answer correctly describes the mutability difference between Python `list` and `tuple`, but adds an incorrect absolute claim that tuples are always more memory-efficient.

Result:

```text
Gold    = FAIL
Human   = FAIL
Strict  = FAIL
Lenient = PASS
```

The lenient rubric over-weighted the correct core statement and allowed an additional false claim to pass.

---

## judge-v02-002

Category:

```text
requirement_completeness
```

The prompt explicitly requests two HTTPS security advantages, but the candidate answer provides only one.

Result:

```text
Gold   = FAIL
Human  = PASS
Strict = FAIL
```

This demonstrates that disagreement with a human label does not necessarily imply Judge error relative to the reference label.

The Human evaluation applied a more permissive completeness standard, while Gold and Strict Judge required all explicit requirements to be satisfied.

---

## judge-v02-004

Category:

```text
format_precision_violation
```

The prompt requires π rounded to two decimal places, while the candidate returns `3.1416`.

Result:

```text
Gold    = FAIL
Human   = FAIL
Strict  = FAIL
Lenient = PASS
```

The numeric value is reasonable, but the response violates the explicit precision requirement.

The lenient rubric treated this requirement as secondary and therefore produced a false positive relative to Gold.

---

# Key Findings

The current experiment supports several observations.

### 1. Human agreement and correctness should be measured separately

The Strict Judge achieved:

```text
Human Agreement = 0.9167
Gold Correctness = 1.0000
```

Therefore, disagreement with Human labels did not necessarily indicate an incorrect Judge decision relative to Gold.

### 2. A more lenient rubric did not improve Human agreement

The Lenient Judge produced:

```text
Human Agreement = 0.7500
Gold Correctness = 0.8333
```

Both were lower than the Strict Judge.

This suggests that simply relaxing evaluation criteria does not necessarily make an LLM Judge more aligned with human decisions.

### 3. Rubric design changes the error profile

The Lenient Judge introduced false positives on cases involving:

- additional incorrect claims
- explicit precision constraints

This shows that Judge reliability depends not only on the underlying model, but also on how the evaluation rubric defines acceptable answers.

### 4. Agreement is not equivalent to correctness

The Harness therefore reports Judge-Human agreement and Judge-Gold correctness separately rather than treating them as interchangeable metrics.

---

# Reproducibility

The project records:

- dataset versions
- model identifiers
- structured Judge outputs
- Human labels
- Gold labels
- Judge result files
- Raw Agreement
- Cohen's Kappa
- Confusion Matrices
- disagreement cases
- structured experiment summaries

The M2 experiment summary is stored as:

```text
results/m2_experiment_summary_v0.2.json
```

This allows the reported metrics to be regenerated from the underlying evaluation artifacts rather than manually copied from terminal output.

---

# Testing

The project uses pytest for unit and integration testing.

Current test status:

```text
72 passed
1 skipped
```

The skipped test is the real LLM integration test, which is disabled by default to avoid unintended API calls.

Tests currently cover:

- API endpoints
- dataset loading
- evaluator behavior
- runner behavior
- LLM system behavior
- structured output
- reproducibility hashing
- result persistence
- Judge behavior
- batch Judge execution
- rubric injection
- raw agreement
- Cohen's Kappa
- confusion matrix
- disagreement identification
- experiment summary generation

---

# Current Limitations

The current M2 experiment is intentionally small and should be interpreted as an engineering validation rather than a large-scale evaluation study.

Important limitations include:

- v0.2 contains only 12 boundary-focused cases.
- The experiment currently uses one underlying LLM model with two different rubrics.
- Human labels for part of the v0.2 dataset were AI-assisted / human-reviewed rather than fully independent human annotation.
- The current label space is binary: `PASS` / `FAIL`.
- No large-scale inter-rater reliability study has yet been performed.

These limitations are preserved explicitly rather than treating the current metrics as general claims about LLM Judge reliability.

---

# Next Steps

The next milestones will extend the Evaluation Harness while keeping the Harness itself as the project主体.

Planned work includes:

### M3｜RAG as System Under Test

Add a small RAG pipeline as a new evaluated system and evaluate dimensions such as:

- answer correctness
- retrieval relevance
- groundedness

### M4｜Tool-calling Agent as System Under Test

Add a simple Agent workflow and evaluate:

- task success
- tool selection
- argument correctness
- execution outcomes

### M5｜Engineering Completion

Complete:

- Docker
- CI refinement
- reproducible experiment commands
- README / architecture documentation
- GitHub cleanup
- final project presentation and interview explanation

The Evaluation Harness remains the central system throughout these milestones.