\# Evaluation Harness



一个用于评测 AI 系统的可复现 Evaluation Harness。



\## 项目目标



本项目旨在构建一个小型、可复现、可测试的 AI 系统评测框架。



核心目标是统一管理：



\- 评测输入

\- 模型输出

\- 评测指标

\- 评测流程

\- 测试与结果记录



Evaluation Harness 是本项目主体。



后续会加入 RAG Pipeline 和 Tool-calling Agent，但它们是“被评测对象”，不是项目主体。


## 当前进度

M0 已完成，包括：

- FastAPI 项目骨架
- `/health` API
- pytest 自动测试
- Git 本地版本管理
- GitHub 远程仓库
- GitHub Actions CI

当前进入 M1：最小评测闭环。



\## 项目结构



```text

evaluation-harness/

├── src/

│   └── eval\_harness/

│       ├── \_\_init\_\_.py

│       └── api.py

├── tests/

│   └── test\_api.py

├── pyproject.toml

├── .gitignore

└── README.md

