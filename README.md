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



\## 当前阶段



当前处于：



M0 —— 项目骨架搭建。



目前已经完成：



\- FastAPI 应用初始化

\- `GET /health` 健康检查接口

\- 基于 pytest 的 API 自动化测试

\- 使用 `pyproject.toml` 管理 Python 项目和依赖

\- Git 本地版本管理初始化



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

