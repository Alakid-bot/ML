# 基于机器学习的网络服务负载预测

<p align="center">
  <strong>中文</strong>
  &nbsp;|&nbsp;
  <a href="./README.en.md"><strong>English</strong></a>
</p>

---

本项目是一个面向网络服务最大可支持负载预测的双语机器学习系统。它包含训练 pipeline、Streamlit 前端界面、训练历史持久化，以及 Zeabur 部署配置。

## 项目目标

本项目将负载预测建模为监督学习回归问题：

```text
输入：资源配置 + 流量 + KPI 指标
输出：max_supported_load_mbps，即最大可支持负载
```

模型学习 CPU、内存、链路容量、延迟、吞吐量、丢包率和输入流量之间的关系，从而预测网络服务在当前条件下能稳定承载的最大负载。

预测结果后续可以提供给 Optimization / Orchestrator 使用，用来判断是否需要扩容、调整资源或重新调度服务。

## 数据集格式

训练 CSV 必须包含以下特征列：

```text
traffic_input_mbps
cpu_cores
ram_gb
link_capacity_mbps
cpu_utilization_percent
memory_utilization_percent
latency_ms
throughput_mbps
packet_loss_percent
```

目标列是：

```text
max_supported_load_mbps
```

详细字段说明和模板见：

```text
docs/dataset_schema.md
data/profiling_dataset_template.csv
```

模板只用于检查字段结构。正式训练至少需要 20 行数据；演示建议准备 50-100 行；如果希望结果更真实，最好使用数百到数千条 profiling 样本。

## 训练方法

系统会训练多个候选回归模型，并根据验证集 RMSE 自动选择最佳模型。

当前候选模型包括：

```text
dummy_mean        均值基线模型
ridge             岭回归模型
mlp               MLP 神经网络回归模型
adaptive_hybrid   自适应混合负载预测模型
```

### 基线与标准模型

`dummy_mean` 只预测训练集中目标值的平均值。它不是实用模型，而是最低基线。如果真正的模型连它都无法超过，说明数据质量或特征设计可能存在问题。

`ridge` 是带 L2 正则化的稳定线性模型。它训练快、小数据集表现稳，适合学习资源配置、KPI 指标与最大负载之间的主要线性关系。

`mlp` 是神经网络回归模型，可以学习非线性关系，但它需要更多数据；数据较小时容易过拟合。

### Adaptive Hybrid 模型

本项目的专属优化模型是 `adaptive_hybrid`。

它采用两步训练方式：

```text
Ridge 基础模型：先学习稳定的主要关系
MLP 补充模型：再学习 Ridge 仍然预测不准的部分
自动回退：当数据太少或验证集没有明确提升时，只使用 Ridge
```

训练时，系统会先比较“只用 Ridge”和“Ridge 加 MLP 补充”在验证集上的 RMSE。只有当两个 RMSE 都能正常计算，并且组合后的 RMSE 至少达到 `min_improvement` 要求时，才会启用 MLP 补充模型；否则保存更稳妥的 Ridge 模型。这样可以避免为了追求复杂模型而牺牲小数据集上的稳定性。

`metrics.json` 中会记录以下诊断字段，便于汇报时说明模型为什么启用或没有启用 MLP 补充：

```text
residual_gate_reason    启用判断结果：达到要求 / 提升不足 / 指标异常 / 样本不足
residual_improvement    组合模型相对 Ridge 的 RMSE 降低比例；无法安全计算时为 null
min_improvement         启用 MLP 补充所需的最低改进比例
validation_size         内部验证集比例
training_rows           用于判断是否启用 MLP 补充的训练行数；小数据回退时为总训练行数
validation_rows         用于判断是否启用 MLP 补充的验证行数；小数据回退时为 0
```

## 模型文件如何使用

训练输出：

```text
load_predictor.joblib
metrics.json
```

`load_predictor.joblib` 是保存好的 scikit-learn Pipeline，包含特征标准化和最终选中的模型。因此使用模型预测时不需要手动做标准化。

示例：

```python
import joblib
import pandas as pd

model = joblib.load("load_predictor.joblib")

features = pd.DataFrame([
    {
        "traffic_input_mbps": 700,
        "cpu_cores": 4,
        "ram_gb": 8,
        "link_capacity_mbps": 1000,
        "cpu_utilization_percent": 65,
        "memory_utilization_percent": 70,
        "latency_ms": 25,
        "throughput_mbps": 680,
        "packet_loss_percent": 0.3,
    }
])

prediction = model.predict(features)
print(prediction[0])
```

只加载由本项目或可信来源生成的模型文件。`joblib` 基于 pickle 序列化，加载不可信文件可能执行不安全代码。

输出结果就是预测的 `max_supported_load_mbps`。

## 应用场景

该模型可以服务于后续优化与编排阶段：

```text
1. 编排器收集当前流量、资源配置和 KPI 指标。
2. ML 模型预测 max_supported_load_mbps。
3. 优化模块比较当前流量和预测容量。
4. 如果当前流量超过预测容量，系统可以扩容 CPU/RAM/带宽，或重新调度服务。
```

## 术语解释

```text
ML                    机器学习。这里指用历史 profiling 数据训练模型，让模型预测网络服务的最大可支持负载。
profiling             性能采样或压测记录。每一行表示一次服务运行实验，包括资源、流量、KPI 和最大可支持负载。
KPI                   关键性能指标。这里主要包括 CPU 使用率、内存使用率、延迟、吞吐量和丢包率。
CSV                   逗号分隔表格文件。项目用 CSV 上传训练数据。
schema                数据字段规范。这里指训练 CSV 必须包含哪些列，以及这些列的含义。
feature / 特征        模型输入列，例如 CPU 核数、内存、链路容量、延迟、吞吐量和丢包率。
target / 目标值       模型要预测的列。本项目的目标值是 max_supported_load_mbps。
Mbps                  兆比特每秒。这里用于表示流量、吞吐量和最大可支持负载。
GB                    吉字节。这里用于表示 RAM 内存大小。
ms                    毫秒。这里用于表示网络延迟。
CPU / RAM             CPU 表示处理器资源；RAM 表示内存资源。它们是预测服务负载能力的重要输入。
Regression / 回归     预测连续数值的机器学习任务。本项目预测的是最大可支持负载，不是类别。
validation set / 验证集 从训练数据中分出来的一部分数据，用来比较候选模型表现，并选择最终保存的模型。
holdout               一次固定的数据划分方式。项目默认用 holdout 验证集选择最终模型。
K-fold cross-validation / K 折交叉验证  把数据分成 K 份轮流训练和验证；本项目中它只写入报告，不改变最终模型选择。
MAE                   平均绝对误差。表示预测值平均偏离真实值多少，越低越好。
RMSE                  均方根误差。它对较大的预测错误更敏感，项目用它作为主要模型选择指标，越低越好。
R²                    决定系数，用来辅助判断模型整体解释能力。越接近 1 通常表示拟合越好；数据太少时可能无法有效计算。
Ridge                 岭回归模型。它是带 L2 正则化的线性回归，训练稳定，适合小数据和主要线性关系。
L2 regularization / L2 正则化  一种限制模型参数过大的方法，用来降低过拟合风险，让模型更稳定。
MLP                   多层感知机神经网络。它可以学习非线性关系，但通常需要更多数据。
adaptive_hybrid       本项目的自适应混合模型。它先训练 Ridge，再判断是否需要启用 MLP 补充模型。
min_improvement       启用 MLP 补充模型所需的最低 RMSE 降低比例。提升不够时，模型会保留更稳妥的 Ridge 结果。
CLI                   命令行界面。这里指通过 python -m ml_project.train 运行训练。
Streamlit             用 Python 构建网页应用的框架。本项目用它提供上传数据、训练模型、查看结果的前端界面。
artifact              训练后生成并保存的文件，例如模型文件、指标 JSON 和上传的数据集副本。
joblib                Python 模型保存工具。项目用 .joblib 文件保存训练好的 scikit-learn Pipeline。
Pipeline              scikit-learn 的流水线对象。本项目的 Pipeline 同时包含特征标准化和最终模型。
JSON                  一种结构化文本格式。项目用 metrics.json 保存训练指标和诊断信息。
SQLite / PostgreSQL   数据库。SQLite 用于本地默认训练历史；PostgreSQL 用于 Zeabur 部署环境。
Zeabur                云部署平台。项目提供 Zeabur 配置，用于部署 Streamlit 应用和持久化训练文件。
facade                兼容门面模块。这里指 train.py 和 streamlit_helpers.py 继续保留旧导入路径，但内部实现已经拆到更清晰的模块。
```

## 本地环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

Zeabur 生产部署可使用：

```text
requirements.txt
```

## 运行测试

```bash
pytest
ruff check .
python -m compileall -q src tests
```

## 命令行训练

仓库内置了一个 3 行 CSV 模板，可以直接用于冒烟测试：

```bash
python -m ml_project.train \
  --data data/profiling_dataset_template.csv \
  --output-dir artifacts \
  --allow-small-dataset \
  --models dummy_mean ridge adaptive_hybrid
```

正式训练时，请先准备符合 schema 的较大 `data/profiling_dataset.csv`，再运行：

```bash
python -m ml_project.train \
  --data data/profiling_dataset.csv \
  --output-dir artifacts
```

如果希望额外查看 K 折交叉验证指标，可以添加 `--cv-folds`：

```bash
python -m ml_project.train \
  --data data/profiling_dataset.csv \
  --output-dir artifacts \
  --cv-folds 5
```

交叉验证是 report-only：它只把每个模型的 K 折 MAE/RMSE/R² 均值和标准差汇总写入 `metrics.json` 的 `cross_validation` 字段，不会替代默认 holdout 验证集，也不会改变最终保存模型的选择逻辑。`--cv-folds` 必须至少为 2，且不能超过数据行数。

## Streamlit 前端界面

本地运行：

```bash
PYTHONPATH=src streamlit run src/ml_project/streamlit_app.py
```

前端支持：

```text
中英文切换
CSV 上传与校验
数据预览与可视化
模型训练与指标展示
SQLite/PostgreSQL 训练历史
下载模型、指标和上传数据集
删除训练记录和 artifact 文件
```

## Zeabur 部署

该 Streamlit 应用可以部署到 Zeabur，并接入 PostgreSQL 与持久化存储卷。

1. 在 Zeabur 创建项目并连接本仓库。
2. 添加 PostgreSQL 服务。
3. 在应用服务中设置环境变量：

```text
DATABASE_URL=postgresql+psycopg://...
ARTIFACT_ROOT=/data/training_runs
```

4. 将持久化 Volume 挂载到 `/data`，确保上传的 CSV、模型和指标文件在重启后仍然保留。
5. 部署最新的 `main` 分支。

`zbpack.json` 的启动命令设置了 `PYTHONPATH=src`，因为项目使用 `src/` 包结构，并直接从 `src/ml_project/streamlit_app.py` 启动 Streamlit。

## 项目结构

```text
data/                              CSV 模板和未来数据集
docs/                              数据集 schema 文档
src/ml_project/
  schema.py                        共享 schema 常量（特征列、目标列、候选模型）
  dataset.py                       数据集加载与校验
  model_factory.py                 候选模型工厂（build_model）
  evaluation.py                    模型评估与 K 折交叉验证
  pipeline.py                      训练 pipeline 核心编排
  adaptive_model.py                Adaptive Hybrid 模型
  training_types.py                训练结果与指标数据类
  train.py                         兼容 facade 与命令行入口
  artifacts.py                     artifact 路径与 run 目录管理
  training_history.py              SQLite/PostgreSQL 训练历史
  streamlit_helpers.py             UI helper 兼容 facade
  ui/                              Streamlit helper 实现（dataframe、downloads、formatting）
  streamlit_app.py                 双语 Streamlit 前端
  i18n.py                          中英文翻译
tests/
  factories.py                     共享测试数据工厂
  test_*.py                        测试套件
requirements.txt                   Zeabur 运行时依赖
zbpack.json                        Zeabur 构建/启动配置
.env.example                       环境变量模板
```

`train.py` 与 `streamlit_helpers.py` 是兼容 facade，把 `schema`/`dataset`/`model_factory`/`evaluation`/`pipeline` 与 `ui/` 下的实现重新导出，保持旧的导入路径可用。命令行训练通过 `python -m ml_project.train` 调用。
