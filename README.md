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

它结合了 Ridge 和 MLP 的优点：

```text
Ridge 主干：学习稳定的主要趋势
MLP 残差修正：学习 Ridge 没有捕捉到的非线性误差
自动回退：当数据太少或验证集没有提升时，自动关闭残差修正
```

核心思想可以理解为：

```text
final_prediction = ridge_prediction + residual_weight * mlp_residual_prediction
```

实际训练时，模型会先验证残差修正是否能降低 RMSE。如果没有提升，就自动回退到 Ridge。这样既能兼容小型演示数据，也能在真实数据较多时利用神经网络学习非线性规律。

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
data/                         CSV 模板和未来数据集
docs/                         数据集 schema 文档
src/ml_project/adaptive_model.py       Adaptive Hybrid 模型
src/ml_project/train.py                训练 pipeline
src/ml_project/streamlit_app.py        双语 Streamlit 前端
src/ml_project/training_history.py     SQLite/PostgreSQL 训练历史
tests/                        测试套件
requirements.txt              Zeabur 运行时依赖
zbpack.json                   Zeabur 构建/启动配置
.env.example                  环境变量模板
```
