# RSNA Knee Abnormality Detection

[English](README.md) | [简体中文](README.zh-CN.md)

面向膝关节多序列 MRI 的 PyTorch 分类框架，预测 **12 项异常**。提供 DICOM 预处理、解剖定位多视图、患者分组五折训练、软标签自蒸馏，以及 [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) **第 21 名、单人金牌方案**使用的模型权重。

最终提交的 **Public Score 为 0.964，Private Score 为 0.955**。这是原竞赛推理流程的成绩；本仓库将其权重和输入配方整理为可复用实现。

## 安装

建议使用 Python 3.10+ 和支持 CUDA 的 PyTorch。先安装适合显卡的 PyTorch 版本，再运行：

```bash
git clone https://github.com/zhuzhirui2005/RSNA-Knee-Abnormality-Detection.git
cd RSNA-Knee-Abnormality-Detection
pip install -e .
knee-mri download --output weights
```

分类权重存放在 [Releases](https://github.com/zhuzhirui2005/RSNA-Knee-Abnormality-Detection/releases)，两个辅助预处理模型从原 Kaggle 数据集下载。完整权重约 2.4 GB，下载器校验 SHA-256。不同权重的来源和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 运行获奖模型

输入采用竞赛解压后的目录结构：

```text
data/rsna/
├── test.csv
├── test_series.csv
└── test_series/
    └── <StudyInstanceUID>/
        └── <SeriesInstanceUID>/
            ├── 1.dcm
            └── ...
```

`test.csv` 包含 `StudyInstanceUID`。`test_series.csv` 包含 `StudyInstanceUID`、`SeriesInstanceUID`、`Anatomical_Plane`、`Fat_Suppression` 和 `Fluid_Sensitive`。使用竞赛原始元数据，平面名称为 `Sagittal`、`Coronal`、`Axial`。**推理不读取文字报告或标签。**

```bash
knee-mri predict \
  --data data/rsna --split test \
  --weights weights --device cuda:0 \
  --cache cache/test --output outputs/submission.csv
```

一张显卡即可依次运行全部模型。默认每批编码 64 个窗口；显存较小时可设 `--chunk 32`，只改变计算批量，不减少实际读取的窗口。`--cache` 保留渲染缓存供后续复用；省略时使用临时目录。较大队列需要为渲染缓存预留磁盘空间。

输出每例一行，包含病例编号及以下 12 项概率：ACL、MCL、Medial Meniscus、Lateral Meniscus、Medial OA、Lateral OA、PF OA、Effusion、Synovitis、Baker's、Contusion、Fracture。所有分支完成后才写入最终 CSV；失败时不会将部分融合结果发布到最终输出路径。

只运行某个分支时，可添加 `--branch llm`、`--branch distilled` 或 `--branch public`。

## 模型与融合

三个分支均使用 **ConvNeXt-Small**。三个相邻 MRI 切片组成一个输入窗口，类别专属空间注意力聚合窗口内特征，第二层注意力聚合整例的多序列窗口。两个五折分支还包含上游架构中的骨关节炎范围分支。

| 分支 | 模型数 | 训练标签 | 渲染尺寸 / 中心裁剪 | 融合比例 |
|---|---:|---|---|---:|
| LLM 标签续训 | 5 | 基础训练后，继续使用 LLM 概率以较小学习率训练两轮 | 384 / 352 | 45% |
| 自蒸馏学生 | 5 | 75% LLM 概率 + 25% 对应折教师概率，从 ImageNet 权重初始化 | 384 / 352 | 45% |
| 公开 ConvNeXt-Small | 1 | [nartaa 发布的权重](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) | 320 / 288 | 10% |

每个五折分支先对 **logit 求均值，再进行 sigmoid**，三个分支最终按概率融合：

```python
p = 0.45 * sigmoid(mean(llm_fold_logits, axis=0)) \
  + 0.45 * sigmoid(mean(student_fold_logits, axis=0)) \
  + 0.10 * sigmoid(public_logits)
```

每例包含全视野及解剖定位的 100 mm 局部视图，窗口内切片物理间距约 3.5 mm，推理覆盖物理步长选出的全部中心。384 像素分支使用图像左右侧校正；320 像素公开分支保留其原始预处理，不使用这项校正。两种分辨率都从原始 DICOM 渲染，不将一个分支的缓存缩放给另一个分支。

发布内容包含 11 个分类模型。另有 SKM-TEA VNet 用于解剖定位、ResNet34 用于方向检查，两者属于预处理模型。五折权重中未启用的报告状态预测头已移除。

## 准备训练数据

训练数据使用对应的 `train.csv`、`train_series.csv` 和 `train_series/` 目录。先构建一次多视图缓存：

```bash
knee-mri prepare --data data/rsna --split train \
  --weights weights --size 384 --device cuda:0 --cache cache/train384
```

每例独立保存为 NPZ，使用横向 uint8 差分和 Zstandard 对**已渲染视图无损压缩**。从 DICOM 到视图的缩放与强度编码仍属于预处理，缓存不能无损替代原始 DICOM。

标签 CSV 使用 UTF-8 编码：

| 字段 | 含义 |
|---|---|
| `StudyInstanceUID` | 与影像元数据一致的唯一病例编号 |
| `group` | 患者／报告分组编号 |
| `fold` | 固定折编号，一般为 0–4 |
| 上述 12 项异常名称 | 0–1 的概率；未知目标留空 |

同组病例必须属于同一折，缺失目标不参与损失。官方 gold 和外部验证病例单独保存，不放入训练表。[examples/labels.csv](examples/labels.csv) 仅为合成数据格式示例。

发布模型的两个五折分支使用 4,349 例训练。仓库不分发影像、临床报告、患者编号、原始折划分或标注表。需要生成自己的标注时，可使用独立的 [RSNA Knee Reader](https://github.com/zhuzhirui2005/rsna-knee-reader) MCP 服务。

## 训练与自蒸馏

### 1. 训练基础教师

```bash
knee-mri train --config configs/train_llm.json \
  --labels data/labels.csv --cache cache/train384 \
  --fold 0 --device cuda:0 --output outputs/base/fold0
```

对折 1–4 分别运行，每折使用独立输出目录；并行时分配不同显卡。每个模型使用其他四折训练。默认采用 ImageNet-22K → ImageNet-1K 初始化、每例最多 176 个物理窗口、同例共享随机 352 像素裁剪、软标签 BCE、AdamW 和梯度累积。编码器／分类头学习率为 `1e-4`／`1e-3`。

验证保持原始留出概率，用 `> 0.5` 得到计算 AUC 所需的阳性标签。`best.pt` 保存宏 AUC 最高的轮次。至少训练 8 轮，连续 3 轮未超过 `0.0005` 的改善幅度时早停，最多 16 轮。针对 LLM 标签的 AUC 反映弱标签一致性，不是独立临床准确率。

原训练使用本折训练分区内的报告共识阳性率初始化分类器偏置。提供 `--bias-labels data/consensus.csv` 可保留这一设置；否则使用当前训练标签的均值。

### 2. LLM 标签续训

```bash
knee-mri train --config configs/continue_llm.json \
  --labels data/labels.csv --cache cache/train384 \
  --fold 0 --initial outputs/base/fold0/best.pt \
  --output outputs/llm/fold0
```

重新初始化优化器，以 `1e-5`／`1e-4` 的学习率续训两轮。该分支选择固定第二轮的 **`last.pt`**，与发布权重一致。蒸馏标签由基础教师生成，不使用这份续训权重生成。

### 3. 生成对应折的教师标签

```bash
knee-mri teacher --labels data/labels.csv --fold 0 \
  --checkpoint outputs/base/fold0/best.pt --cache cache/train384 \
  --output outputs/teacher/fold0.csv
```

教师 0 仅预测学生 0 的训练分区，两者都排除 fold 0。教师预测在训练分区内属于训练集预测，**不是 OOF 标签**；留出集标签保持不变。

### 4. 从 ImageNet 训练蒸馏学生

```bash
knee-mri train --config configs/train_llm.json \
  --labels data/labels.csv --cache cache/train384 \
  --fold 0 --teacher outputs/teacher/fold0.csv --teacher-weight 0.25 \
  --output outputs/student/fold0
```

不传 `--initial`，学生独立从 ImageNet 权重初始化。五折分别执行上述步骤，学生使用各自的 `best.pt` 进行推理。

### 恢复训练与评价

```bash
knee-mri train --config configs/train_llm.json \
  --labels data/labels.csv --cache cache/train384 \
  --fold 0 --output outputs/base/fold0 --resume

knee-mri evaluate --labels data/labels.csv \
  --predictions outputs/base/fold0/best_validation.csv
```

恢复时保持原配置和参数。`resume.pt` 保存最近完成的轮次、优化器、学习率调度器和随机状态，`best_validation.csv`、`history.json` 保存留出预测和逐类指标。仓库提供可用的训练配方；从头复现已发布权重还需要未随仓库分发的原始标注快照与划分。

## 目录结构

```text
knee_mri/
  model.py               ConvNeXt 与整例注意力
  preprocessing/         DICOM 方向、解剖定位和多视图渲染
  data.py, sampling.py    压缩缓存与物理窗口采样
  training.py             分组训练与软标签蒸馏
  inference.py            完整病例推理与固定融合
  resources/weights.json  下载地址、校验值和许可
configs/                  基础训练与续训配方
examples/                 合成输入格式示例
tests/                    几何、采样、融合与训练检查
```

## 来源与许可

模型架构与解剖预处理源自 [Heliosli 的公开方案](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click)，公开融合模型由 [nartaa](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) 提供。解剖定位使用 [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) 权重，VNet 转换模块改编自 MONAI 0.6。

框架代码和新发布的十个分类权重采用 Apache-2.0。nartaa 权重采用 **CC BY-NC 4.0**，保留非商业使用限制；辅助权重保留上游许可。详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [THIRD_PARTY.md](THIRD_PARTY.md)。本项目面向研究和竞赛，不是经过临床验证的诊断系统。
