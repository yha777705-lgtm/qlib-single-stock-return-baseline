# 基于 Qlib 数据的单股票收益预测基线

使用 Qlib 读取行情数据，构建简单特征，通过时间划分比较不同模型的预测误差，并记录可复现的实验过程。

## 预测任务

使用 SH600000 在交易日 t 收盘后可获得的数据，预测下一个交易日的收盘到收盘收益率：

```python

target_return_1d = close.shift(-1) / close - 1

```

这是离线预测任务，尚未建立可执行的交易策略。

## 特征

- return_1d：过去1个交易日的收益率。

- return_5d：过去5个交易日的收益率。

- volume_ratio_5d：当日成交量除以包含当日在内的5日平均成交量，再减1。

## 数据与划分

数据来自 SunsetWolf/qlib_dataset 的 v2 发布包：

qlib_data_simple_cn_1d_0.9.zip。

本实验只使用 SH600000 在2010—2020年的数据。

原始2674行，清洗后2546行。

按时间顺序划分，并删除标签跨越划分边界的2条样本：

| 数据集 | 特征日期范围 | 样本数 |
|---|---|---:|
| 训练集 | 2010-01-11 至 2016-12-29 | 1587 |
| 验证集 | 2017-01-03 至 2018-12-27 | 479 |
| 测试集 | 2019-01-02 至 2020-12-30 | 478 |

测试集尚未用于模型性能评估。

## 验证集结果

主要指标为 RMSE，辅助指标为 MAE。

下表单位为收益率的百分点，即原始误差乘以100。

| 模型 | MAE | RMSE |
|---|---:|---:|
| ZERO | 0.8475 | 1.2463 |
| TRAIN_MEAN | 0.8553 | 1.2500 |
| RIDGE | 0.8595 | 1.2530 |
| LIGHTGBM_FIXED | 0.8590 | 1.2559 |

ZERO 始终预测0，TRAIN_MEAN 始终预测训练集标签均值。

当前验证集上，Ridge 和固定参数 LightGBM 均未超过零预测基线。

LightGBM 在训练集上的改善没有延续到该验证时期；

原因尚未确定，不能仅据此断定过拟合。

### 验证集按年诊断

沿用既有预测，按特征日期的年份分组；没有重新训练或调整模型。RMSE 单位为百分点。

| 年份 | 样本数 | ZERO RMSE | LightGBM RMSE | LightGBM 相对零预测的 RMSE 改善 |
|---|---:|---:|---:|---:|
| 2017 | 237 | 1.1030 | 1.1108 | -0.7032% |
| 2018 | 242 | 1.3723 | 1.3833 | -0.8020% |

负改善表示误差升高。这两个年度切片中，LightGBM 均未超过零预测。
这只是描述性结果，未检验差异的不确定性，不能据此认定过拟合或解释失败原因。

生成原有验证集预测后，在项目根目录运行 python analyze_validation_by_year.py。
完整的四模型分年结果见 reports/validation_by_year.json；本项脚本与诊断由 Codex 代为完成。

### 固定 Ridge 的特征消融

固定 StandardScaler + Ridge(alpha=1.0)，每次去掉一个特征；四组使用完全相同的训练1587行和验证479行。标准化只在训练集拟合。RMSE 单位为百分点。

| 配置 | 验证集 RMSE |
|---|---:|
| 零预测（对照） | 1.2463 |
| Ridge：全部三个特征 | 1.2530 |
| Ridge：去掉 return_1d | 1.2530 |
| Ridge：去掉 return_5d | 1.2528 |
| Ridge：去掉 volume_ratio_5d | 1.2500 |

表中采用四位小数展示，完整精度见报告。去掉成交量指标的配置在四组 Ridge 中误差最低，但仍未超过零预测。
该比较限于当前模型和验证期，没有不确定性检验，不能证明某个特征普遍无效或存在因果影响，也没有据此确定最终模型。

生成数据划分和原始基线报告后，运行 python ablate_ridge_features.py。
结果见 reports/ridge_feature_ablation.json；本项脚本与分析由 Codex 代为完成，测试集未评估。

### 原训练期内的两折时间验证

在原2010—2016年训练期内部做两个预先固定的扩展窗口。每折新建 StandardScaler + Ridge(alpha=1.0)，仅用该折训练样本拟合标准化和回归；跨边界标签按原规则剔除。训练与评分没有使用2017—2018年的原验证样本或测试样本。

| 验证年 | 训练行数 | 验证行数 | ZERO RMSE（pp） | Ridge RMSE（pp） | 相对改善 |
|---|---:|---:|---:|---:|---:|
| 2015 | 1149 | 217 | 2.7507 | 2.7667 | -0.5815% |
| 2016 | 1367 | 219 | 1.1314 | 1.1251 | +0.5646% |
| 合并折外预测 | — | 436 | 2.0997 | 2.1084 | -0.4152% |

2016折略优于零预测，2015折略差，合并后仍略差。两折结果只作历史诊断，尚不能证明稳定优势；未进行网格搜索、不确定性检验或最终选模。
合并指标直接由436条不重复的折外预测计算，不是各折 RMSE 的算术平均。

运行 python validate_ridge_over_time.py；报告见 reports/ridge_time_validation.json，其中记录了每折训练日期、边界剔除、标准化拟合行数及均值与尺度。本项由 Codex 实现并运行，测试集未评估。

## 环境与数据路径

已验证环境：Windows、Python 3.11.18、Qlib 0.9.7。

requirements.txt 记录当前环境的包版本，

属于环境快照，不保证跨平台安装成功。

## 复现方法

### 1. 安装环境

在项目根目录打开 Anaconda PowerShell Prompt：

```powershell
conda create -n qlib-reproduce python=3.11 -y
conda activate qlib-reproduce
python -m pip install -r requirements.txt
```

### 2. 准备数据

使用浏览器下载社区数据包：

https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_simple_cn_1d_0.9.zip

解压到当前用户 Downloads 目录下的
qlib_data_simple_cn_1d_0.9 文件夹。
该文件夹内应直接包含 calendars 和 features。

如果使用其他位置，可以设置环境变量。以下路径仅为示例，
运行时应替换成自己的实际数据目录：

```powershell
$env:QLIB_DATA_DIR = "D:\datasets\qlib_data_simple_cn_1d_0.9"
```

### 3. 按顺序运行

在项目根目录、已激活的环境中执行：

```powershell
python read_data.py
python build_dataset.py
python split_dataset.py
python train_baselines.py
python train_lightgbm.py
python diagnose_model.py
```

生成的数据位于 data，模型文件位于 artifacts，
实验报告位于 reports。

已于2026-11-09在同一台 Windows 电脑上，使用新建 Conda 环境和干净源码副本完成数据构建、时间划分、训练及诊断。

10个数据、报告和验证集预测文件的 SHA-256 均与原实验一致，证据见 reports/reproduction_check.json。
本次复用了已有原始数据，未重新验证下载，也未验证其他电脑或平台。
当前训练与诊断脚本评估训练集和验证集，尚未评估测试集。
config.py 统一管理数据路径。

默认读取当前用户 Downloads 下的 qlib_data_simple_cn_1d_0.9 目录，

也可以通过环境变量 QLIB_DATA_DIR 指定其他位置。

数据根目录应包含 calendars 和 features 两个子目录。

原始数据、生成的 CSV 和模型文件不纳入 Git 版本管理。

## 项目边界

- 单只股票、三个简单特征的学习实验。

- 尚未复现 Qlib 官方 Alpha158 基准。

- 尚未进行交易回测，也没有盈利能力结论。

- 当前结果不能推广到其他股票或市场时期。

- 详细实验设置见 EXPERIMENTS.md。

## AI 辅助说明

初始脚本由 AI 辅助搭建；

本人执行实验、修改配置、检查输出，并学习解释实验结果。

2026-11-09 的本机新环境复现由 Codex 经本人授权代为执行与核对，记录见 reports/reproduction_check.json。
