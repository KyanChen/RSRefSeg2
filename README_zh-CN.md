<div align="center">
    <h2>
        RSRefSeg 2: Decoupling Referring Remote Sensing Image Segmentation with Foundation Models
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/RSRefSeg2.png" width="800"/>
</div>
<br>
<div align="center">
  <a href="https://github.com/KyanChen/RSRefSeg2">
    <span style="font-size: 20px;">项目主页</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2507.06231">
    <span style="font-size: 20px;">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/RSRefSeg2.pdf">
    <span style="font-size: 20px;">PDF</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/RSRefSeg2)](https://github.com/KyanChen/RSRefSeg2)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2507.06231-b31b1b.svg)](https://arxiv.org/abs/2507.06231)

<br>
<br>

<div align="center">

[English](README.md) | 简体中文

</div>


## 简介

本项目仓库是论文 [RSRefSeg 2: Decoupling Referring Remote Sensing Image Segmentation with Foundation Models](https://arxiv.org/abs/2507.06231) 的代码实现，基于 [OpenMMLab](https://openmmlab.com/codebase) 代码库进行开发。

当前分支在 Linux 系统，PyTorch 2.x 和 CUDA 12.1 下测试通过，支持 Python 3.10+，能兼容绝大多数的 CUDA 版本。

如果你觉得本项目对你有帮助，请给我们一个 Star ⭐️，你的支持是我们最大的动力。

<details open>
<summary>主要特性</summary>

- 与 OpenMMLab 高度保持一致的 API 接口及使用方法
- 开源了论文中不同数据集上的模型和权重
- 支持模型的训练和测试

</details>

## 更新日志

<details>

🌟 **2025.07.10** 发布了 RSRefSeg2 项目。

</details>


## 目录

- [简介](#简介)
- [更新日志](#更新日志)
- [目录](#目录)
- [安装](#安装)
- [数据集准备](#数据准备)
- [模型训练测试](#模型训练测试)
- [模型权重下载](#模型权重下载)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [联系我们](#联系我们)



## 安装

### 依赖项

- Linux 系统（Windows 也支持）
- Python 3.10+（推荐 3.11）
- PyTorch 2.0 或更高版本（推荐 2.4）
- CUDA 11.7 或更高版本（推荐 12.1）
- MMCV 2.0 或更高版本（推荐 2.2）

### 环境安装

推荐使用 Miniconda 进行环境管理和安装。以下命令将创建名为 `rsrefseg2` 的虚拟环境，并安装 PyTorch 和 MMCV。默认安装的 CUDA 版本为 **12.1**，请根据您的实际 CUDA 版本进行相应调整。

**注意**：如果您已经熟悉 PyTorch 并已安装，可以直接跳至下一小节。否则，请按照以下步骤进行准备。

<details>

**步骤 0**：安装 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)。

**步骤 1**：创建并激活名为 `rsrefseg2` 的虚拟环境。

```shell
conda create -n rsrefseg2 python=3.11 -y
conda activate rsrefseg2
```



**步骤 2**：安装 [PyTorch2.4.x](https://pytorch.org/get-started/previous-versions/)。

```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# 或者
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**步骤 3**：安装 [MMCV2.2.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

```shell
pip install -U openmim
mim install mmcv==2.2.0
# 或者
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

**步骤 4**：安装其他依赖项。

```shell
pip install deepspeed==0.17.2  # Windows系统不支持DeepSpeed训练，不需要安装，使用AMP混合精度训练
pip install transformers==4.53.1 datasets
pip install -U ipdb braceexpand mat4py pycocotools shapely ftfy scipy terminaltables wandb prettytable torchmetrics importlib_metadata einops peft
pip install hydra-core iopath
```

</details>


### 安装 RSRefSeg2

通过 Git 下载或克隆 RSRefSeg2 仓库：

```shell
git clone git@github.com:KyanChen/RSRefSeg2.git
cd RSRefSeg2
```

## 数据准备

### 数据下载

- [RefSegRS](https://huggingface.co/datasets/JessicaYuan/RefSegRS/tree/main)
- [RRSIS-D](https://github.com/Lsan2401/RMSIN)
- [RISBench](https://github.com/hit-sirs/crobim)

### 数据组织方式

将下载的数据解压，并按原文件夹的组织方式放置。

**注**：在项目文件夹 `datainfo` 中，我们提供了数据集转换后的划分文件。您也可以使用 [Python 脚本](tools_RSRefSeg2/data_tools) 进行数据集转换。

## 模型训练测试

### Config 文件及主要参数解析

我们提供了论文中不同数据集的 RSRefSeg2 模型配置文件，您可以在 [配置文件](configs_RSRefSeg2) 文件夹中找到它们。Config 文件完全与 OpenMMLab 保持一致的 API 接口及使用方法。下面我们提供了主要参数的解析。如需了解更多参数含义，请参考 [OpenMMLab 相关文档](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html)。

<details>

**参数解析**：

- `work_dir`：模型训练的输出路径，一般不需要修改。
- `data_root`：数据集根目录，**修改为数据集根目录的绝对路径**。
- `cache_dir`: 模型缓存目录，**修改为需要缓存模型目录的绝对路径**。
- `batch_size`：单卡的 batch size，**需要根据显存大小进行修改**。
- `max_epochs`：最大训练轮数，一般不需要修改。
- `val_interval`：验证集的间隔轮数，一般不需要修改。
- `dst_size`: CLIP 模型特征图像的大小，**需要根据实际情况进行修改**。
- `vis_backends/WandbVisBackend`：网络端可视化工具的配置，**打开注释后，需要在 `wandb` 官网注册账号，可在网络浏览器中查看训练过程的可视化结果**。
- `load_from`：模型预训练的检查点路径，一般不需要修改。
- `resume`: 是否断点续训，一般不需要修改。
- `default_hooks/CheckpointHook`：模型训练过程中的检查点保存配置，一般不需要修改。
- `default_hooks/visualization`：将 `draw` 设置为 `True` 可以在验证集和测试集上进行可视化，`interval` 设置为可视化的间隔轮数，一般不需要修改。
- `model/lora_cfg`：模型视觉骨干的 LoRA 配置，一般不需要修改。
- `model/backbone`：模型的视觉骨干，一般不需要修改。
- `model/prompter`：模型的提示器配置，一般不需要修改。
- `model/decode_head`：模型的解码头配置，一般不需要修改。
- `AMP training config`：混合精度训练的配置，如果不使用 DeepSpeed 训练，则打开注释，一般不需要修改。
- `DeepSpeed training config`：DeepSpeed 训练的配置，如使用 DeepSpeed 训练，则打开注释，将 AMP training config 注释掉，注意 Windows 系统不支持 DeepSpeed 训练。
- `data_preprocessor/mean/std`：数据预处理的均值和标准差，一般不需要修改。

</details>

### 训练

```shell
# 单卡训练
python tools_mmseg/train.py configs_RSRefSeg2/name_to_config.py  # name_to_config.py 为你想要使用的配置文件
# 多卡训练
sh tools_mmseg/dist_train.sh configs_RSRefSeg2/name_to_config.py ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，GPU_NUM 为使用的 GPU 数量
```

### 测试

如需在测试过程中可视化并保存预测结果图片，请在配置文件中将 `default_hooks/visualization` 下的 `draw` 参数设置为 `True`。

```shell
# 单卡测试
python tools_mmseg/test.py configs_RSRefSeg2/name_to_config_infer.py ${CHECKPOINT_FILE}  # name_to_config_infer.py 为您要使用的配置文件，CHECKPOINT_FILE 为您要评估的模型检查点路径

# 多卡测试
sh tools_mmseg/dist_test.sh configs_RSRefSeg2/name_to_config_infer.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config_infer.py 为您要使用的配置文件，CHECKPOINT_FILE 为您要评估的模型检查点路径，GPU_NUM 为您要使用的 GPU 数量
```

## 模型权重下载

您可以在 [Hugging Face](https://huggingface.co/KyanChen/RSRefSeg2) 平台上获取并下载预训练模型权重。

## 常见问题

<details>

我们在此列出了使用过程中可能遇到的一些常见问题及其解决方案。如果您发现有未涵盖的问题，欢迎提交 PR 来丰富此列表。如果您在此处未能找到所需帮助，请通过 [GitHub Issues](https://github.com/KyanChen/RSRefSeg2/issues) 提交问题。请务必填写所有必要信息，这将有助于我们更快地定位和解决问题。

### 1. 是否需要安装MM系列包？

我们强烈建议您不要安装MM系列包（如MMSeg），因为本项目已经包含了所有必需的组件。安装额外的MM系列包可能导致代码运行冲突。如果您遇到"模块尚未被注册"的错误，请按以下步骤检查：

- 确认该模块是否为需要安装的外部依赖，如是则进行安装
- 检查是否安装了MM系列包，若已安装则卸载
- 确认相关类名前是否添加了`@MODELS.register_module()`装饰器
- 检查`__init__.py`文件中是否包含了`from .xxx import xxx`导入语句
- 验证配置文件中是否添加了`custom_imports = dict(imports=['refseg'], allow_failed_imports=False)`配置项

### 2. 解决dist_train.sh: Bad substitution错误

如果在执行`dist_train.sh`脚本时遇到`Bad substitution`错误，请尝试使用`bash dist_train.sh`命令来运行脚本。

### 3. Hugging Face模型下载问题解决方案

如果您在下载Hugging Face模型时遇到困难，请尝试在命令行中设置以下环境变量：

```shell
export HF_ENDPOINT=https://hf-mirror.com
```
</details>


## 致谢

本项目基于 [OpenMMLab](https://openmmlab.com/codebase) 生态系统开发，我们衷心感谢 OpenMMLab 社区的所有贡献者。

## 引用

如果您在研究或项目中使用了本代码或性能基准，请使用以下 BibTeX 引用我们的工作：

```
@article{chen2025rsrefseg,
  title={Rsrefseg: Referring remote sensing image segmentation with foundation models},
  author={Chen, Keyan and Zhang, Jiafan and Liu, Chenyang and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2501.06809},
  year={2025}
}

@article{chen2025rsrefseg2,
  title={RSRefSeg 2: Decoupling Referring Remote Sensing Image Segmentation with Foundation Models},
  author={Chen, Keyan and Liu, Chenyang and Chen, Bowen and Zhang, Jiafan and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2507.06231},
  year={2025}
}
```


## 开源许可证

本项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 联系我们

如有任何疑问或建议❓，请随时与我们团队联系 👬

