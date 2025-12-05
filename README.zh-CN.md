<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**项目简介**
---
本项目基于 [HundredCV-Chat](https://huggingface.co/datasets/Jax-dan/HundredCV-Chat)
或[HundredCV-Chat (镜像)](https://hf-mirror.com/datasets/Jax-dan/HundredCV-Chat)多模态对话数据集，使用 **PyTorch** 和
**RNN** 神经网络架构，实现智能输入法中的下一个**词预测**（Next Word
Prediction）功能。通过分析简历图片相关的对话文本，模型能够学习语言模式，为用户输入提供智能补全建议。


**功能特色**
---

**🔧 1. 一键模型初始化**

- 自动加载保存的 RNN 模型参数
- 自动加载词典（dictionary）与反向词典（re_dict）
- 从 SQLite 数据库读取全部测试数据
- 自动分割、抽取得到用于预测的句子集合

**✂️ 2. 中文分词与预处理**

- 使用 `cut_only()` 对所有句子进行精准中文分词
- 自动过滤标点、非汉字字符
- 支持全量数据或自定义数量的快速 Tokenization（含进度条）

**🎲 3. 随机样本抽取（Pick up a Data）**

- 自动将分词结果映射为索引序列
- 支持固定长度窗口截取（随机位置）
- 将序列自动转换为 PyTorch Tensor，并添加 Batch 维度

**🔮 4. 下一词预测（Next Word Prediction）**

- 使用训练好的 `NormalRNNForClassification` 模型进行推理
- 支持温度采样（Temperature Scaling）
- 自动进行 Softmax 概率归一化
- 支持 Top-K 预测（可配置）

**📊 5. 完整可视化展示**

- 词典、反向词典可视化（data_editor）
- 显示选中的原始中文序列
- 显示对应的 Token 序列与 Tensor 输入
- 显示预测结果（Top-K 的汉字）

**🌟 6. 全流程计时（内置 Timer）**

- 初始化计时
- 随机抽取计时
- 推理预测计时
- 结果自动展示，便于性能分析

**🔁 7. 交互控制**

- 支持 Repick（重新选取样本）
- 支持 Repredict（重复预测）
- 调用 `streamlit.rerun()` 自动刷新 UI

**快速开始**
---

1. 将本仓库克隆到本地计算机。
2. 使用以下命令安装所需依赖项：`pip install -r requirements.txt`
3. 使用以下命令运行应用程序：`streamlit run main.py`
4. 你也可以通过点击以下链接在线体验该应用：  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://dl-next-word.streamlit.app/)

**隐私声明**
---
本应用程序旨在处理您提供的数据以生成定制化的建议和结果。您的隐私至关重要。

**我们不会收集、存储或传输您的个人信息或数据。** 所有处理都在您的设备本地进行（在浏览器或运行时环境中），*
*数据永远不会发送到任何外部服务器或第三方。**

- **本地处理：** 您的数据永远不会离开您的设备。整个分析和生成过程都在本地进行。
- **无数据保留：** 由于没有数据传输，因此不会在任何服务器上存储数据。关闭应用程序通常会清除任何临时本地数据。
- **透明度：** 整个代码库都是开源的。我们鼓励您随时审查[代码](./)以验证您的数据处理方式。

总之，您始终完全控制和拥有自己的数据。

**许可声明**
---
本项目是开源的，可在 [BSD-3-Clause 许可证](LICENCE) 下使用。

简单来说，这是一个非常宽松的许可证，允许您几乎出于任何目的自由使用此代码，包括在专有项目中，只要您包含原始的版权和许可证声明。

欢迎随意分叉、修改并在此作品基础上进行构建！我们只要求您在适当的地方给予认可。

**环境设置**
---
本项目使用 **Python 3.12** 和 [uv](https://docs.astral.sh/uv/) 进行快速的依赖管理和虚拟环境处理。所需的 Python
版本会自动从 [.python-version](.python-version) 文件中检测到。

1. **安装 uv**：  
   如果您还没有安装 `uv`，可以使用以下命令安装：
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 此安装方法适用于 macOS 和 Linux。
    ```
   或者，您可以运行以下 PowerShell 命令来安装：
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # 此安装方法适用于 Windows。
    ```

   **💡 推荐**：为了获得最佳体验，请将 `uv` 作为独立工具安装。避免在 `pip` 或 `conda` 环境中安装，以防止潜在的依赖冲突。

2. **添加依赖**：

- 添加主要（生产）依赖：
    ```bash
    uv add <package_name>
    # 这会自动更新 pyproject.toml 并安装包
    ```
- 添加开发依赖：
    ```bash
    uv add <package_name> --group dev
    # 示例：uv add ruff --group dev
    # 这会自动将包添加到 [project.optional-dependencies.dev] 部分
    ```
- 添加其他类型的可选依赖（例如测试、文档）：
    ```bash
    uv add <package_name> --group test
    uv add <package_name> --group docs
    ```
- 从 `requirements.txt` 文件导入依赖：
    ```bash
    uv add -r requirements.txt
    # 这会从 requirements.txt 读取包并将其添加到 pyproject.toml
    ```
- 从当前依赖生成 `requirements.txt` 文件：
    ```bash
    # 这会将所有依赖（包括可选依赖）导出到 requirements-all.txt
    uv pip compile pyproject.toml --all-extras -o requirements.txt
    ```

3. **移除依赖**

- 移除主要（生产）依赖：
    ```bash
    uv remove <package_name>
    # 这会自动更新 pyproject.toml 并移除包
    ```
- 移除开发依赖：
    ```bash
    uv remove <package_name> --group dev
    # 示例：uv remove ruff --group dev
    # 这会从 [project.optional-dependencies.dev] 部分移除包
    ```
- 移除其他类型的可选依赖：
    ```bash
    uv remove <package_name> --group test
    uv remove <package_name> --group docs
    ```

4. **管理环境**

- 使用添加/移除命令后，同步环境：
    ```bash
    uv sync
    ```

**更新日志**
---
本项目使用 [git-changelog](https://github.com/pawamoy/git-changelog)
基于 [Conventional Commits](https://www.conventionalcommits.org/) 自动生成和维护更新日志。

1. **安装**
   ```bash
   pip install git-changelog
   # 或使用 uv 将其添加为开发依赖
   uv add git-changelog --group dev
   ```
2. **验证安装**
   ```bash
   pip show git-changelog
   # 或专门检查版本
   pip show git-changelog | grep Version
   ```
3. **配置**
   确保您在项目根目录有一个正确配置的 `pyproject.toml` 文件。配置应将 Conventional Commits 指定为更新日志样式。以下是示例配置：
   ```toml
   [tool.git-changelog]
   version = "0.1.0"
   style = "conventional-commits"
   output = "CHANGELOG.md"
   ```
4. **生成更新日志**
   ```bash
   git-changelog --output CHANGELOG.md
   # 或者使用 uv 运行
   uv run git-changelog --output CHANGELOG.md
   ```
   此命令会根据您的 git 历史记录创建或更新 `CHANGELOG.md` 文件。
5. **推送更改**
   ```bash
   git push origin main
   ```
   或者，使用您 IDE 的 Git 界面（例如，在许多编辑器中的 `Git → Push`）。
6. **注意**：

- 更新日志是根据您的提交消息遵循 Conventional Commits 规范自动生成的。
- 每当您想要更新更新日志时（通常在发布之前或进行重大更改之后），运行生成命令。
