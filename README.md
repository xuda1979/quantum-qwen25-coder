# 量子编程代码生成 AI 项目

## 项目背景与目标

随着量子计算技术的发展，编写正确高效的量子程序逐渐成为开发者的一大挑战。本项目旨在以 **Qwen2.5‑Coder‑7B‑Instruct** 为基础，通过微调（包括监督微调 **SFT** 和参数高效微调 **PEFT / LoRA**）打造一套专精于量子计算代码生成的 AI 辅助工具。该工具的目标包括：

- **从自然语言描述自动生成有效的量子电路代码**：例如根据用户描述构建量子电路、调用 Qiskit 等框架完成任务。
- **优化量子算法实现**：如 Grover 搜索、变分量子算法等，并生成相应的程序模板。
- **代码调试与错误修复**：根据报错信息或需求描述辅助定位并修复量子代码问题。
- **生成噪声中等规模量子（NISQ）设备的模拟脚本**。

通过以上功能，我们期望使模型在定制的量子基准测试（例如将 HumanEval 扩展到量子领域）中取得至少 **80%** 的代码有效率和准确率，从而超越通用大型语言模型在量子领域的表现。同时，该模型应保持较低的推理和训练成本，便于在 NPU/GPU 上部署，助力明年战略项目的子任务落地。

## 仓库结构

```
quantum-qwen25-coder/
├── models/qwen2/
│   ├── configuration_qwen2.py       # Qwen2 模型配置
│   ├── modeling_qwen2.py            # Qwen2 模型实现（来自 transformers）
│   └── tokenization_qwen2.py        # Qwen2 分词器实现
├── train_sft.py                     # 监督微调脚本
├── train_peft.py                    # LoRA/PEFT 微调脚本
├── evaluate.py                      # 简单评测脚本，用于验证生成代码的编译通过率
├── crawler.py                       # 数据爬虫示例，演示如何收集量子编程样例
├── data/
│   ├── train.jsonl                  # 示例训练集
│   ├── valid.jsonl                  # 示例验证集
│   └── eval.jsonl                   # 示例评测集（仅提供 prompt）
└── README.md                        # 项目说明文档（当前文件）
```

> **注意：**
> - `models/qwen2/` 目录下的三个文件来自于开源项目 [`huggingface/transformers` 中的 Qwen2 模型实现](https://huggingface.co/). 我们将其复制到仓库中，便于在无法联网的环境下查阅与自定义。
> - 预训练权重未包含在仓库中，请按照下面的说明从 Hugging Face 下载 `Qwen/Qwen2.5-Coder-7B-Instruct` 权重。

## 数据准备

1. **数据来源：**
   - 量子编程教程与文档，例如 Qiskit Textbook、IBM Quantum Lab 官方示例、QuTiP 教程等。
   - 公开的量子算法论文附录中的代码示例。
   - 量子编程竞赛题目及其参考解答。
   - 自有任务描述与代码对（如公司内部项目）。

2. **数据格式：**
   - 建议采用 **JSONL**（一行一个 JSON 对象）的格式存储训练/验证样本。
   - 每条数据应包含 `prompt` 字段（自然语言问题或任务描述）和 `code` 字段（对应的量子代码）。可选地包含 `analysis` 或 `reasoning` 字段，用于引导模型先思考再作答。

3. **数据爬取：**
   - 本仓库提供的 `crawler.py` 演示了如何从 Qiskit 教程页面提取标题和代码块，保存为 JSONL 文件。具体解析规则需要根据网页结构调整。
   - 在爬取网站内容时请遵守目标站点的使用条款，设置合理的请求间隔（脚本中默认每个页面暂停 1 秒）。
   - 如果目标数据集需要授权，请确保具备相应的权限后再下载使用。

4. **PDF 论文转数据集：**
   - 使用 `tools/pdf_to_sft.py` 可将论文 PDF 批量转化为 JSONL 数据。脚本会抽取 PDF 文本、按照指定窗口大小切分为多个片段，并生成包含 `prompt` 与 `code` 字段的训练样本（默认目标是原始文本，可通过模板参数自定义）。
   - 示例命令：

     ```bash
     pip install pypdf  # 如未安装 PDF 解析依赖
     python tools/pdf_to_sft.py \
         --input data/papers \
         --output data/papers.jsonl \
         --chunk-size 800 \
         --chunk-overlap 120
     ```

   - 生成的数据可直接作为 `train_sft.py` 或 `train_peft.py` 的输入文件使用，必要时可结合额外的人工标注或模板进一步加工。

5. **数据清洗与增强：**
   - 清理掉无关信息，例如纯文字描述或缺少代码的记录。
   - 对代码进行格式化，确保缩进和语法正确。
   - 可添加自定义系统提示，鼓励模型遵循某些编码规范（如使用 Qiskit 优雅 API）。

## 环境准备

1. **硬件**：建议使用带有充足显存的 GPU 或 NPU。如果使用华为 Ascend NPU，请安装支持 PyTorch 的 `ascend‑pytorch` 套件，并设置 `ASCEND_VISIBLE_DEVICES` 等环境变量。
2. **软件依赖**（需要在外部真实训练环境中安装）：
   - Python >= 3.8
   - `torch` >= 2.0（GPU 或 NPU 版本）
   - `transformers` >= 4.37
   - `peft` >= 0.7
   - `datasets`、`accelerate`、`qiskit`（用于代码验证）等

可使用如下命令安装（示例，实际版本请根据硬件环境调整）：

```bash
pip install torch==2.1.0 transformers==4.38.2 peft==0.7.0 datasets qiskit
```

### 下载预训练权重

由于模型体积较大（约数十 GB），建议在训练环境中通过 Hugging Face CLI 或 Transformers API 下载：

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
```

## 监督微调（SFT）

`train_sft.py` 用于对预训练模型进行全参数监督微调。其核心步骤包括：

1. **加载训练数据**：脚本通过 `load_jsonl` 读取包含 `prompt` 和 `code` 的 JSONL 文件。
2. **拼接提示**：将系统提示、用户输入和参考代码拼接为一条训练样本。
3. **使用 `Trainer` 训练**：定义 `TrainingArguments`，选择合适的 batch size、学习率和 epoch 数，执行 `trainer.train()`。
4. **模型保存**：训练结束后，保存模型权重和分词器。

示例命令：

```bash
python train_sft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/train.jsonl \
    --validation_file data/valid.jsonl \
    --output_dir outputs/sft_qwen25_quantum \
    --per_device_train_batch_size 1 \
    --num_train_epochs 3
```

> 若在显存紧张的情况下无法完成全模型微调，可考虑降低 `per_device_train_batch_size` 并增加梯度累积步数。

## 参数高效微调（PEFT / LoRA）

`train_peft.py` 提供了 LoRA 微调实现，通过在特定投影层添加低秩矩阵，仅训练少量可学习参数，极大降低了训练资源消耗。主要流程：

1. **加载基础模型**：与 SFT 类似，首先加载预训练模型和分词器。
2. **定义 LoRA 配置**：通过 `LoraConfig` 指定 `r`、`alpha`、`dropout` 以及应用的模块列表（默认对注意力中的 `q_proj` 和 `v_proj` 进行适配）。
3. **构建 PEFT 模型**：使用 `get_peft_model` 包装原始模型，模型大多数参数被冻结，仅 LoRA 层可以训练。
4. **使用 `Trainer` 训练**：设置合适的学习率和 batch size，并启动训练。

示例命令：

```bash
python train_peft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/train.jsonl \
    --validation_file data/valid.jsonl \
    --output_dir outputs/peft_qwen25_quantum \
    --target_modules q_proj v_proj \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_train_epochs 3
```

PEFT 模型训练完成后，权重存储在 `output_dir` 中，可使用 `AutoModelForCausalLM.from_pretrained` 加载。生成时无需合并 LoRA 权重，因为 PEFT 框架会自动注入。

## 评测与验证

`evaluate.py` 提供了一个简单的自动化评测管道：

1. 读取评测集 JSONL 文件，评测集仅需包含 `prompt` 字段。
2. 对每个 `prompt` 生成代码，默认使用系统提示引导模型专注于量子编程。
3. 使用 Python 内置函数 `compile()` 检查生成代码的语法正确性，以此粗略评估代码有效率。
4. 输出每个样本的生成代码及编译是否通过，并统计总体通过率。

示例命令：

```bash
python evaluate.py \
    --model_dir outputs/peft_qwen25_quantum \
    --eval_file data/eval.jsonl \
    --result_file outputs/eval_results.jsonl
```

> 若要进行更严格的评测，可引入 Qiskit 或其他量子模拟器，对生成代码进行实际运行并比对输出。

## 在 NPU 上训练

本项目假设部分训练将在华为 Ascend NPU 等硬件上进行。建议参考以下步骤：

1. 安装适配 NPU 的 PyTorch 版本（如 `ascend‑pytorch`）。
2. 设置环境变量，例如 `ASCEND_VISIBLE_DEVICES=0` 指定使用的 NPU 卡。
3. 在训练脚本中不需要显式指定设备，Transformers 会自动选择 `cuda` 或 `npu` 作为默认设备。如果需要自定义，可以使用 `model.to('npu')`。
4. LoRA 微调通常对显存需求较低，更适合在 NPU 上快速迭代。

## 开发建议

1. **数据规模**：为了取得理想效果，建议准备包含数万到数十万条问答/代码对的数据。我们提供的示例仅用于演示流程。
2. **安全性与鲁棒性**：在训练和评测中加入非法操作过滤（例如禁止访问网络、文件系统等），确保模型生成安全可靠的代码。
3. **增量迭代**：先进行小规模 SFT，再在此基础上做 LoRA 微调；或先完成 LoRA 微调，再全量训练一个 epoch，用以提炼高质量的适应层。
4. **版本控制**：建议将不同阶段的模型、数据和脚本保存在独立分支或目录，便于回溯和比较。

## 致谢

感谢阿里云通义千问团队开源的 Qwen2.5‑Coder 系列模型和官方文档。本项目遵循其 Apache‑2.0 许可证并在此基础上进行二次开发。通过社区力量，我们期待助力量子计算行业的发展，让更多开发者能够低门槛地编写、调试和优化量子程序。