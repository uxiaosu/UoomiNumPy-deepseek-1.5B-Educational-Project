# Quick Start Guide / 快速开始指南

<!-- Language Toggle -->
<div align="center" id="language-toggle">
  <button onclick="showEnglish()" id="en-btn" style="background: #4CAF50; color: white; border: none; padding: 8px 16px; margin: 4px; border-radius: 4px; cursor: pointer;">🇺🇸 English</button>
  <button onclick="showChinese()" id="zh-btn" style="background: #2196F3; color: white; border: none; padding: 8px 16px; margin: 4px; border-radius: 4px; cursor: pointer;">🇨🇳 中文</button>
</div>

<script>
function showEnglish() {
  document.getElementById('english-content').style.display = 'block';
  document.getElementById('chinese-content').style.display = 'none';
  document.getElementById('en-btn').style.background = '#4CAF50';
  document.getElementById('zh-btn').style.background = '#ccc';
}

function showChinese() {
  document.getElementById('english-content').style.display = 'none';
  document.getElementById('chinese-content').style.display = 'block';
  document.getElementById('zh-btn').style.background = '#2196F3';
  document.getElementById('en-btn').style.background = '#ccc';
}

// Default to English
showEnglish();
</script>

---

<div id="english-content">

# 🇺🇸 English Version

🚀 Get up and running with UoomiNumPy deepseek Educational Project in minutes!

## 🌟 Project Overview

UoomiNumPy deepseek Educational Project is a **pure NumPy implementation** of the DeepSeek language model, designed for educational purposes and deep learning research.

## 🔧 Technical Features

- **🧠 Pure NumPy**: Complete transformer implementation using only NumPy
- **📚 Educational Focus**: Perfect for understanding LLM internals
- **🎯 Adaptive Thinking**: AI generates dynamic thinking templates
- **⚡ Lightweight**: No heavy dependencies, runs anywhere
- **🔍 Transparent**: Every operation is visible and understandable
- **🔄 Weight Conversion**: Convert from safetensors/PyTorch to NumPy

## 📋 Dependencies

### Required
- **Python**: 3.7 or higher
- **NumPy**: Core computation library

### Optional
- **safetensors**: For weight conversion
- **torch**: For PyTorch model conversion

## ⚡ Quick Installation

```bash
# Clone the project
cd UoomiNumPy-deepseek-1.5B-Educational-Project

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Basic Usage

```python
# Simple text generation
from src.api import load_model

model = load_model("./weights/my_model")
result = model.generate(
    prompt="Hello, world!",
    max_new_tokens=30
)
print(result)
```

```bash
# Command line usage
python main.py generate --prompt "Your text here"
```

</div>

<div id="chinese-content" style="display: none;">

# 🇨🇳 中文版本

🚀 几分钟内快速上手 UoomiNumPy deepseek 教育项目！

## 🌟 项目简介

UoomiNumPy deepseek 教育项目是 DeepSeek 语言模型的**纯 NumPy 实现**，专为教育目的和深度学习研究而设计。

## 🔧 技术特性

- **🧠 纯 NumPy**: 仅使用 NumPy 完整实现 transformer
- **📚 教育导向**: 完美理解大语言模型内部机制
- **🎯 自适应思维**: AI 动态生成思维模板
- **⚡ 轻量级**: 无重型依赖，随处运行
- **🔍 透明化**: 每个操作都可见且可理解
- **🔄 权重转换**: 支持从 safetensors/PyTorch 转换到 NumPy

## 📋 依赖要求

### 必需依赖
- **Python**: 3.7 或更高版本
- **NumPy**: 核心计算库

### 可选依赖
- **safetensors**: 用于权重转换
- **torch**: 用于 PyTorch 模型转换

## ⚡ 快速安装

```bash
# 克隆项目
cd UoomiNumPy-deepseek-1.5B-Educational-Project

# 安装依赖
pip install -r requirements.txt
```

## 🎯 基本使用

```python
# 简单文本生成
from src.api import load_model

model = load_model("./weights/my_model")
result = model.generate(
    prompt="你好，世界！",
    max_new_tokens=30
)
print(result)
```

```bash
# 命令行使用
python main.py generate --prompt "您的文本"
```

</div>

---

**Happy learning! 🎊📚 | 祝您学习愉快！🎊📚**

For more details, check `README.md` | 更多详情请查看 `README.md`