# 发布包验收

从 `Siamese-release` 根目录运行：

```bash
PYTHONPATH=src python -m scripts.verify_release
```

验收工具检查：

- 必需 README、配置、测试和模型文件；
- YAML 语法及 `algorithm_name: Siamese`；
- 补充 checkpoint SHA-256；
- 发布包是否包含 `data/`；
- 文件名是否为 ASCII 且不含空格、括号和中文；
- 是否存在 `__pycache__` 或 `.pyc`；
- 文档是否残留作者机器绝对路径；
- 正式脚本是否能执行 `--help`。

工具默认生成 `release_verification_report.json`。报告失败时不得创建正式 DOI 版本；失败项必须修复或在发布记录中明确豁免理由。
