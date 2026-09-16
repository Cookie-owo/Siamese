# Siamese 发布包文件命名标准

## 适用范围

本标准适用于最终提交到代码仓库、模型仓库和结果归档的所有文件与目录。当前 `Siamese-release` 中保留的历史文件可以暂时使用 `legacy` 标识，但正式发布前仍须通过本标准的最终审计。

## 命名规则

- 统一使用 ASCII 字符；
- 文件和目录使用小写 `snake_case`，模型文件例外可使用完整语义化名称；
- 不使用空格、中文、括号、`副本`、`copy`、`tmp` 或随机生成后缀；
- 扩展名使用标准小写形式：`.py`、`.yaml`、`.md`、`.csv`、`.json`、`.txt`、`.png`、`.pdf`、`.pth`；
- 日期统一使用 `YYYY-MM-DD`；
- 不使用 `final_final`、`new`、`latest` 等不可审计名称；
- 版本号只用于发布版本、数据版本或配置版本，不用于掩盖模型用途；
- 每个文件名应能从名称判断其内容和科学角色。

## 当前命名结论

以下命名符合正式发布语义：

- `siamese_attention_resnet34_supplementary_checkpoint.pth`；
- `release_supplementary_inference.yaml`；
- `simolfa_evaluation` 和 `test_set_evaluation`；
- `CHECKPOINT_EVALUATION_2026-09-08.md`。

`legacy` 表示历史实验角色，不能被解释为最终主实验。日期用于追踪审计时点，不表示模型版本。

## 发布前必须排除

- `__pycache__/` 目录；
- `.pyc` 字节码文件；
- 重复嵌套目录，例如 `tests/tests/`；
- 含内部绝对路径的临时日志；
- 任何患者原始文件名或受限数据文件。

## 当前发现

当前整理目录曾因复制和测试运行产生 `__pycache__`、`.pyc` 和 `tests/tests/`。这些是构建缓存或复制残留，不是正式发布资产，必须在最终打包前清除。当前权重和主要文档名称已按本标准修正。
