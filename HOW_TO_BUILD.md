# 本地编译 / 复现说明

## 1. 生成 PDF（提交用的正本）

作业要求原文：*"You are required to use LaTeX to generate PDF. Word or other
formats are not accepted."* —— 交上去的必须是 LaTeX 编译出来的 PDF。

```bash
cd report
make            # latexmk -pdf，自动跑 bibtex 和多遍编译
make pages      # 打印页数（上限 20，当前 20）
make clean      # 清中间文件，不删 PDF
```

没有 latexmk 就用 `make manual`（pdflatex → bibtex → pdflatex ×2）。

### TinyTeX / BasicTeX 缺包

最小安装会报 `File 'xxx.sty' not found` 或 `Font ... rsfs10 ... not found`。
一次性补齐：

```bash
sudo tlmgr update --self
sudo tlmgr install titlesec titling caption subcaption float booktabs \
     multirow algorithms algorithmicx rsfs amsfonts cm-super \
     setspace parskip xcolor colortbl enumitem epstopdf-pkg latexmk
```

`report.tex` 已经把 `titlesec`、`algorithm`、`algpseudocode`、`multirow`
用 `\IfFileExists` 包成可选依赖，缺了也能编译（只是章节标题退回默认字号）。
`rsfs` 是唯一必须装的字体包。

### 用 Overleaf

1. 把 `code/results/*.png` 复制到 `report/figures/`
2. 把 `report/` 打包 zip 上传，主文档选 `report.tex`，编译器 pdfLaTeX

（`report.tex` 里 `\graphicspath{{../code/results/}{figures/}}` 两个路径都找，
所以复制到 `figures/` 后不用改代码。）

## 2. 生成 Word 草稿（只用来改，不提交）

`report/report_draft.docx` 已经给了。改了 .tex 想重导：

```bash
cd report
pandoc report.tex --bibliography=references.bib --citeproc \
       --resource-path=.:../code/results -o report_draft.docx
```

工作流：Word 里改文字 → 把改动同步回 `report.tex` → `make` 出最终 PDF。
**不要把 Word 当正本**，公式和交叉引用会在来回转换中丢失。

## 3. 复现实验

```bash
cd code
pip install -r requirements.txt
python3 test_algorithms.py      # 39 项自检，不需要数据集，一分钟内跑完

# 把 ORL/ 和 CroppedYaleB/ 放进 code/data/，然后：
bash run_all.sh                 # 完整复现：ORL 约 20 分钟 + YaleB 约 1 小时
FAST=1 bash run_all.sh          # 缩小网格的冒烟测试，几分钟
```

单独跑某一项：

```bash
python3 run_experiment.py  --dataset yaleb --data-root data --output results \
                           --max-iter 250 --runs 5
python3 run_convergence.py --dataset orl   --data-root data --output results
python3 run_sensitivity.py --study delta   --dataset orl --data-root data
python3 make_figures.py    --dataset orl   --data-root data --output results
python3 make_tables.py     --results results --out ../report/tables
```

`--data-root` 传 `data` 或 `data/data` 都行，loader 会自动定位。
`--include-clustering` 作为兼容参数保留（聚类指标本来就默认计算）。
