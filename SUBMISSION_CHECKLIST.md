# 提交前检查清单

对照 instruction.pdf 第 4 节和第 5 节评分表。

## 必须手动填的（报告里目前是红色占位）

- [ ] `report.tex` 封面：Group ID
- [ ] `report.tex` 封面：4 位成员的 Name / Student ID / unikey
- [ ] `report.tex` 附录 A：贡献表的姓名、具体分工、百分比
- [ ] 填完后 `cd report && make`，确认页数仍 ≤ 20

## 打包

ZIP 文件名必须是（下划线分隔，Group ID 在前）：

```
GroupIDXX_StudentIDxxxxxxxx_StudentIDxxxxxxxx_StudentIDxxxxxxxx.zip
```

内容结构：

```
report.pdf              ← LaTeX 编译出来的，不是 Word
code/
  algorithm/            ← 五个 NMF 实现 + noise / data_io / evaluate
  data/                 ← 必须是空文件夹（助教会自己拷数据集进去）
  *.py, run_all.sh, requirements.txt
```

**只需要一个人提交。** 逾期每天扣 20%，最多迟交 5 个工作日。

## 评分表自查

| 项目 | 分值 | 状态 |
|---|---|---|
| Abstract：problem / methods / organization | 3 | ✅ |
| Introduction：要解决什么、为什么重要 | 5 | ✅ |
| Previous work：文献中的相关方法 + 各自优缺点 | 6 | ✅ 四小节，每个方法都写了 advantage / disadvantage |
| Methods：预处理、代价函数、噪声描述与图示 | 25 | ✅ 五个算法的 MM 推导 + 理论鲁棒性分析 + 噪声图 |
| Experiments：对比、深入分析、个人反思 | 25 | ✅ 三条噪声轴 × 两数据集 × 三指标 + 收敛 + 敏感性 + 反思 |
| Conclusion：有依据的结论 + 未来工作 | 3 | ✅ |
| Presentation：语法、结构、引用、图表 | 5 | ✅ 14 个引用，图表齐全 |
| Other：让 marker 印象深刻 | 8 | ✅ LaTeX、五个算法、自检脚本、跨平台复现 |
| 附录有运行说明 | 否则 −20 | ✅ 附录 B |
| 指明每个成员的贡献 | 必需 | ⬜ 附录 A，待填 |
| 页数 10–15 理想 / 20 上限 | | ✅ 20 页 |
| 字体 Times，标题 14pt，正文 12pt | | ✅ |
| NMF 只用 numpy/scipy 实现 | | ✅ sklearn 只用于评估 |
