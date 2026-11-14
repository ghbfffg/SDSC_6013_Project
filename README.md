# DSC 6013 小组项目：Du & Zhang (2015) 复现

复现论文 *Home-purchase restriction, property tax and housing price in China: A counterfactual analysis*（Du & Zhang, Journal of Econometrics 188, 2015）。

论文用"反事实"方法评估三项政策对房价增速的影响：北京限购（2010-05）、上海房产税试点（2011-01）、重庆房产税试点（2011-01）。方法是：用政策前的数据，拿没有该政策的对照城市去拟合目标城市的房价增速，再用拟合关系预测政策后"如果没有政策会怎样"，实际值与预测值之差就是政策效应。

## 目录

```
data/raw/    原始数据（统计局新建商品住宅价格指数的三个来源）
code/        全部代码
outputs/     代码跑出来的表格和图，按步骤分文件夹
```

## 我们做了什么

| 步骤 | 内容 | 输出 |
|---|---|---|
| 1 | 三个数据来源拼成一张 34 城 × 66 月的面板，并核对来源重叠部分 | `outputs/step1_data` |
| 2 | 用 CV(nv) 从 21 个候选城市里选北京的对照城市（搜索约 140 万个组合） | `outputs/step2_cv` |
| 3 | 北京限购的效应 | `outputs/step3_beijing` |
| 4 | 上海房产税的效应 | `outputs/step4_shanghai` |
| 5 | 重庆房产税的效应 | `outputs/step5_chongqing` |
| 6 | 三个案例与论文对比的汇总 | `outputs/step6_summary` |
| 7 | 蒙特卡洛：CV(nv) 与 AIC、AICC 比较（每个设定重复 20 次） | `outputs/step7_monte_carlo` |

`code/t_stat_check.py` 是单独的小脚本，用来核对 t 统计量公式，不在主流程里。

## 主要结果

- 三个案例的回归系数、逐月效应、平均效应和 t 值都与论文一致：北京 −7.69，重庆 −2.52，上海约 0（单位：百分点）。
- 论文选的北京对照组在我们的 CV(nv) 搜索里不是最优（排第 593 名）。换成我们选出的组合，北京效应约 −3.5 到 −4.6，仍然显著，但大小约为论文的一半。
- 蒙特卡洛：样本短时（T1 = 25）CV(nv) 明显优于 AIC 和 AICC，样本长时三者接近。27 个设定里 CV(nv) 有 20 个最好（论文为 23 个）。

## 怎么运行

需要 Python 3 和 pandas、numpy、statsmodels、scipy、matplotlib、openpyxl。

```bash
cd code
python run_all.py                    # 步骤 1 到 6，几分钟
python s7_monte_carlo.py --reps 20   # 步骤 7，要几个小时，可中断后续跑；结果已放在 outputs/step7_monte_carlo
```

也可以单独运行某一步，如 `python s3_beijing.py`。每一步的表格和图会写进 `outputs/` 对应文件夹，`code/README.md` 有更细的说明。

## Tips

- 论文没有给出 t 统计量的完整公式，我们参考 Li & Bell (2017) 实现，滞后阶数用常规规则选取。
- 论文里有几处细节没写清（如蒙特卡洛里 AIC/AICC 的具体流程），我们按自己的理解实现。

## 关于展示和 PPT

- 建议主线：问题是什么、方法一句话怎么讲、我们复现得怎么样、我们多发现了什么。
- 最值得讲的是"北京对照组换了之后效应变小"这个发现。
- 图表可以直接用 `outputs/` 里的图。
