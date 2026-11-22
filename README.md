# video-optical-flow

## 题面

接入金字塔光流后，大位移恢复不出来、亚像素精度还只有半像素：粗到细时把上一层位移除以 2，整体位移被缩小 2^(levels−1)；downsample 用 w//2 取整，奇数尺寸丢掉最后一行/列、金字塔各层尺寸不满足 ceil 关系；gradients 的边界用零填充，边界位移被钉在 0；lk_step 用逐轴累加代替解 2×2 线性方程，纹理弱的方向位移爆炸；warp 的采样方向用 +flow 而不是 −flow，恢复出的图与 b 反向；warp 的双线性权重用 int() 截断，亚像素精度只剩半像素；flow 每层都重新初始化窗口，细层丢失粗层的先验；blur 的核不归一，金字塔各层亮度逐层变暗；validate_flow 只检查 NaN 不检查 |flow| 是否超界，异常矢量被当作有效；flow(a,b) 与 flow(b,a) 的符号约定不一致、都返回同一方向，双向一致性检查失败。难点在尺度与符号：金字塔位移要乘 2，warp 要反向采样，两者错一个结果就整体反向。optflow/lk.py 是核心，tests/test_lk.py 15个红。要修的是：把上述十处尺度、符号、边界与正则化问题全部改对，直到 15 个测试全绿。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态必须恰好为 `15 failed`，且不得出现 collection、import 或 syntax error。修复完成后，同一命令应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

- 仅使用 Python 标准库（math）与 pytest；离线运行，不使用随机数，不 sleep。
- `tests/test_lk.py` 是评测基线，不可修改；修复只能改 `optflow/lk.py`。
- 图像为 h×w 的行优先二维灰度 `list[list[float]]`；位移约定为 `b(y, x) ≈ a(y + vy, x + vx)`，`warp` 在 `(y - vy, x - vx)` 处做双线性采样；金字塔第 k 层尺寸为 `(ceil(h / 2**k), ceil(w / 2**k))`；浮点容差 1e-6。
