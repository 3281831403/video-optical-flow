"""金字塔 Lucas–Kanade 光流核心。

图像统一表示为二维灰度图 ``list[list[float]]``，形状为 h×w（h 行、w 列），
``img[y][x]`` 是第 y 行第 x 列的灰度值。允许非方图、奇数尺寸和 1×1 图像，
但输入必须是矩形（各行等长）。

位移约定（全模块统一）：

* ``flow(a, b)[y][x] = (vx, vy)`` 表示 ``b(y, x) ≈ a(y + vy, x + vx)``；
* 等价地 ``a(p) ≈ b(p - v)``。因此 ``warp(b, v)`` 在像素 p 处采样 b 的
  ``(y - vy, x - vx)``，可把 b 还原到 a；
* 交换输入应反号：``flow(b, a) ≈ -flow(a, b)``。

金字塔约定：``level 0`` 是原图，``level k`` 由 ``level k-1`` 做 2 倍
降采样得到；第 k 层尺寸为 ``(ceil(h / 2**k), ceil(w / 2**k))``，顶部最小。
浮点比较容差为 1e-6（测试中精确几何断言用更小的 1e-9）。
所有函数都不修改输入。
"""

import math

__all__ = [
    "blur",
    "build_pyramid",
    "downsample",
    "flow",
    "gradients",
    "lk_step",
    "validate_flow",
    "warp",
]


def _shape(img):
    """返回 ``(h, w)``；空图宽度按 0 处理。"""
    return len(img), (len(img[0]) if img else 0)


def _clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _median(values):
    """样本中位数；空序列返回 0.0。偶数个取中间两个的平均。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def blur(img):
    """用 3×3 高斯核对图像做平滑，返回同尺寸新图。

    核为::

        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]] / 16

    即权重之和严格为 1；边界像素复制最近的内侧像素（clamp / edge
    replication），因此常量图经过 blur 后逐像素保持不变（误差 < 1e-12）。
    """
    h, w = _shape(img)
    if h == 0 or w == 0:
        return [[] for _ in range(h)]

    kernel = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
    total = 16.0
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            acc = 0.0
            for ky in range(3):
                sy = _clamp(y + ky - 1, 0, h - 1)
                source = img[sy]
                for kx in range(3):
                    sx = _clamp(x + kx - 1, 0, w - 1)
                    acc += kernel[ky][kx] * source[sx]
            # 核和为 16，除以 16 保证归一化（常量图亮度不变）。
            row.append(acc / total)
        out.append(row)
    return out


def downsample(img):
    """2 倍降采样，返回新图。

    先对 img 做 ``blur``，再取偶数行、偶数列（步长 2）。输出尺寸为
    ``(ceil(h/2), ceil(w/2))``，即 ``((h + 1) // 2, (w + 1) // 2)``；
    奇数尺寸时最后一行/最后一列必须保留，不得用 ``h // 2`` 把它丢掉。
    """
    h, w = _shape(img)
    if h == 0 or w == 0:
        return [[] for _ in range(0)]
    smoothed = blur(img)
    # ceil 关系：保留偶数索引，尺寸为 ((h + 1) // 2, (w + 1) // 2)。
    out = []
    for y in range(0, h, 2):
        out.append([smoothed[y][x] for x in range(0, w, 2)])
    return out


def build_pyramid(img, levels):
    """构建 levels 层金字塔，返回 ``[level0, level1, ..., level_{levels-1}]``。

    ``level0`` 为原图，之后每层对上一层调用 ``downsample``；因此顶层最小。
    ``levels`` 必须是 >= 1 的整数，``levels == 1`` 时返回 ``[img]``。
    第 k 层尺寸为 ``(ceil(h / 2**k), ceil(w / 2**k))``。
    """
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
        raise ValueError("levels must be an integer >= 1")
    pyramid = [img]
    for _ in range(1, levels):
        pyramid.append(downsample(pyramid[-1]))
    return pyramid


def gradients(img):
    """返回 ``(Ix, Iy)`` 两个与 img 同尺寸的梯度图。

    内部像素用中心差分::

        Ix[y][x] = (img[y][x+1] - img[y][x-1]) / 2
        Iy[y][x] = (img[y+1][x] - img[y-1][x]) / 2

    边界用单侧差分（不得用零填充）::

        Ix[y][0]     = img[y][1] - img[y][0]
        Ix[y][w-1]   = img[y][w-1] - img[y][w-2]
        Iy[0][x]     = img[1][x] - img[0][x]
        Iy[h-1][x]   = img[h-1][x] - img[h-2][x]

    宽度或高度为 1 时对应方向梯度为 0。对仿射图 ``gx*x + gy*y + c``，
    包括边界在内的每个像素都应精确得到 ``(gx, gy)``（误差 < 1e-12）。
    """
    h, w = _shape(img)
    ix = [[0.0] * w for _ in range(h)]
    iy = [[0.0] * w for _ in range(h)]
    if h == 0 or w == 0:
        return ix, iy

    for y in range(h):
        for x in range(w):
            # 边界用单侧差分，内部用中心差分；不做零填充。
            if w == 1:
                ix[y][x] = 0.0
            elif x == 0:
                ix[y][x] = img[y][1] - img[y][0]
            elif x == w - 1:
                ix[y][x] = img[y][w - 1] - img[y][w - 2]
            else:
                ix[y][x] = (img[y][x + 1] - img[y][x - 1]) / 2.0

            if h == 1:
                iy[y][x] = 0.0
            elif y == 0:
                iy[y][x] = img[1][x] - img[0][x]
            elif y == h - 1:
                iy[y][x] = img[h - 1][x] - img[h - 2][x]
            else:
                iy[y][x] = (img[y + 1][x] - img[y - 1][x]) / 2.0
    return ix, iy


def lk_step(a, b, window):
    """单层 Lucas–Kanade：返回每个像素的位移 ``(vx, vy)``。

    对像素 (y, x)，在以它为中心、半径 ``window`` 的方形窗口内（越界部分
    裁剪到图像内）最小化 ``sum (b - a - vx*Ix - vy*Iy)**2``，其中
    ``(Ix, Iy) = gradients(a)``、残差 ``e = b - a``。累加::

        M = [[sum(Ix*Ix), sum(Ix*Iy)],
             [sum(Ix*Iy), sum(Iy*Iy)]]
        r = (sum(Ix*e), sum(Iy*e))

    并解 2×2 线性方程 ``M @ (vx, vy) = r``；不得用逐轴累加
    ``sum(Ix*e)/sum(Ix*Ix)`` 之类的可分离近似代替耦合求解。``window``
    必须是非负整数（0 表示 1×1 窗口）。

    结构张量近奇异（``lambda_min <= 1e-8 * lambda_max``，例如只有单向
    边缘的弱纹理）时，只保留可观测方向：返回沿最大特征向量的最小范数解，
    不可观测方向取 0；完全无梯度（``trace(M) <= 1e-12``）时返回 (0, 0)。
    这样纹理弱的方向不会爆炸或产生 NaN。a、b 必须同尺寸；``window`` 为负
    时抛 ValueError。位移方向约定与模块一致：``b(p) ≈ a(p + v)``。
    """
    if (not isinstance(window, int) or isinstance(window, bool)
            or window < 0):
        raise ValueError("window must be a non-negative integer")
    ha, wa = _shape(a)
    hb, wb = _shape(b)
    if (ha, wa) != (hb, wb):
        raise ValueError("a and b must have the same shape")
    h, w = ha, wa
    if h == 0 or w == 0:
        return []

    ix, iy = gradients(a)
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            y0 = max(0, y - window)
            y1 = min(h - 1, y + window)
            x0 = max(0, x - window)
            x1 = min(w - 1, x + window)

            sxx = sxy = syy = sxr = syr = 0.0
            for yy in range(y0, y1 + 1):
                ix_row = ix[yy]
                iy_row = iy[yy]
                a_row = a[yy]
                b_row = b[yy]
                for xx in range(x0, x1 + 1):
                    gx = ix_row[xx]
                    gy = iy_row[xx]
                    # 残差 e = b - a，配准方向：b(p) ≈ a(p + v)。
                    e = b_row[xx] - a_row[xx]
                    sxx += gx * gx
                    sxy += gx * gy
                    syy += gy * gy
                    sxr += gx * e
                    syr += gy * e

            trace = sxx + syy
            if trace <= 1e-12:
                # 完全无梯度：位移不可观测。
                row.append((0.0, 0.0))
                continue

            # 对称 2×2 的特征值，判别式用非负形式避免相减抵消。
            disc = math.sqrt(max(0.0, (sxx - syy) ** 2 + 4.0 * sxy * sxy))
            lam_max = 0.5 * (trace + disc)
            lam_min = 0.5 * (trace - disc)

            if lam_min <= 1e-8 * lam_max:
                # 结构张量近奇异：只保留最大特征向量方向的最小范数解。
                if abs(sxy) > 0.0:
                    ux, uy = lam_max - syy, sxy
                else:
                    ux, uy = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
                norm = math.hypot(ux, uy)
                ux, uy = ux / norm, uy / norm
                proj = ux * sxr + uy * syr
                if proj != 0.0:
                    # 特征值在数值上不可靠时按 trace 估计，避免除零/爆炸。
                    scale = lam_max if lam_max > 1e-12 else trace
                    proj /= scale
                vx, vy = proj * ux, proj * uy
            else:
                # 解耦合的 2×2 线性方程 M @ (vx, vy) = r。
                det = sxx * syy - sxy * sxy
                vx = (syy * sxr - sxy * syr) / det
                vy = (sxx * syr - sxy * sxr) / det

            if not (math.isfinite(vx) and math.isfinite(vy)):
                vx, vy = 0.0, 0.0
            row.append((vx, vy))
        out.append(row)
    return out


def flow(a, b, levels, window):
    """由粗到细估计位移场，返回与 a 同尺寸的每像素 ``(vx, vy)``。

    先对 a、b 分别调用 ``build_pyramid`` 得到 levels 层金字塔，从最粗层
    （``levels-1``）开始向细层推进：

    1. 若上一层已有位移 v，把它放大 2 倍作为本层初值
       （``v_fine = 2 * v_coarse``，不是除以 2）；
    2. 按当前 v 调用 ``warp`` 把 b 的该层采样到 a 的坐标系；
    3. 调用 ``lk_step`` 求增量并累加到 v 上；细层不得丢弃粗层先验。

    为避免弱纹理/重复纹理产生离群矢量，flow 必须对每层的增量场做空间
    稳健聚合（例如取全图分量中位数）后再累加，而不是把逐像素的噪声增量
    直接反馈进 warp；对整体平移输入，正确结果是各像素相同的常量位移场，
    且交换 a、b 后应严格反号。``levels`` 必须是 >= 1 的整数，否则抛
    ValueError。数值容差 1e-6。
    """
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
        raise ValueError("levels must be an integer >= 1")
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("window must be a non-negative integer")

    pyramid_a = build_pyramid(a, levels)
    pyramid_b = build_pyramid(b, levels)

    # v 初始为最粗层的零位移。
    top = levels - 1
    h, w = _shape(pyramid_a[top])
    v = [[(0.0, 0.0)] * w for _ in range(h)]

    for level in range(top, -1, -1):
        a_level = pyramid_a[level]
        b_level = pyramid_b[level]
        h, w = _shape(a_level)

        # 1) 上一层位移放大 2 倍作为本层先验（尺度约定：乘 2 而非除 2）。
        #    奇数尺寸下 2*coarse 不等于本层尺寸（如 5 -> 9），按 y//2, x//2
        #    把粗层位移场重采样到本层尺寸，再做 ×2 缩放。
        if level != top:
            ch = len(v)
            cw = len(v[0]) if v else 0
            scaled = []
            for y in range(h):
                src = v[min(y // 2, ch - 1)]
                scaled.append([
                    (2.0 * src[min(x // 2, cw - 1)][0],
                     2.0 * src[min(x // 2, cw - 1)][1])
                    for x in range(w)
                ])
            v = scaled

        # 2) 按当前 v 反向采样 b，把 b 对齐到 a 的坐标系。
        b_warped = warp(b_level, v)

        # 3) 求增量；对增量场做稳健聚合，避免弱/重复纹理的离群矢量反馈进 warp。
        delta = lk_step(a_level, b_warped, window)
        dxs = []
        dys = []
        for row in delta:
            for dvx, dvy in row:
                dxs.append(dvx)
                dys.append(dvy)
        step_x = _median(dxs)
        step_y = _median(dys)

        # 4) 增量累加到当前层位移；细层保留粗层先验。
        v = [[(vx + step_x, vy + step_y) for vx, vy in row] for row in v]

    return v


def warp(img, flow):
    """按位移场对 img 做反向采样（backward warp），返回同尺寸新图。

    对输出像素 (y, x)，采样源坐标::

        sy = y - vy
        sx = x - vx

    （注意是减号：``warp(b, flow(a, b))`` 应还原 a，而不是把 b 再推远。）
    采样使用双线性插值：取 ``x0 = floor(sx)``，权重 ``fx = sx - x0``，
    对四个邻点加权；不得把权重截断为 0/1。源坐标越界时先 clamp 到
    ``[0, w-1]`` / ``[0, h-1]``（边界复制）。

    对仿射图，正确的双线性采样在任意亚像素位置都应精确还原（误差
    < 1e-9），因此该函数是检验“反向 + 插值”两个约定的关键。
    """
    h, w = _shape(img)
    fh, fw = _shape(flow)
    if (fh, fw) != (h, w):
        raise ValueError("flow must have the same shape as img")
    out = []
    for y in range(h):
        flow_row = flow[y]
        row = []
        for x in range(w):
            vx, vy = flow_row[x]
            # 反向采样：b(p - v) 还原 a(p)。
            sx = _clamp(x - vx, 0.0, w - 1.0)
            sy = _clamp(y - vy, 0.0, h - 1.0)
            x0 = int(math.floor(sx))
            y0 = int(math.floor(sy))
            x1 = min(x0 + 1, w - 1) if w > 1 else x0
            y1 = min(y0 + 1, h - 1) if h > 1 else y0
            # 亚像素权重保留小数，不能截断成 0/1。
            fx = sx - x0
            fy = sy - y0
            top = img[y0][x0] * (1.0 - fx) + img[y0][x1] * fx
            bottom = img[y1][x0] * (1.0 - fx) + img[y1][x1] * fx
            row.append(top * (1.0 - fy) + bottom * fy)
        out.append(row)
    return out


def validate_flow(flow, shape):
    """检查位移场，返回问题描述字符串列表；完全合法时返回空列表。

    ``shape`` 为 ``(h, w)``。依次检查（按行优先顺序，便于复现）：

    1. 形状：flow 必须是长度为 h 的 list，每行是长度为 w 的序列；
    2. 数值：每个元素必须是二元组/二元序列，分量为有限浮点数；
       NaN、inf 都算非法；
    3. 超界：``abs(vx) > w`` 或 ``abs(vy) > h`` 的位移视为异常
       （位移分量不可能大于图像本身）；只查 NaN、不查超界是不合格的。

    每条消息需包含出错像素坐标 ``(y, x)`` 或形状信息；不要抛异常。
    """
    messages = []
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        return ["shape must be a (h, w) pair"]
    h, w = shape
    if not isinstance(flow, (list, tuple)):
        return ["flow must be a list of rows for shape (%d, %d)" % (h, w)]
    if len(flow) != h:
        messages.append(
            "flow has %d rows, expected %d for shape (%d, %d)"
            % (len(flow), h, h, w)
        )
        return messages

    for y in range(h):
        row = flow[y]
        if not isinstance(row, (list, tuple)):
            messages.append("row %d is not a sequence, expected width %d" % (y, w))
            continue
        if len(row) != w:
            messages.append(
                "row %d has length %d, expected %d" % (y, len(row), w)
            )
            continue
        for x in range(w):
            item = row[x]
            try:
                pair = tuple(item)
            except TypeError:
                messages.append(
                    "pixel (%d, %d) is not a (vx, vy) pair" % (y, x)
                )
                continue
            if len(pair) != 2:
                messages.append(
                    "pixel (%d, %d) is not a (vx, vy) pair" % (y, x)
                )
                continue
            vx, vy = pair
            if not isinstance(vx, (int, float)) or not isinstance(vy, (int, float)):
                messages.append(
                    "pixel (%d, %d) has non-numeric component" % (y, x)
                )
                continue
            if not (math.isfinite(vx) and math.isfinite(vy)):
                messages.append(
                    "pixel (%d, %d) has non-finite value (%r, %r)" % (y, x, vx, vy)
                )
                continue
            if abs(vx) > w or abs(vy) > h:
                messages.append(
                    "pixel (%d, %d) out of range: |vx|=%.6g > w=%d or "
                    "|vy|=%.6g > h=%d" % (y, x, abs(vx), w, abs(vy), h)
                )
    return messages
