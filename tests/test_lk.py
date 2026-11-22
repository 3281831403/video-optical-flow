"""video-optical-flow 的红色基线测试。

10 个测试分别覆盖题面列出的 10 个缺陷，另有 5 个组合/边界测试。
初始状态下所有测试都调用尚未实现的 optflow.lk 桩，因此应恰好 15 failed。

图像统一为 ``list[list[float]]``（行优先，img[y][x]）。
位移约定：``flow(a, b)[y][x] = (vx, vy)`` 表示 ``b(y, x) ≈ a(y + vy, x + vx)``。
"""

import math

import pytest

from optflow.lk import (
    blur,
    build_pyramid,
    downsample,
    flow,
    gradients,
    lk_step,
    validate_flow,
    warp,
)

TOL = 1e-6


# ---------------------------------------------------------------------------
# 合成图：全部由解析函数在任意实数坐标取值，因此平移图可精确构造。
# ---------------------------------------------------------------------------


def plaid(x, y):
    """低频网格纹理（两个正交方向，周期 40 / 32）。"""
    return (
        1.0
        + 0.7 * math.sin(2.0 * math.pi * x / 40.0 + 0.3)
        + 0.6 * math.cos(2.0 * math.pi * y / 32.0 - 0.2)
    )


def fringe(x, y):
    """低频网格 + 周期 3 的细条纹；先验错半个条纹周期就会落入错误极小值。"""
    return plaid(x, y) + 0.45 * math.sin(2.0 * math.pi * x / 3.0 + 1.1) * math.sin(
        2.0 * math.pi * y / 11.0 + 0.2
    )


def ramp(x, y, gx, gy, offset=1.0):
    """仿射图：离散中心差分与单侧差分都精确等于 (gx, gy)。"""
    return offset + gx * x + gy * y


def rotated_plaid(x, y):
    """旋转 30° 的低频纹理：Ix·Iy 不为零，必须解耦合的 2×2 方程。"""
    theta = math.radians(30.0)
    u = math.cos(theta) * x + math.sin(theta) * y
    v = -math.sin(theta) * x + math.cos(theta) * y
    return (
        1.0
        + 0.7 * math.sin(2.0 * math.pi * u / 40.0 + 0.3)
        + 0.6 * math.cos(2.0 * math.pi * v / 32.0 - 0.2)
    )


def _image(h, w, fn):
    return [[fn(x, y) for x in range(w)] for y in range(h)]


def _shifted(h, w, fn, vx, vy):
    """b(y, x) = fn(x + vx, y + vy)，即 b(p) = a(p + v)。"""
    return [[fn(x + vx, y + vy) for x in range(w)] for y in range(h)]


def _flow_mean_error(est, vx, vy):
    total = 0.0
    count = 0
    for row in est:
        for ex, ey in row:
            total += max(abs(ex - vx), abs(ey - vy))
            count += 1
    return total / count


def _flow_max_error(est, vx, vy):
    worst = 0.0
    for row in est:
        for ex, ey in row:
            worst = max(worst, abs(ex - vx), abs(ey - vy))
    return worst


def _max_abs_diff(left, right):
    worst = 0.0
    for y, row in enumerate(left):
        for x, value in enumerate(row):
            worst = max(worst, abs(value - right[y][x]))
    return worst


# ---------------------------------------------------------------------------
# 10 条缺陷测试
# ---------------------------------------------------------------------------


def test_flow_scales_coarse_displacement_up_by_two():
    """缺陷 1：由粗到细把上一层位移除以 2，3 px 平移会缩成 0.75 px。"""
    a = _image(32, 32, fringe)
    b = _shifted(32, 32, fringe, 3.0, 0.0)

    est = flow(a, b, levels=2, window=2)

    assert _flow_mean_error(est, 3.0, 0.0) < 0.25


def test_downsample_returns_ceil_size():
    """缺陷 2：downsample 用 w//2 取整，奇数尺寸会丢掉最后一行/列。"""
    cases = [(9, 7), (8, 6), (3, 5), (2, 2), (1, 1)]
    for h, w in cases:
        image = [[1.0] * w for _ in range(h)]
        reduced = downsample(image)
        assert len(reduced) == (h + 1) // 2
        assert len(reduced[0]) == (w + 1) // 2

    odd = [[0.0] * 7 for _ in range(9)]
    odd[8][6] = 1.0
    reduced = downsample(odd)
    assert len(reduced) == 5
    assert len(reduced[0]) == 4
    assert reduced[4][3] > 0.05


def test_gradients_use_one_sided_difference_at_borders():
    """缺陷 3：边界用零填充会把边界梯度减半，边界位移被钉在 0。"""
    h, w = 9, 11
    gx, gy = 0.6, 0.8
    image = _image(h, w, lambda x, y: ramp(x, y, gx, gy))

    ix, iy = gradients(image)
    for y in range(h):
        for x in range(w):
            assert ix[y][x] == pytest.approx(gx, abs=1e-12)
            assert iy[y][x] == pytest.approx(gy, abs=1e-12)

    vx, vy = 2.0 * gx, 2.0 * gy  # 与梯度同向，孔径问题不影响可观测分量
    b = _shifted(h, w, lambda x, y: ramp(x, y, gx, gy), vx, vy)
    est = lk_step(image, b, window=2)
    assert _flow_max_error(est, vx, vy) < 1e-6


def test_lk_step_solves_2x2_system_without_exploding():
    """缺陷 4：用累加代替 2×2 线性方程，弱纹理方向会爆炸或被放大。"""
    h, w = 8, 10
    gx, gy = 0.6, 0.8
    image = _image(h, w, lambda x, y: ramp(x, y, gx, gy))
    vx, vy = 1.5 * gx, 1.5 * gy

    b = _shifted(h, w, lambda x, y: ramp(x, y, gx, gy), vx, vy)
    est = lk_step(image, b, window=2)
    assert _flow_max_error(est, vx, vy) < 1e-6

    # 旋转纹理上 Ix·Iy 非零：按轴分别相除会把交叉项丢掉。
    side = 24
    tex = _image(side, side, rotated_plaid)
    shifted = _shifted(side, side, rotated_plaid, 0.5, 0.0)
    coupled = lk_step(tex, shifted, window=3)
    assert _flow_mean_error(coupled, 0.5, 0.0) < 0.1

    flat = [[2.0] * 6 for _ in range(5)]
    zero = lk_step(flat, flat, window=2)
    assert _flow_max_error(zero, 0.0, 0.0) < 1e-6


def test_warp_uses_backward_sampling():
    """缺陷 5：warp 采样方向用 +flow，重建图会朝反方向平移。"""
    h, w = 8, 10
    gx, gy = 0.5, -0.3
    vx, vy = 2.0, -1.0

    a = _image(h, w, lambda x, y: ramp(x, y, gx, gy))
    b = _shifted(h, w, lambda x, y: ramp(x, y, gx, gy), vx, vy)
    known = [[(vx, vy)] * w for _ in range(h)]

    recovered = warp(b, known)
    for y in range(0, h - 1):
        for x in range(2, w):
            assert recovered[y][x] == pytest.approx(a[y][x], abs=1e-6)


def test_warp_uses_bilinear_weights_for_subpixel_flow():
    """缺陷 6：双线性权重用 int() 截断，亚像素精度会退化为半像素。"""
    h, w = 8, 10
    gx, gy = 0.5, -0.3
    vx, vy = 0.5, -0.25

    a = _image(h, w, lambda x, y: ramp(x, y, gx, gy))
    b = _shifted(h, w, lambda x, y: ramp(x, y, gx, gy), vx, vy)
    known = [[(vx, vy)] * w for _ in range(h)]

    recovered = warp(b, known)
    for y in range(0, h - 1):
        for x in range(1, w):
            assert recovered[y][x] == pytest.approx(a[y][x], abs=1e-6)


def test_flow_keeps_coarse_prior():
    """缺陷 7：每层重新初始化窗口，细层丢失粗层先验，大位移恢复失败。"""
    a = _image(32, 32, plaid)
    b = _shifted(32, 32, plaid, 6.0, 0.0)

    est = flow(a, b, levels=3, window=3)

    assert _flow_mean_error(est, 6.0, 0.0) < 0.25


def test_blur_kernel_is_normalized():
    """缺陷 8：blur 核不归一，金字塔各层亮度会逐层变暗。"""
    level = 3.25
    constant = [[level] * 13 for _ in range(9)]

    smoothed = blur(constant)
    assert _max_abs_diff(smoothed, constant) < 1e-12

    pyramid = build_pyramid(constant, 4)
    for layer in pyramid:
        assert _max_abs_diff(layer, constant) < 1e-9


def test_validate_flow_flags_out_of_range_vectors():
    """缺陷 9：只检查 NaN 不检查 |flow| 超界，异常矢量被当作有效。"""
    h, w = 8, 6
    good = [[(0.25, -0.5)] * w for _ in range(h)]
    assert validate_flow(good, (h, w)) == []

    bad = [row[:] for row in good]
    bad[3][4] = (10.0 * w, 0.0)
    messages = validate_flow(bad, (h, w))
    assert len(messages) >= 1

    nan_flow = [row[:] for row in good]
    nan_flow[0][0] = (float("nan"), 0.0)
    assert len(validate_flow(nan_flow, (h, w))) >= 1

    assert len(validate_flow(good[:-1], (h, w))) >= 1


def test_flow_reverses_direction_when_arguments_swap():
    """缺陷 10：flow(a,b) 与 flow(b,a) 同向，双向一致性检查失败。"""
    h, w = 24, 24
    a = _image(h, w, plaid)
    b = _shifted(h, w, plaid, 1.0, 0.0)

    forward = flow(a, b, levels=2, window=2)
    backward = flow(b, a, levels=2, window=2)

    for y in range(h):
        for x in range(w):
            fx, fy = forward[y][x]
            bx, by = backward[y][x]
            assert abs(fx + bx) < 0.05
            assert abs(fy + by) < 0.05
    assert _flow_mean_error(forward, 1.0, 0.0) < 0.25
    assert _flow_mean_error(backward, -1.0, 0.0) < 0.25


# ---------------------------------------------------------------------------
# 5 条组合 / 边界测试
# ---------------------------------------------------------------------------


def test_recovers_known_three_pixel_translation():
    """组合：3 px 已知平移的恢复误差应小于 0.25 px。"""
    a = _image(32, 32, plaid)
    b = _shifted(32, 32, plaid, 3.0, 0.0)

    est = flow(a, b, levels=3, window=3)

    assert _flow_mean_error(est, 3.0, 0.0) < 0.25
    assert _flow_max_error(est, 3.0, 0.0) < 0.25


def test_flow_of_identical_images_is_zero():
    """组合：flow(a, a) 必须处处为 0（容差 1e-6）。"""
    a = _image(24, 30, plaid)

    est = flow(a, a, levels=3, window=2)

    for row in est:
        for vx, vy in row:
            assert abs(vx) <= 1e-6
            assert abs(vy) <= 1e-6


def test_warp_roundtrip_through_flow_recovers_a():
    """组合：warp(b, flow(a,b)) 应还原 a，误差 < 1e-6。"""
    h, w = 10, 12
    gx, gy = 0.6, 0.8
    vx, vy = 1.2, 1.6  # 与梯度同向

    a = _image(h, w, lambda x, y: ramp(x, y, gx, gy))
    b = _shifted(h, w, lambda x, y: ramp(x, y, gx, gy), vx, vy)

    est = flow(a, b, levels=1, window=2)
    assert _flow_mean_error(est, vx, vy) < 1e-6

    recovered = warp(b, est)
    for y in range(2, h):
        for x in range(2, w):
            assert recovered[y][x] == pytest.approx(a[y][x], abs=1e-6)


def test_levels_one_and_three_agree_for_small_motion():
    """组合：小平移下 1 层与 3 层金字塔的结果应一致（±0.5 px）。"""
    a = _image(32, 32, plaid)
    b = _shifted(32, 32, plaid, 0.5, 0.0)

    single = flow(a, b, levels=1, window=3)
    pyramid = flow(a, b, levels=3, window=3)

    for y in range(32):
        for x in range(32):
            assert abs(single[y][x][0] - pyramid[y][x][0]) <= 0.5
            assert abs(single[y][x][1] - pyramid[y][x][1]) <= 0.5
    assert _flow_mean_error(single, 0.5, 0.0) < 0.25
    assert _flow_mean_error(pyramid, 0.5, 0.0) < 0.25


def test_odd_size_pyramid_keeps_last_row_and_column():
    """组合：奇数尺寸金字塔各层尺寸满足 ceil，最后一行/列的内容不丢。"""
    image = [[0.0] * 7 for _ in range(9)]
    image[8][6] = 1.0

    pyramid = build_pyramid(image, 3)

    assert [(len(layer), len(layer[0])) for layer in pyramid] == [(9, 7), (5, 4), (3, 2)]
    assert pyramid[1][4][3] > 0.05
    assert pyramid[2][2][1] > 0.01
