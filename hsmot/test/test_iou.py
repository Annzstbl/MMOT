from mmcv.ops import box_iou_rotated
from mmcv.ops import diff_iou_rotated_2d
import pytest
from hsmot.util.dist import *
import torch
import numpy as np

# @pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise, expexted", \
#                          [
#                              (
#                                 torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.25],
#                                     [0.0, 0.0, 0.5, 0.5, 0.75]]),
#                                 torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.5],
#                                     [0.1, 0.1, 0.5, 0.5, 0.2]]),
#                                 'iou',
#                                 True,
#                                 True,
#                                 torch.tensor([1.25, 0.65])
#                              )
#                          ])
# def test_l1_dist_rotate(bboxes1, bboxes2, mode, aligned, clockwise, expexted):
#     result = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
#     print(result)
#     print('sth')


# 测试旋转<变化性>
@pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise", \
                         [
                            pytest.param
                             (
                                torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25],
                                    [0.5, 0.5, 0.3, 0.3, 1.25]]),
                                torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5],
                                    [0.3, 0.3, 0.5, 0.5, 1.5]]),
                                'iou',
                                True,
                                True,
                                id = "旋转变化性"
                             )
                         ])
def test1(bboxes1, bboxes2, mode, aligned, clockwise):
    result = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
    assert result[0] != result[1]

# 测试fix ratio的尺度<不变性>
@pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise", \
                         [
                            pytest.param
                             (
                                torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25],
                                    [50, 50, 30, 30, 0.25]]),
                                torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5],
                                    [30, 30, 50, 50, 0.5]]),
                                'iou',
                                True,
                                True,
                                id = "fix ratio尺度不变"
                             )
                         ])
def test2(bboxes1, bboxes2, mode, aligned, clockwise):
    result = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
    print(f'原值:{result[0]} 尺度变化值: {result[1]}')
    assert result[0] == pytest.approx(result[1], abs=1e-4)

 # 测试 不固定 ratio的尺度<变化性>
@pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise", \
                         [
                            pytest.param
                             (
                                torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25],
                                    [50, 50, 60, 60, 0.25]]),
                                torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5],
                                    [30, 30, 100, 100, 0.5]]),
                                'iou',
                                True,
                                True,
                                id = "fix ratio尺度不变"
                             )
                         ])
def test3(bboxes1, bboxes2, mode, aligned, clockwise):
    result = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
    print(f'原值:{result[0]} 尺度变化值: {result[1]}')
    assert result[0] != pytest.approx(result[1], abs=1e-4)

# 测试 顺时针逆时针的 《不变性》
@pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise", \
                         [
                            pytest.param
                             (
                                torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25],
                                    [0.5, 0.5, 0.3, 0.3, -0.25]]),
                                torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5],
                                    [0.3, 0.3, 0.5, 0.5, -0.5]]),
                                'iou',
                                True,
                                True,
                             )
                         ])
def test4(bboxes1, bboxes2, mode, aligned, clockwise):
    result = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
    print(f'原值:{result[0]} 顺时针逆时针变化: {result[1]}')
    assert result[0] == pytest.approx(result[1], abs=1e-4)



 # 测试与蒙特卡洛方法计算的iou是否一致
@pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise", \
                         [
                            pytest.param
                             (
                                torch.tensor([[339.15, 230.95, 308.2183, 33.1234, 1.5331],]),
                                torch.tensor([[339.15, 230.95, 307.2183, 33.1234, 1.5331],]),
                                'iou',
                                True,
                                True,
                                id = "fix ratio尺度不变"
                             )
                         ])
def test5(bboxes1, bboxes2, mode, aligned, clockwise):
    result = box_iou_rotated(bboxes1, bboxes2, mode, aligned, clockwise)
    monte_carlo_result = monte_carlo_iou(bboxes1[0], bboxes2[0])
    print(f'原值:{result[0]} 蒙特卡洛值: {monte_carlo_result}')
    assert result[0] == pytest.approx(monte_carlo_result, abs=1e-2)

 # 测试与蒙特卡洛方法计算的iou是否一致
@pytest.mark.parametrize("bboxes1, bboxes2, mode, aligned, clockwise", \
                         [
                            pytest.param
                             (
                                torch.tensor([339.15, 230.95, 308.2183, 33.1234, 1.5331]).reshape(1,-1),
                                torch.tensor([339.15, 230.95, 307.2183, 33.1234, 1.5331]).reshape(1,-1),
                                'iou',
                                True,
                                True,
                                id = "fix ratio尺度不变"
                             )
                         ])
def test6(bboxes1, bboxes2, mode, aligned, clockwise):
    device = torch.device('cuda:0')
    result = diff_iou_rotated_2d(bboxes1.to(device).reshape(1,1,-1), bboxes2.to(device).reshape(1,1,-1))
    result.to('cpu').detach().numpy()
    monte_carlo_result = monte_carlo_iou(bboxes1[0], bboxes2[0])
    print(f'原值:{result[0]} 蒙特卡洛值: {monte_carlo_result}')
    assert result[0] == pytest.approx(monte_carlo_result, abs=1e-2)


import numpy as np

def generate_rotated_box(cx, cy, w, h, theta):
    """
    生成旋转框的四个顶点坐标。
    """
    # 转换角度为弧度
    theta_rad = theta
    
    # 四个顶点的局部坐标（相对于中心点）
    dx = w / 2
    dy = h / 2
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])
    
    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    
    # 全局坐标
    rotated_corners = (rotation_matrix @ corners.T).T + np.array([cx, cy])
    return rotated_corners

def is_point_in_rotated_box(point, corners):
    """
    判断一个点是否在旋转框内（利用叉积法）。
    """
    x, y = point
    n = len(corners)
    for i in range(n):
        p1 = corners[i]
        p2 = corners[(i + 1) % n]
        edge = p2 - p1
        to_point = np.array([x, y]) - p1
        if np.cross(edge, to_point) < 0:
            return False
    return True

def monte_carlo_iou(box1, box2, num_samples=50000):
    """
    使用蒙特卡洛方法计算两个旋转框的IoU。
    """
    # 获取两个框的四个顶点
    corners1 = generate_rotated_box(*box1)
    corners2 = generate_rotated_box(*box2)
    
    # 获取联合区域的边界框
    all_corners = np.vstack((corners1, corners2))
    x_min, y_min = np.min(all_corners, axis=0)
    x_max, y_max = np.max(all_corners, axis=0)
    
    # 随机采样点
    samples = np.random.uniform([x_min, y_min], [x_max, y_max], (num_samples, 2))
    
    # 判断点是否在各框内
    in_box1 = np.array([is_point_in_rotated_box(pt, corners1) for pt in samples])
    in_box2 = np.array([is_point_in_rotated_box(pt, corners2) for pt in samples])
    
    # 计算IoU
    intersection = np.sum(in_box1 & in_box2)
    union = np.sum(in_box1 | in_box2)
    iou = intersection / union
    return iou