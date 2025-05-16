from hsmot.mmlab.hs_mmrotate import obb2poly_np, poly2obb_np
from mmcv.ops import box_iou_rotated
import numpy as np
from torch import tensor
import torch



def poly2obb_np_woscore(polys, version='le135'):
    '''
        没有score
    '''
    # polys支持两个维度 一个是 [N, 8] 一个是 [8]
    assert polys.shape[-1] == 8
    if polys.ndim == 1:
        return np.array(poly2obb_np(polys, version=version))
    else:
        ret = []
        for poly in polys:
            result = poly2obb_np(poly, version=version)
            if result == None:
                cx = (poly[0] + poly[2] + poly[4] + poly[6]) / 4
                cy = (poly[1] + poly[3] + poly[5] + poly[7]) / 4
                result = [cx, cy, -1, -1, 0]#TODO框太小时候的处理
            ret.append(result)
        return np.array(ret)

    
def obb2poly_np_woscore(obbs, version='le135'):
    '''
        没有score
    '''
    # obbs支持两个维度 一个是 [N, 5] 一个是 [5]
    assert obbs.shape[-1] == 5

    # 需要先加一个score,最后再删掉
    obbs = obbs.reshape(-1, 5)
    obbs = np.concatenate([obbs, np.ones((obbs.shape[0], 1))], axis=1)

    assert version=='le135'
    ret = obb2poly_np(obbs, version=version)

    if obbs.ndim == 1:
        return ret[0][:-1]
    else:
        return ret[:, :-1]
    

def poly_iou(apolys, bpolys, aligned=False):
    '''
        计算两个多边形的iou
        apolys: [N, 8]
        bpolys: [M, 8]
    '''
    if aligned:
        assert apolys.shape[0] == bpolys.shape[0]
    apolys = apolys[:, :8]
    bpolys = bpolys[:, :8]

    aobbs = poly2obb_np_woscore(apolys, version='le135')
    bobbs = poly2obb_np_woscore(bpolys, version='le135')

    aobbs_tensor = torch.tensor(aobbs, dtype=torch.float32)
    bobbs_tensor = torch.tensor(bobbs, dtype=torch.float32)

    ret = box_iou_rotated(aobbs_tensor, bobbs_tensor, aligned=aligned)

    return ret.cpu().numpy()