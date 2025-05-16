"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np

from filterpy.kalman import KalmanFilter
from hsmot.util.bbox import poly2obb_np_woscore, obb2poly_np_woscore, poly_iou

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   bb_gt = np.expand_dims(bb_gt, 0)
#   bb_test = np.expand_dims(bb_test, 1)
  
#   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#   w = np.maximum(0., xx2 - xx1)
#   h = np.maximum(0., yy2 - yy1)
#   wh = w * h
#   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
#   return(o)  


def convert_bbox_to_z(bbox):
  '''
    from xyxyxyxy to [x, y, theta, area, w/h/ratio]
  '''
  ret = poly2obb_np_woscore(bbox[:8])
  x = ret[0]
  y = ret[1]
  theta = ret[4]
  s = ret[2] * ret[3]
  r = ret[2] / float(ret[3])
  return np.array([x, y, theta, s, r]).reshape((5, 1))
  # """
  # Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
  #   [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
  #   the aspect ratio
  # """
  # w = bbox[2] - bbox[0]
  # h = bbox[3] - bbox[1]
  # x = bbox[0] + w/2.
  # y = bbox[1] + h/2.
  # s = w * h    #scale is just area
  # r = w / float(h)
  # return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  '''
    from [x, y, theta, area, w/h/ratio] to xyxyxyxy
  '''
  w = np.sqrt(x[3] * x[4])
  h = x[3] / w

  xywha = np.array([x[0], x[1], w, h, x[2]]).reshape(5)
  ret = obb2poly_np_woscore(xywha)
  return ret


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox, cls,score):
    """
    Initialises a tracker using initial bounding box.
    """

    #define constant velocity model
    self.kf = KalmanFilter(dim_x=9, dim_z=5) 
    self.kf.F = np.array([
      [1,0,0,0,0,1,0,0,0], # u = u + du
      [0,1,0,0,0,0,1,0,0], # v = v + dv
      [0,0,1,0,0,0,0,1,0], # theta = theta + dtheta
      [0,0,0,1,0,0,0,0,1], # s = s + ds
      [0,0,0,0,1,0,0,0,0], # r = r
      [0,0,0,0,0,1,0,0,0],
      [0,0,0,0,0,0,1,0,0],
      [0,0,0,0,0,0,0,1,0],
      [0,0,0,0,0,0,0,0,1]
      ])# 状态转移矩阵
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0]])# 观测矩阵 只取位置区域

    self.kf.R[2:,2:] *= 10.   #z空间 测量噪声协方差矩阵 认为角度 面积 宽高比不确定度较高
    self.kf.P[5:,5:] *= 1000. #x空间 give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.          #状态方差矩阵， 给与高不确定度
    self.kf.Q[-1,-1] *= 0.01  #过程噪声协方差 第七项降低 认为面积变化较小
    self.kf.Q[5:,5:] *= 0.01  #过程噪声协方差，从第五项降低 认为速度变化较小


    self.kf.x[:5] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.cls = cls
    self.score = score
    self.xyxyxyxy = bbox

  def update(self,bbox, cls, score):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

    # 更新
    self.cls = cls
    self.score = score
    self.xyxyxyxy = bbox

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[7]+self.kf.x[3])<=0):
      self.kf.x[7] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(np.concatenate([convert_x_to_bbox(self.kf.x), np.array([self.cls, self.score]).reshape(1, -1)], axis=-1))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)
  
  def get_cls(self):
    return self.cls
  
  def get_score(self):
    return self.score

  def get_xyxyxyxy_from_detection(self):
    return self.xyxyxyxy


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  # iou_matrix = iou_batch(detections, trackers)
  iou_matrix = poly_iou(detections, trackers)#! 这里是计算iou, 不是计算cost
  a_cls = detections[:, 8].reshape(-1)
  b_cls = trackers[:, 8].reshape(-1)
  cls_matrix = a_cls[:, None] != b_cls[None, :]
  cls_matrix = cls_matrix.astype(np.float32) #! 这里是cost


  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      all_matrx = (1-iou_matrix) + cls_matrix
      matched_indices = linear_assignment(all_matrx)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, det_thresh, max_age=30, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    self.det_thresh = det_thresh

  def update(self, output_results, img_info, img_size):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # post_process detections
    try:
      output_results = output_results.cpu().numpy()
    except:
      pass
    scores = output_results[:, 9]
    bboxes = output_results[:, :8]  # x1y1x2y2x3y3x4y4
    clses = output_results[:, 8] 
    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes /= scale
    dets = np.concatenate((bboxes, np.expand_dims(clses, axis=-1), np.expand_dims(scores, axis=-1), ), axis=1)
    remain_inds = scores > self.det_thresh
    dets = dets[remain_inds]
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 10))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7], pos[8], pos[9]]# pos + cls + score
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :8], dets[m[0], 8], dets[m[0], 9])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:8], dets[i, 8], dets[i, 9])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        # d = trk.get_state()[0]
        d = trk.get_xyxyxyxy_from_detection()
        cls = trk.get_cls()
        score = trk.get_score()
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d, [cls], [score],[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,10))