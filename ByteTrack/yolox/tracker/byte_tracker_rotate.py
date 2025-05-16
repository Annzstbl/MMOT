import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter_rotate import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
# from mmrotate.core.bbox.transforms import obb2poly, poly2obb
from hsmot.util.bbox import poly2obb_np_woscore, obb2poly_np_woscore
from mmcv.ops import box_iou_rotated

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, xyxyxyxy, score, cls):
        '''
            卡尔曼滤波状态
            x, y, a, h, theta, vx, vy, va, vh, vtheta
        '''

        # wait activate
        self.xyxyxyxy = np.asarray(xyxyxyxy, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.cls = cls
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[8] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][8] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyxyxyxy_to_xyaha(self.xyxyxyxy))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.xyxyxyxy_to_xyaha(new_track.xyxyxyxy)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.xyxyxyxy = new_track.xyxyxyxy

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.xyxyxyxy = new_track.xyxyxyxy
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.xyxyxyxy_to_xyaha(self.xyxyxyxy))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls#

    # @property
    # # @jit(nopython=True)
    # def tlwh(self):
    #     """Get current position in bounding box format `(top left x, top left y,
    #             width, height)`.
    #     """
    #     if self.mean is None:
    #         return self._tlwh.copy()
    #     ret = self.mean[:4].copy()
    #     ret[2] *= ret[3]
    #     ret[:2] -= ret[2:] / 2
    #     return ret


    @property
    def xywha(self):
        if self.mean is None:
            ret = self.xyxyxyxy_to_xyaha(self.xyxyxyxy)
        else:
            ret = self.mean[:5].copy()
        ret[2] *= ret[3]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def xyxyxyxy_to_xyaha(xyxyxyxy, version='le135'):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height, angle)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(xyxyxyxy).copy()
        ret = poly2obb_np_woscore(ret, version)# xywha
        ret[2] /= ret[3]
        return ret

    def to_xyaha(self):
        return self.xyxyxyxy_to_xyaha(self.xyxyxyxy)


    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

def fuse_rotate_iou_and_cls_distance(atracks, btracks):
    return rotate_iou_distance(atracks, btracks) + 0.5 * cls_distance(atracks, btracks)

def rotate_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        axyahas = atracks
        bxyahas = btracks
    else:
        axyahas = [track.xywha for track in atracks]
        bxyahas = [track.xywha for track in btracks]

    axyahas_torch = torch.tensor(axyahas, dtype=torch.float32) if len(axyahas) > 0 else torch.zeros((0, 5), dtype=torch.float32)
    bxyahas_torch = torch.tensor(bxyahas, dtype=torch.float32) if len(bxyahas) > 0 else torch.zeros((0, 5), dtype=torch.float32)
    

    _ious = box_iou_rotated(axyahas_torch, bxyahas_torch)
    cost_matrix = 1 - _ious

    return cost_matrix.cpu().numpy()

def cls_distance(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        axyahas = atracks
        bxyahas = btracks
    else:
        axyahas = [track.cls for track in atracks]
        bxyahas = [track.cls for track in btracks]

    # cost_matrix在cls相同的地方为0, 不同的地方为1
    cost_boolean = np.array(axyahas)[:, None] != np.array(bxyahas)[None, :]
    cost_matrix = cost_boolean.astype(np.float32)
    return cost_matrix
    

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # xy xy xy xy cls score
        assert output_results.shape[1] == 10
        scores = output_results[:, 9]
        bboxes = output_results[:, :8]
        clses = output_results[:, 8]


        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = clses[remain_inds]
        cls_second = clses[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(xyxyxyxy, s, cls) for
                          (xyxyxyxy, s, cls) in zip(dets, scores_keep, cls_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = fuse_rotate_iou_and_cls_distance(strack_pool, detections)
 
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(xyxyxyxy, s, cls) for
                          (xyxyxyxy, s, cls) in zip(dets_second, scores_second, cls_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = fuse_rotate_iou_and_cls_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = fuse_rotate_iou_and_cls_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = fuse_rotate_iou_and_cls_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
