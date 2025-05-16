from hsmot.mmlab.hs_mmdet import DefaultFormatBundle, Collect, to_tensor, DC
from hsmot.mmlab.hs_mmcv import imdenormalize
import cv2
from datetime import datetime
import os
from hsmot.mmlab.hs_mmrotate import obb2poly_np
import numpy as np

class MotDefaultFormatBundle(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose & to tensor, (2)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        super(MotDefaultFormatBundle, self).__init__(img_to_float=img_to_float,pad_val=pad_val)

    def __call__(self, results_list):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        for results in results_list:
            if 'img' in results:
                img = results['img']
                if self.img_to_float is True and img.dtype == np.uint8:
                    # Normally, image is of uint8 type without normalization.
                    # At this time, it needs to be forced to be converted to
                    # flot32, otherwise the model training and inference
                    # will be wrong. Only used for YOLOX currently .
                    img = img.astype(np.float32)
                # add default meta keys
                results = self._add_default_meta_keys(results)
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # If image is not contiguous, use
                # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
                # If image is already contiguous, use
                # `torch.permute()` followed by `torch.contiguous()`
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if not img.flags.c_contiguous:
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                    img = to_tensor(img)
                else:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                results['img'] = DC(
                    img, padding_value=self.pad_val['img'], stack=True)
            for key in ['proposal_scores', 'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_trackids']:
                if key not in results:
                    continue
                results[key] = DC(to_tensor(results[key]))
            if 'gt_masks' in results:
                results['gt_masks'] = DC(
                    results['gt_masks'],
                    padding_value=self.pad_val['masks'],
                    cpu_only=True)
            if 'gt_semantic_seg' in results:
                results['gt_semantic_seg'] = DC(
                    to_tensor(results['gt_semantic_seg'][None, ...]),
                    padding_value=self.pad_val['seg'],
                    stack=True)
        return results_list

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        return super(MotDefaultFormatBundle, self)._add_default_meta_keys(results=results)

class MotCollect(Collect):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        super(MotCollect, self).__init__(keys=keys, meta_keys=meta_keys)

    def __call__(self, results_list):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """
        data_list = []
        for results in results_list:
            data_list.append(super(MotCollect, self).__call__(results))
        return data_list

class MotShow():

    def __init__(self, save_path,  mean=None, std=None, to_bgr=True, version='le135', img_name_tail = None, show_proposals=False):
        """
        初始化类。
        :param save_path: 图像保存路径。
        :param to_rgb: 是否将图像转换为RGB格式。
        """
        if mean is None:
            self.mean = np.array([0, 0, 0], dtype=np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)
        if std is None:
            self.std = np.array([1, 1, 1], dtype=np.float32)
        else:
            self.std = np.array(std, dtype=np.float32)

        self.save_path = save_path
        self.to_bgr = to_bgr
        self.version = version
        self.img_name_tail = img_name_tail
        self.show_proposals = show_proposals
        os.makedirs(save_path, exist_ok=True)

    def __call__(self, results_list):
        """
        处理results, 保存图像并绘制目标框、标签和跟踪ID。
        :param results: 包含图像及标注信息的字典。
        """
        # 获取时间戳，用于图像命名,包括毫秒级
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        for img_index, results in enumerate(results_list):
            # 提取图像和标注信息
            img = results['img']
            gt_bboxes = results.get('gt_bboxes', [])
            gt_labels = results.get('gt_labels', [])
            gt_trackids = results.get('gt_trackids', [])
            
            multi_spec_img = img.shape[2] > 3
            
            # 反归一化
            if img.dtype == np.uint8:                         
                img = img.copy().astype(np.float32)
            img = imdenormalize(img, self.mean, self.std, to_bgr=self.to_bgr)
            
            # 如果是多光谱, 随机选取三通道
            if multi_spec_img:
                random_3_channels = np.random.choice(range(img.shape[2]), 3, replace=False)
                img = np.ascontiguousarray(img[:, :, random_3_channels])
                            
            if self.show_proposals:
                for i, (bbox, score) in enumerate(zip(results['proposals'], results['proposal_scores'])):
                    color = (0, 0, 255)
                    bbox = np.concatenate([bbox, np.array([score])])    
                    bbox = bbox[None, :]
                    four_points = obb2poly_np(bbox, self.version)[0].astype(np.int16)
                    # 绘制边框
                    cv2.line(img, (four_points[0], four_points[1]), (four_points[2],four_points[3]), color=color,thickness=1)
                    cv2.line(img, (four_points[2], four_points[3]), (four_points[4],four_points[5]), color=color,thickness=1)
                    cv2.line(img, (four_points[4], four_points[5]), (four_points[6],four_points[7]), color=color,thickness=1)
                    cv2.line(img, (four_points[6], four_points[7]), (four_points[0],four_points[1]), color=color,thickness=1)
                    # 显示得分
                    text = f"Score: {score:.2f}"
                    cv2.putText(img, text, (four_points[0], four_points[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 绘制每个目标框
            for i, bbox in enumerate(gt_bboxes):
                x,y,w,h,a = map(float, bbox)
                label = gt_labels[i] if i < len(gt_labels) else 'N/A'
                track_id = gt_trackids[i] if i < len(gt_trackids) else 'N/A'
                color = (0, 255, 0)  # 绘制框的颜色
                
                # 转四点坐标
                '''
                    obb2poly_np要求输入带score
                '''
                bbox = bbox[None, :]
                bbox_with_score = np.hstack([bbox, np.ones((bbox.shape[0], 1)) * 1.0])  # 添加 score，默认为 1.0
                four_points = obb2poly_np(bbox_with_score, self.version)[0].astype(np.int16)
                # 绘制边框
                cv2.line(img, (four_points[0], four_points[1]), (four_points[2],four_points[3]), color=color,)
                cv2.line(img, (four_points[2], four_points[3]), (four_points[4],four_points[5]), color=color,)
                cv2.line(img, (four_points[4], four_points[5]), (four_points[6],four_points[7]), color=color,)
                cv2.line(img, (four_points[6], four_points[7]), (four_points[0],four_points[1]), color=color,)             
                # 显示标签和跟踪ID
                text = f"Label: {label}, ID: {track_id}"
                cv2.putText(img, text, (four_points[0], four_points[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if multi_spec_img:
                img_name = f"{timestamp}_ch{('|').join([str(v) for v in random_3_channels])}_{('_').join(results['img_info']['filename'].split(os.sep)[-2:])}".replace('npy', 'jpg')
            else:
                img_name = f"{timestamp}_{('_').join(results['img_info']['filename'].split(os.sep)[-2:])}".replace('npy', 'jpg')
                
            if self.img_name_tail is not None:
                img_name = img_name.replace('.png', f"_{self.img_name_tail}.png").replace('.jpg', f"_{self.img_name_tail}.jpg")
                
            save_file = os.path.join(self.save_path, img_name)
            
            # 保存图像
            cv2.imwrite(save_file, img)
            print(f"Image saved at: {save_file}")
        return results_list
        