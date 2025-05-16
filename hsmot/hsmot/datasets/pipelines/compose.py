# Copyright (c) OpenMMLab. All rights reserved.

from hsmot.mmlab.hs_mmdet import Compose
from typing import List, Optional, Union, Dict
import mmengine
import numpy as np
import time

class MotCompose(Compose):
 
    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            # start_time = time.time()
            data = t(data)
            if data is None:
                return None
            if any(item is None for item in data):
                return None
            # end_time = time.time()
            # print(f'{t.__class__.__name__} time: {end_time - start_time}')

        return data

class MotRandomChoice():
    """Process data with a randomly chosen transform from given candidates.

    Args:
        transforms (list[list]): A list of transform candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """          
    def __init__(self,
                 transforms: List,
                 prob: Optional[List[float]] = None):

        if prob is not None:
            assert mmengine.is_seq_of(prob, float)
            assert len(transforms) == len(prob), \
                '``transforms`` and ``prob`` must have same lengths. ' \
                f'Got {len(transforms)} vs {len(prob)}.'
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [MotCompose(transforms) for transforms in transforms]

    def __iter__(self):
        return iter(self.transforms)

    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)
    
    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str
    
    def __call__(self, results_list):
        pipline_index = self.random_pipeline_index()
        results_list = self.transforms[pipline_index](results_list)
        return results_list