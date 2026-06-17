import os.path as osp

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class StrokeNCCTDataset(BaseSegDataset):
    # Class 0: Background (Hitam), Class 1: Stroke Iskemik (Putih)
    METAINFO = dict(
        classes=('background', 'stroke'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self,
                 stroke_positive_only=False,
                 stroke_positive_file=None,
                 **kwargs):
        self.stroke_positive_only = stroke_positive_only
        self.stroke_positive_set = None
        if stroke_positive_only and stroke_positive_file:
            with open(stroke_positive_file) as f:
                self.stroke_positive_set = set(
                    line.strip() for line in f if line.strip())
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

    def load_data_list(self):
        """Load data list, optionally filtering to stroke-positive images."""
        data_list = super().load_data_list()
        if self.stroke_positive_only and self.stroke_positive_set is not None:
            filtered = []
            for item in data_list:
                fname = osp.basename(item['img_path'])
                if fname in self.stroke_positive_set:
                    filtered.append(item)
            return filtered
        return data_list