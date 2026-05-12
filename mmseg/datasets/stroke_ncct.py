from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class StrokeNCCTDataset(BaseSegDataset):
    # Class 0: Background (Hitam), Class 1: Stroke Iskemik (Putih)
    METAINFO = dict(
        classes=('background', 'stroke'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )
    
    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png', 
            seg_map_suffix='.png', 
            **kwargs)