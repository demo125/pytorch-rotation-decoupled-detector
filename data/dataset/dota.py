

from config import CATEGORY_DOTA_V10 as NAMES

from .dataset import DetDataset


class DOTA(DetDataset):
    def __init__(self, json_path, aug=None):
        super(DOTA, self).__init__(json_path, NAMES, aug)
