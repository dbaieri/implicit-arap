from __future__ import annotations

import iarap.data as data
import iarap.model.nn as nn

from typing import Dict, Type
from dataclasses import dataclass, field
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig



class SDFTrainer:

    def __init__(self, config: SDFTrainerConfig):
        
        self.config = config

        self.make_data()
        self.make_model()
        self.make_optimizers()

    def make_data(self):
        pass

    def make_model(self):
        pass

    def make_optimizers(self):
        pass

    def run(self):
        print(self.config)
        print("Starting SDF training procedure.")
        for it in tqdm(range(self.config.num_steps)):
            pass



@dataclass
class SDFTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: SDFTrainer)

    num_steps: int = 1

