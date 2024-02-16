from __future__ import annotations

import iarap.data as data
import iarap.model.nn as nn

from typing import Dict, Type
from dataclasses import dataclass, field
from tqdm import tqdm

from iarap.config.base_config import InstantiateConfig



class DeformTrainer:

    def __init__(self, config: DeformTrainerConfig):
        self.config = config

    def run(self):
        print(self.config)
        print("Starting SDF deformation procedure.")
        for it in tqdm(range(self.config.num_steps)):
            pass



@dataclass
class DeformTrainerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: DeformTrainer)

    num_steps: int = 1