from __future__ import annotations

import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader

from iarap.config.base_config import InstantiateConfig


class Trainer:

    def __init__(self, config: InstantiateConfig):
        
        self.config = config

        self.setup_data()
        self.setup_model()
        self.setup_optimizer()

    def setup_data(self):
        self.data = self.config.data.setup()
        self.loader = DataLoader(self.data.dataset)

    def setup_model(self):
        self.model = self.config.model.setup().to(self.config.data.device)
        self.loss = self.config.loss.setup()

    def setup_optimizer(self):
        param_groups = list(self.model.parameters())
        self.optimizer = self.config.optimizer.setup(params=param_groups)
        self.scheduler = self.config.scheduler.setup(optimizer=self.optimizer)

    def run(self):
        print(self.config)
        print("Running {} procedure".format(self.__class__.__name__))
        self.logger = wandb.init(project='iARAP', config=self.config.to_dict())
        for it in tqdm(range(self.config.num_steps)):
            for batch in self.loader:
                self.optimizer.zero_grad()

                loss_dict = self.train_step(batch)

                loss = sum(loss_dict.values())
                loss_dict.update({'loss': loss})
                loss.backward()
                self.optimizer.step()

                self.logger.log(loss_dict)

            self.scheduler.step()

        self.postprocess()

    def train_step(self, batch):
        raise NotImplementedError()
    
    def postprocess(self):
        return