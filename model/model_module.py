import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper import ModelWrapper

class ModelModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})
        self.automatic_optimization = False

        # build model
        self.model = ModelWrapper(self.hparam.arch, kwargs).net

        # Synthetic Dataset
        self.register_buffer("image_syn",
            torch.randn((num_classes * ipc, channel, im_size[0], im_size[1]),dtype=torch.float,requires_grad=True)
        )
        self.register_buffer("label_syn",
            torch.randn([np.ones(ipc)*i for i in range(num_classes)],dtype=torch.long,requires_grad=False).view(-1)
        )

    def configure_optimizers(self):
        optimizer_net = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr_net,
        )
        optimizer_img = torch.optim.SGD(
                [self.image_syn,],
                lr=self.hparams.lr_img,
                momentum=0.5,
        )
        return optimizer_img, optimizer_net

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)
    
    def matching_gradient_loss(self, real, syn):
        return None

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        # Reset Network and Optimizer
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.optimizers()[0] = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr_net,
        )

    def unpack_batch(self, batch):
        # Split batch(whole data of same class) to preferred batch size
        batch_size = self.hparams.batch_size
        images, labels = batch
        num_images = images.size(0)
        idx_shuffle = np.random.permutation(num_images)[:batch_size]
        img = images[idx_shuffle]
        lab = labels[idx_shuffle]
        return img, lab

    def training_step(self, batch, batch_idx):
        optimizer_img, optimzer_net = self.optimizers()
        img_real, lab_real = self.unpack_batch(batch)
        
        
        opt = self.optimizers()
        opt.zero_grad()
        
        
        
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()
        
        # Update Syn Dataset
        for cl in classes:
            x_real, x_syn
            pred_real = model(x_real)
            grad_real
            pred_syn = model(x_syn)
            grad_syn
            matching loss
        loss update
        # Update model w/Syn Dataset
        for inner in inner_loop:
            model(x_syn)
        x, y = batch
        outputs = self.model(x)
    
    def training_epoch_end(self, batch):
        # predictions from each GPU
        predictions = batch_parts["pred"]
        # losses from each GPU
        losses = batch_parts["loss"]

        gpu_0_prediction = predictions[0]
        gpu_1_prediction = predictions[1]

        # do something with both outputs
        return (losses[0] + losses[1]) / 2
        pass

    def validation_step(self, batch):
        pass

    def validation_epoch_end(self):
        pass



for experiment:
    # init syn data
    for iteration:
        # Evaluate Syn Dataset
        if iteration in eval_it_pool:
            for model_eval in model_eval_pool:
                for eval_epoch:
                    # train model
                    # eval model
                # Log
        # Train Syn Dataset
        for outer in outer_loop:
            # Update Syn Dataset
            for cl in classes:
                x_real, x_syn
                pred_real = model(x_real)
                grad_real
                pred_syn = model(x_syn)
                grad_syn
                matching loss
            loss update
            # Update model w/Syn Dataset
            for inner in inner_loop:
                model(x_syn)
        log loss
    log and save

