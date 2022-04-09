# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from lm.helpers.metric import LikelihoodMetric
from lm.models import  ProjRankSpace

import time as timep

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, iter, total_iter):
        self.model.train()
        t = tqdm(iter, total=total_iter,  position=0, leave=True)
        train_arg = self.args.train
        
        ppl_metric = LikelihoodMetric()
        cnt = 0
        for i, batch in enumerate(t):
            

            text = batch
            
            mask, lengths, n_tokens = self.get_mask_lengths(text, self.V)
            
            if self.model.timing:
                start_forward = timep.time()

            self.optimizer.zero_grad()

            # Do not support bptt now
            losses, _ = self.model.score(text, lpz=None, last_states=None, mask=mask, lengths=lengths)
            
            if self.model.timing:
                print(f"forward time: {timep.time() - start_forward}")
            
            ppl_metric(losses.evidence, n_tokens, losses.elbo)
                        
            loss = -losses.loss / n_tokens
            
            if self.model.timing:
                start_backward = timep.time()
                
            loss.backward()
             
            if self.model.timing:
                print(f"backward time: {timep.time() - start_backward}")
                
            if train_arg.clip > 0:
                gradnorm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
                
            self.optimizer.step()
            
            if self.args.wandb:
                self.wandb_step += 1
                wandb.log({
                "running_training_loss": ppl_metric.avg_likelihood,
                "running_training_ppl": ppl_metric.perplexity,
                "running_training_elbo": ppl_metric.elbo,
                "gradnorm": gradnorm,
            }, step=self.wandb_step)
            
            t.set_postfix(loss=loss.item())

        return ppl_metric


    @torch.no_grad()
    def evaluate(self, loader, total_iter, model=None):
        if model == None:
            model = self.model
        model.eval()
        ppl_metric = LikelihoodMetric()
        t = tqdm(loader, total=total_iter,  position=0, leave=True)
        for i, batch in enumerate(t):
            # text = batch.text
            text = batch
            
            mask, lengths, n_tokens = self.get_mask_lengths(text, self.V)
        
            losses, _ = model.score(text, lpz=None, last_states=None, mask=mask, lengths=lengths)
                        
            ppl_metric(losses.evidence, n_tokens, losses.elbo)
            
        return ppl_metric
        
    def get_model(self, dataset):
        if  self.args.model.model_name == 'projrankspace':
            return ProjRankSpace(dataset.V, self.args).to(self.device)     
            
        else:
            raise AttributeError("Not implemented model")




