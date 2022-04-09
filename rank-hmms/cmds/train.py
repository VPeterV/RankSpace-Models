# -*- coding: utf-8 -*-
import time
# from datetime import datetime, timedelta
from cmds.cmd import CMD
from lm.helpers.metric import LikelihoodMetric, Metric
from lm.datasets.data_module import DataModule
from lm.datasets.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from lm.helpers.utils import get_logger, \
                            create_save_path, get_optimizer, \
                            get_scheduler, get_mask_lengths, weight_initializer, \
                            get_total_iter
from pathlib import Path
import nni
import wandb
from functools import partial

class Train(CMD):

    def __call__(self, args):

        self.args = args
        self.device = args.device
        self.get_mask_lengths = get_mask_lengths
        
        dataset = DataModule(args)
        self.V = dataset.V
        self.model = self.get_model(dataset)
        weight_initializer(self.model, args.init, args.special_weights.split('I'))
        log = get_logger(args)
        create_save_path(args)
        self.optimizer = get_optimizer(args.optimizer, self.model)
        self.scheduler = get_scheduler(args.scheduler, self.optimizer)
        log.info("Create the model")
        log.info(f"{self.model}\n")
        total_time = 0
        best_e, best_dev_ppl = 0, Metric()
        log.info(self.optimizer)
        log.info(args)
        train_loader = dataset.train_dataloader
        eval_loader = dataset.val_dataloader
        test_loader = dataset.test_dataloader
        
        train_loader_autodevice = DataPrefetcher(train_loader, device=self.device)
        eval_loader_autodevice = DataPrefetcher(eval_loader, device=self.device)
        test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
        
        train_len = get_total_iter(train_loader_autodevice)
        val_len = get_total_iter(eval_loader_autodevice)
        test_len = get_total_iter(test_loader_autodevice)
        
        test_model_partial = partial(self.__test_model, test_loader, test_len, log)

        '''
        Training
        '''
        train_arg = args.train
        self.train_arg = train_arg
        
        if args.wandb:
            self.wandb_step = -1
        
        for epoch in range(0, train_arg.max_epochs):
            '''
            Auto .to(self.device)
            '''

            train_loader_autodevice = DataPrefetcher(train_loader, device=self.device)
            eval_loader_autodevice = DataPrefetcher(eval_loader, device=self.device)
            
            start = time.time()

            train_ppl_metric = self.train(train_loader_autodevice, train_len)
            
            log.info(f"Epoch {epoch} / {train_arg.max_epochs}:")

            dev_ppl_metric = self.evaluate(eval_loader_autodevice, val_len)
            
            log.info(f"{'dev ppl:':6}   {dev_ppl_metric}")
            log.info("lr:{}".format(self.optimizer.param_groups[0]["lr"]))
            
            if args.nni:
                nni.report_intermediate_result(dev_ppl_metric.score)

            t = time.time() - start
                
            # save the model if it is the best so far
            best_dev_ppl, best_e = self.__update_best_valid(dev_ppl_metric, best_dev_ppl, log, epoch, best_e, t, test_model_partial)
                        
            total_time += t
            
            if args.wandb:
                wandb.log({
                    "train_loss": train_ppl_metric.avg_likelihood,
                    "train_ppl": train_ppl_metric.perplexity,
                    "valid_loss": dev_ppl_metric.avg_likelihood,
                    "valid_ppl": dev_ppl_metric.perplexity,
                    "best_valid_loss": best_dev_ppl.avg_likelihood,
                    "best_valid_ppl": best_dev_ppl.perplexity,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": t,
                    "epoch": epoch,
                }, step=self.wandb_step)
                
                wandb.run.summary["best_valid_ppl"] = best_dev_ppl.perplexity
                wandb.run.summary["best_valid_loss"] = best_dev_ppl.avg_likelihood
            
            self.scheduler.step(dev_ppl_metric.evidence)
            
            if train_arg.patience > 0 and epoch - best_e >= train_arg.patience:
                break
        
        if args.nni:
            nni.report_final_result(best_dev_ppl.score)
        
    def __update_best_valid(self, dev_ppl_metric, best_dev_ppl, log, epoch, best_e, t, test_model_partial):
        if dev_ppl_metric < best_dev_ppl:
            best_dev_ppl = dev_ppl_metric
            best_e = epoch            
            torch.save(
               {         
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),},
               f = self.args.save_dir + "/best.pt"
            )
            log.info(f"{t}s elapsed (saved)")
            
            test_model_partial()
                
        else:
            log.info(f"{t}s elapsed")
            
        log.info(f"{'best dev ppl:':6}   {best_dev_ppl} {'best epoch:':6} {best_e}\n")
            
        return best_dev_ppl, best_e
        
    def __test_model(self, test_loader, test_len, log):
        test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
        test_ppl_metric = self.evaluate(test_loader_autodevice, test_len)
        log.info(f"{'test ppl:':6}   {test_ppl_metric}")
        
        if self.args.wandb:
            wandb.log({
                "test_loss": test_ppl_metric.avg_likelihood,
                "test_ppl": test_ppl_metric.perplexity,
            }, step=self.wandb_step)            
