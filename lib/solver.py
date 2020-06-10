'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter


sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper import get_loss
from utils.eta import decode_eta

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_lang_loss: {train_lang_loss}
[loss] train_obj_cls_loss: {train_obj_cls_loss}
[sco.] train_lang_acc: {train_lang_acc}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_iou_rate_25: {train_iou_rate_25}
[sco.] train_iou_rate_5: {train_iou_rate_5}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_lang_loss: {train_lang_loss}
[train] train_obj_cls_loss: {train_obj_cls_loss}
[train] train_lang_acc: {train_lang_acc}
[train] train_ref_acc: {train_ref_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_iou_rate_25: {train_iou_rate_25}
[train] train_iou_rate_5: {train_iou_rate_5}
[val]   val_loss: {val_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_obj_cls_loss: {val_obj_cls_loss}
[val]   val_lang_loss: {val_lang_loss}
[val]   val_lang_acc: {val_lang_acc}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_iou_rate_25: {val_iou_rate_25}
[val]   val_iou_rate_5: {val_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] lang_loss: {lang_loss}
[loss] ref_loss: {ref_loss}
[loss] obj_cls_loss: {obj_cls_loss}
[sco.] lang_acc: {lang_acc}
[sco.] ref_acc: {ref_acc}
[sco.] obj_acc: {obj_acc}
[sco.] iou_rate_25: {iou_rate_25}
[sco.] iou_rate_5: {iou_rate_5}
"""

class Solver():
    def __init__(self, model, config, dataloader, optimizer, stamp, val_step=10, use_lang_classifier=True, use_max_iou=False):
        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step
        self.use_lang_classifier = use_lang_classifier
        self.use_max_iou = use_max_iou

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "lang_loss": float("inf"),
            "obj_cls_loss": float("inf"),
            "lang_acc": -float("inf"),
            "ref_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf")
        }

        # log
        # contains all necessary info for all phases
        self.log = {
            phase: {
                # info
                "forward": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                "ref_loss": [],
                "lang_loss": [],
                "obj_cls_loss": [],
                # scores (float, not torch.cuda.FloatTensor)
                "lang_acc": [],
                "ref_acc": [],
                "obj_acc": [],
                "iou_rate_0.25": [],
                "iou_rate_0.5": []
            } for phase in ["train", "val"]
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step
        
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        _, data_dict = get_loss(data_dict, self.config, True, self.use_lang_classifier, self.use_max_iou)

        # dump

        self._running_log["lang_loss"] = data_dict["lang_loss"]
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["obj_cls_loss"] = data_dict["obj_cls_loss"]
        self._running_log["loss"] = data_dict["loss"]
      
    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                "lang_loss": 0,
                "obj_cls_loss": 0,
                # acc
                "lang_acc": 0,
                "ref_acc": 0,
                "obj_acc": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(True):
                # forward
                start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train":
                    start = time.time()
                    self._backward()
                    self.log[phase]["backward"].append(time.time() - start)
            
            # eval
            start = time.time()
            self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
            self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
            self.log[phase]["obj_cls_loss"].append(self._running_log["obj_cls_loss"].item())

            self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
            self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)

                # dump log
                self._dump_log("train")
                self._global_iter_id += 1


        # check best
        if phase == "val":
            cur_criterion = "ref_acc"
            cur_best = np.mean(self.log[phase][cur_criterion])
            if cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
                self.best["lang_loss"] = np.mean(self.log[phase]["lang_loss"])
                self.best["obj_cls_loss"] = np.mean(self.log[phase]["obj_cls_loss"])
                self.best["lang_acc"] = np.mean(self.log[phase]["lang_acc"])
                self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
                self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
                self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
                self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _eval(self, data_dict):
        # dump
        self._running_log["lang_acc"] = data_dict["lang_acc"].item()
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "ref_loss", "lang_loss", "obj_cls_loss"],
            "score": ["lang_acc", "ref_acc", "obj_acc", "iou_rate_0.25", "iou_rate_0.5"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_obj_cls_loss=round(np.mean([v for v in self.log["train"]["obj_cls_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_obj_cls_loss=round(np.mean([v for v in self.log["train"]["obj_cls_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
            val_obj_cls_loss=round(np.mean([v for v in self.log["val"]["obj_cls_loss"]]), 5),
            val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
            val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
        )
        self._log(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            lang_loss=round(self.best["lang_loss"], 5),
            obj_cls_loss=round(self.best["obj_cls_loss"], 5),
            lang_acc=round(self.best["lang_acc"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
