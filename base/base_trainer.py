import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from models.loss import WBCELoss, PKDLoss, ContLoss,calculate_certainty


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, logger, gpu):
        self.config = config
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['epochs'] if cfg_trainer['save_period'] == -1 else cfg_trainer['save_period']
        # self.save_period = 1
        self.validation_period = cfg_trainer['validation_period'] if cfg_trainer['validation_period'] == -1 else cfg_trainer['validation_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.reset_best_mnt = cfg_trainer['reset_best_mnt']
        self.rank = torch.distributed.get_rank()

        if logger is None:
            self.logger = config.get_logger('trainer', cfg_trainer['verbosity'])
        else:
            self.logger = logger
            # setup visualization writer instance
            if self.rank == 0:
                self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
            else:
                self.writer = TensorboardWriter(config.log_dir, self.logger, False)
        
        if gpu is None:
            # setup GPU device if available, move model into configured device
            self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        else:
            self.device = gpu
            self.device_ids = None

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir


        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        not_improved_count = 0
        if self.config['data_loader']['args']['task']['step'] > 0:
            self.true_prototypes =self.prev_prototypes
            self.true_noise = self.prev_noise
            self.true_current_numbers = self.current_numbers
            self.prototypes_bias = self.prev_prototypes-self.prev_prototypes
        if self.config['data_loader']['args']['task']['step'] ==0:
            self.epochs = self.config['trainer']['epochs']
        else:
            self.epochs = self.config['trainer']['epochs_incre']
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, val_flag = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.rank == 0:

                if val_flag and (self.mnt_mode != 'off'):
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and float(log[self.mnt_metric]) <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and float(log[self.mnt_metric])-self.mnt_best >= 0)

                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False
                    if self.config['data_loader']['args']['task']['step'] == 0:
                        improved = True
                    if improved:
                        self.mnt_best = float(log[self.mnt_metric])
                        not_improved_count = 0

                    else:
                        not_improved_count += 1

                if (self.early_stop > 0) and (not_improved_count >= self.early_stop):
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(epoch))
                    self.writer.close()
                    break

                if epoch % self.save_period == 0:



                    # #Protopes refining
                    # if self.config['data_loader']['args']['task']['step'] > 0:
                    #     self.prev_prototypes = self.true_prototypes
                    #     self.prev_noise = self.true_noise
                    #     self.current_numbers = self.true_current_numbers
                    #     self.refine_prototypes()

                    self._save_checkpoint(self.epochs)
                    self.compute_prototypes(self.config)
                    self.compute_noise(self.config)
                    self.save_prototypes(self.config, self.epochs)
        # close TensorboardX
        self.writer.close()

    def refine_prototypes(self):
        self.logger.info("refine to mitigate bias of model on old class...")

        pred_numbers = torch.zeros(self.n_old_classes + 1).to(self.device)
        pred_numbers_new = torch.zeros(self.n_new_classes + self.n_old_classes + 1).to(self.device)
        prototypes_old = torch.zeros(self.n_old_classes, 256).to(self.device)
        prototypes_new = torch.zeros(self.n_old_classes, 256).to(self.device)
        for batch_idx, data in enumerate(self.train_loader):
            with torch.no_grad():
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                logit_old, features_old, _ = self.model_old(data['image'], ret_intermediate=True)
                logit_new, features_new, _ = self.model(data['image'], ret_intermediate=True)

                logit_old = logit_old.detach()
                normalized_features_old = F.normalize(features_old[-1], p=2, dim=1)
                normalized_features_new = F.normalize(features_new[-1], p=2, dim=1)

                pred_old = logit_old.argmax(dim=1) + 1  # pred: [N. H, W]
                idx_old = (logit_old > 0.5).float()  # logit: [N, C, H, W]
                idx_old = idx_old.sum(dim=1)  # logit: [N, H, W]
                pred_old[idx_old == 0] = 0  # set background (non-target class)

                pred_new = logit_new.argmax(dim=1) + 1  # pred: [N. H, W]
                idx_new = (logit_new > 0.5).float()  # logit: [N, C, H, W]
                idx_new = idx_new.sum(dim=1)  # logit: [N, H, W]
                pred_new[idx_new == 0] = 0  # set background (non-target class)
                if logit_old.shape[1]<=1:
                    uncer_old = torch.sigmoid(logit_old).squeeze(1)[:, 8::16, 8::16]
                else:
                    uncer_old = calculate_certainty(torch.sigmoid(logit_old)).squeeze(1)[:, 8::16, 8::16]
                uncer_new = calculate_certainty(torch.sigmoid(logit_new)).squeeze(1)[:, 8::16, 8::16]
                pred_region_old = (pred_old * (data['label'] == 0))[:, 8::16, 8::16]
                pred_region_new = (pred_new * (data['label'] == 0))[:, 8::16, 8::16]

                real_bg_region_old = torch.logical_and(pred_old == 0, data['label'] == 0)[:, 8::16, 8::16]
                real_bg_region_new = torch.logical_and(pred_new == 0, data['label'] == 0)[:, 8::16, 8::16]
                target_old_n_new = label_to_one_old_n_new(data['label'], logit_old[:, :, 8::16, 8::16], pred_region_old,
                                                          pred_region_new, uncer_old, uncer_new)
                class_region_old = target_old_n_new.unsqueeze(2) * normalized_features_old.unsqueeze(1)
                class_region_new = target_old_n_new.unsqueeze(2) * normalized_features_new.unsqueeze(1)

                prototypes_old = prototypes_old + class_region_old.sum(dim=[0, 3, 4])
                prototypes_new = prototypes_new + class_region_new.sum(dim=[0, 3, 4])

                pred_numbers[0] = pred_numbers[0] + real_bg_region_old.sum()
                pred_numbers_new[0] = pred_numbers_new[0] + real_bg_region_new.sum()
                for cls in pred_region_old.unique():
                    if cls in [0, 255]:
                        continue
                    pred_numbers[cls] = pred_numbers[cls] + (
                                (pred_region_old == cls) * (pred_region_new == cls) * (uncer_old > 0.7) * (
                                    uncer_new > 0.7)).sum()
                    pred_numbers_new[cls] = pred_numbers_new[cls] + (pred_region_old == cls).sum()

            self.progress(self.logger, batch_idx, len(self.train_loader))
        self.current_numbers = torch.cat([self.current_numbers,self.numbers[-self.n_new_classes:]],dim=0)
        ratio = pred_numbers[1:]/self.current_numbers[1:self.n_old_classes+1]

        prototypes_old = F.normalize(prototypes_old, p=2, dim=1)
        prototypes_new = F.normalize(prototypes_new, p=2, dim=1)
        for i in range(self.n_old_classes):
            if prototypes_old[i].sum()==0 or prototypes_new[i].sum()==0:
                prototypes_old[i] = self.prev_prototypes[i]
                prototypes_new[i] = self.prev_prototypes[i]
                continue
        prototypes_bias = prototypes_new-prototypes_old
        current_prototypes = self.prev_prototypes+prototypes_bias
        self.prev_prototypes = (1-ratio.unsqueeze(1))*self.prev_prototypes + ratio.unsqueeze(1)* current_prototypes
        self.prev_noise = self.true_noise + torch.abs(self.prev_prototypes-self.true_prototypes)

    def save_prototypes(self, config, epoch):
        save_file = str(config.save_dir) + "/prototypes-epoch{}.pth".format(epoch)
        if config['data_loader']['args']['task']['step'] == 0:
            all_info = {
                "numbers": self.numbers,
                "prototypes": self.prototypes,
                "current_numbers": self.numbers,
                "norm_mean_and_std": self.norm_mean_and_std,
                "noise": self.noise
            }
        else:
            all_info = {
                "numbers": self.numbers,
                "prototypes": self.prototypes,
                "current_numbers": self.current_numbers,
                "norm_mean_and_std": self.norm_mean_and_std,
                "noise": self.noise
            }

        torch.save(all_info, save_file)

    def compute_cls_number(self, config):
        self.logger.info("computing number of pixels...")

        number_save_file = str(config.save_dir) + "/numbers_tmp.pth"
        if os.path.exists(number_save_file):
            self.numbers = torch.load(number_save_file,map_location='cpu')
            return

        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes
        numbers = torch.zeros(n_new_classes + 1).to(self.device)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):


                small_label = data['label'][:, 8::16, 8::16].to(self.device)
                for i in range(n_new_classes + 1):
                    if i == 0:
                        numbers[i] = numbers[i] + torch.sum(small_label == 0).item()
                        continue
                    numbers[i] = numbers[i] + torch.sum(small_label == i + n_old_classes).item()
                self.progress(self.logger, batch_idx, len(self.train_loader))
        self.numbers = numbers

        torch.save(numbers, number_save_file)

    def compute_prototypes(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes
        prototypes = torch.zeros(n_new_classes, 256, device='cuda')
        norms = {k: [] for k in range(n_new_classes)}
        norm_mean_and_std = torch.zeros(2, n_new_classes, device='cuda')
        self.logger.info("computing prototypes...")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)
                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]

                small_label = data['label'][:, 8::16, 8::16]
                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)
                prototypes = prototypes + class_region.sum(dim=[0, 3, 4])
                norm = torch.norm(features[-1], p=2, dim=1)

                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    norms[int(cls) - n_old_classes - 1].append(norm[small_label == cls])
                self.progress(self.logger, batch_idx, len(self.train_loader))
            prototypes = F.normalize(prototypes, p=2, dim=1)
            if config['data_loader']['args']['task']['step'] == 0:
                self.prototypes = prototypes
            else:
                 self.prototypes = torch.cat([self.prev_prototypes, prototypes], dim=0)
            for k in range(n_new_classes):
                norms[k] = torch.cat(norms[k], dim=0)
                norm_mean_and_std[0, k] = norms[k].mean()
                norm_mean_and_std[1, k] = norms[k].std()

            if config['data_loader']['args']['task']['step'] == 0:
                self.norm_mean_and_std = norm_mean_and_std
            else:
                self.norm_mean_and_std = torch.cat([self.prev_norm, norm_mean_and_std], dim=1)

    def compute_noise(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes
        prototypes = self.prototypes

        noise = torch.zeros(n_new_classes, 256, device='cuda')
        noise_cnt = torch.zeros(n_new_classes, device='cuda')

        self.logger.info("computing noise...")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)
                pred = torch.argmax(logit,dim=1) + 1  # pred: [N. H, W]
                idx = (logit > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0
                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]


                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)

                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    dist = \
                        class_region[:, int(cls) - n_old_classes - 1].permute(1, 0, 2, 3)[:,
                        (small_label == cls)] - \
                        prototypes[int(cls) - n_old_classes - 1].unsqueeze(1)
                    dist = dist ** 2
                    noise[int(cls) - n_old_classes - 1] = \
                        noise[int(cls) - n_old_classes - 1] + dist.sum(dim=1)
                    noise_cnt[int(cls) - n_old_classes - 1] = \
                        noise_cnt[int(cls) - n_old_classes - 1] + (small_label == cls).sum()
                self.progress(self.logger, batch_idx, len(self.train_loader))

            noise = torch.sqrt(noise / noise_cnt.unsqueeze(1))

            if config['data_loader']['args']['task']['step'] == 0:
                self.noise = noise
            else:
                self.noise = torch.cat([self.prev_noise, noise], dim=0)
    def test(self):
        result = self._test()
        
        if self.rank == 0:
            log = {}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

    def progress(self, logger, i, total_length):
        period = total_length // 5
        if period == 0:
            return
        elif (i % period == 0):
            logger.info(f'[{i}/{total_length}]')

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best_model(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        best_path = str(self.checkpoint_dir / 'checkpoint-epoch60.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, test=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        if not self.reset_best_mnt:
            self.mnt_best = checkpoint['monitor_best']

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'])
            # self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        if test is False:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

def label_to_one_hot(label, logit, n_old_classes, ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    pred = logit.argmax(dim=1) + 1  # pred: [N. H, W]
    idx = (logit > 0.5).float()  # logit: [N, C, H, W]
    idx = idx.sum(dim=1)  # logit: [N, H, W]
    pred[idx == 0] = 0
    for cls_idx in label.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - (n_old_classes + 1)] = ((label == int(cls_idx))).float()
    return target

def label_to_one_old_n_new(label, logit, pred_old,pred_new, uncer_old, uncer_new,ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    for cls_idx in pred_old.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - 1] = ((pred_old == int(cls_idx)) * (pred_new == int(cls_idx)) * (uncer_old > 0.7) * (
                    uncer_new > 0.7)).float()
    return target
