import os
import time
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
from utils.utils import stats

# custom modules
from schedulers import get_scheduler
from models.model import get_model
from utils.metrics import AverageMeter, accuracy
from utils.utils import to_device, make_inf_dl, save_output_img

# summary
from tensorboardX import SummaryWriter

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LinearRampdown(_LRScheduler):
    def __init__(self, opt, rampdown_from=1000, rampdown_till=1200, last_epoch=-1):
        self.rampdown_from = rampdown_from
        self.rampdown_till = rampdown_till
        super(LinearRampdown, self).__init__(opt, last_epoch)

    def ramp(self, e):
        if e > self.rampdown_from:
            f = (e - self.rampdown_from) / (self.rampdown_till - self.rampdown_from)
            return 1 - f
        else:
            return 1.0

    def get_lr(self):
        factor = self.ramp(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


class AuxModel:

    def __init__(self, config, logger, wandb):
        self.config = config
        self.logger = logger
        self.writer = SummaryWriter(config.log_dir)
        self.wandb = wandb
        cudnn.enabled = True

        # set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(config)
        if len(config.gpus) > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.best_acc = 0
        self.best_AUC = 0
        self.class_loss_func = nn.CrossEntropyLoss()
        self.pixel_loss = nn.L1Loss()
        if config.mode == 'train':
            # set up optimizer, lr scheduler and loss functions
            lr = config.lr
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.5, .999))
            # self.optimizer =torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=config.weight_decay)
            self.scheduler = LinearRampdown(self.optimizer, rampdown_from=1000, rampdown_till=1200)
            # self.scheduler = MultiStepLR(self.optimizer, milestones=[50,100,150,300], gamma=0.1)
            self.wandb.watch(self.model)
            self.start_iter = 0

            # resume
            if config.training_resume:
                self.load(config.model_dir + '/' + config.training_resume)

            cudnn.benchmark = True
        elif config.mode == 'val':
            self.load(os.path.join(config.testing_model))
        else:
            self.load(os.path.join(config.testing_model))

    def entropy_loss(self, x):
        return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()

    def train_epoch_main_task(self, src_loader, tar_loader, epoch, print_freq):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        main_loss = AverageMeter()
        top1 = AverageMeter()

        for it, src_batch in enumerate(src_loader):
            t = time.time()
            self.optimizer.zero_grad()
            src = src_batch
            src = to_device(src, self.device)
            src_imgs, src_cls_lbls, _, _, _, _,_= src

            self.optimizer.zero_grad()

            src_main_logits = self.model(src_imgs, 'main_task')
            src_main_loss = self.class_loss_func(src_main_logits, src_cls_lbls)
            loss = src_main_loss * self.config.loss_weight['main_task']
            main_loss.update(loss.item(), src_imgs.size(0))
            precision1_train, precision2_train = accuracy(src_main_logits, src_cls_lbls, topk=(1, 2))
            top1.update(precision1_train[0], src_imgs.size(0))

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), src_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - t)

            self.start_iter += 1

            if self.start_iter % print_freq == 0:
                print_string = 'Epoch {:>2} | iter {:>4} | loss:{:.3f}| acc:{:.3f}| src_main: {:.3f} |' + '|{:4.2f} s/it'
                self.logger.info(print_string.format(epoch, self.start_iter,
                                                     losses.avg,
                                                     top1.avg,
                                                     main_loss.avg,
                                                     batch_time.avg))
                self.writer.add_scalar('losses/all_loss', losses.avg, self.start_iter)
                self.writer.add_scalar('losses/src_main_loss', src_main_loss, self.start_iter)
        self.wandb.log({"Train Loss": main_loss.avg})
        self.scheduler.step()

        # del loss, src_class_loss, src_aux_loss, tar_aux_loss, tar_entropy_loss
        # del src_aux_logits, src_class_logits
        # del tar_aux_logits, tar_class_logits

    def train_epoch_all_tasks(self, src_loader, tar_loader, epoch, print_freq):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        main_loss = AverageMeter()
        top1 = AverageMeter()
        start_steps = epoch * len(src_loader['main_task'])
        total_steps = self.config.num_epochs * len(tar_loader['main_task'])

        max_num_iter = len(src_loader['main_task'])
        for it in range(max_num_iter):
            t = time.time()

            # this is based on DANN paper
            p = float(it + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            self.optimizer.zero_grad()

            src = next(iter(src_loader['main_task']))
            tar = next(iter(tar_loader['main_task']))
            src = to_device(src, self.device)
            tar = to_device(tar, self.device)
            src_imgs, src_cls_lbls = src
            tar_imgs, _ = tar


            src_main_logits = self.model(src_imgs, 'main_task')
            src_main_loss = self.class_loss_func(src_main_logits, src_cls_lbls)
            loss = src_main_loss * self.config.loss_weight['main_task']
            main_loss.update(loss.item(), src_imgs.size(0))
            tar_main_logits = self.model(tar_imgs, 'main_task')
            tar_main_loss = self.entropy_loss(tar_main_logits)
            loss += tar_main_loss
            tar_aux_loss = {}
            src_aux_loss = {}

            #TO DO: separating dataloaders and iterate over tasks
            for task in self.config.task_names:
                if self.config.tasks[task]['type'] == 'classification_adapt':
                    r = torch.randperm(src_imgs.size()[0] + tar_imgs.size()[0])
                    src_tar_imgs = torch.cat((src_imgs, tar_imgs), dim=0)
                    src_tar_imgs = src_tar_imgs[r, :, :, :]
                    src_tar_img = src_tar_imgs[:src_imgs.size()[0], :, :, :]
                    src_tar_lbls = torch.cat((torch.zeros((src_imgs.size()[0])), torch.ones((tar_imgs.size()[0]))),
                                             dim=0)
                    src_tar_lbls = src_tar_lbls[r]
                    src_tar_lbls = src_tar_lbls[:src_imgs.size()[0]]
                    src_tar_lbls = src_tar_lbls.long().cuda()
                    src_tar_logits = self.model(src_tar_img, 'domain_classifier', alpha)
                    tar_aux_loss['domain_classifier'] = self.class_loss_func(src_tar_logits, src_tar_lbls)
                    loss += tar_aux_loss['domain_classifier'] * self.config.loss_weight['domain_classifier']
                if self.config.tasks[task]['type'] == 'classification_self':
                    src = next(iter(src_loader[task]))
                    tar = next(iter(tar_loader[task]))
                    src = to_device(src, self.device)
                    tar = to_device(tar, self.device)
                    src_aux_imgs, src_aux_lbls = src
                    tar_aux_imgs, tar_aux_lbls = tar
                    tar_aux_logits = self.model(tar_aux_imgs, task)
                    src_aux_logits = self.model(src_aux_imgs, task)
                    tar_aux_loss[task] = self.class_loss_func(tar_aux_logits, tar_aux_lbls)
                    src_aux_loss[task] = self.class_loss_func(src_aux_logits, src_aux_lbls)
                    loss += src_aux_loss[task] * self.config.loss_weight[task]  # todo: magnification weight
                    loss += tar_aux_loss[task] * self.config.loss_weight[task]  # todo: main task weight
                if self.config.tasks[task]['type'] == 'pixel_self':
                    src = next(iter(src_loader[task]))
                    tar = next(iter(tar_loader[task]))
                    src = to_device(src, self.device)
                    tar = to_device(tar, self.device)
                    src_aux_imgs, src_aux_lbls = src
                    tar_aux_imgs, tar_aux_lbls = tar
                    tar_aux_mag_logits = self.model(tar_aux_imgs, task)
                    src_aux_mag_logits = self.model(src_aux_imgs, task)
                    tar_aux_loss[task] = self.pixel_loss(tar_aux_mag_logits, tar_aux_lbls)
                    src_aux_loss[task] = self.pixel_loss(src_aux_mag_logits, src_aux_lbls)
                    loss += src_aux_loss[task] * self.config.loss_weight[task]  # todo: magnification weight
                    loss += tar_aux_loss[task] * self.config.loss_weight[task]


            precision1_train, precision2_train = accuracy(src_main_logits, src_cls_lbls, topk=(1, 2))
            top1.update(precision1_train[0], src_imgs.size(0))
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), src_imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - t)
            self.start_iter += 1
            if self.start_iter % print_freq == 0:
                printt = ''
                for task_name in self.config.aux_task_names:
                    if task_name == 'domain_classifier':
                        printt = printt + ' | tar_aux_' + task_name + ': {:.3f} |'
                    else:
                        printt = printt + 'src_aux_' + task_name + ': {:.3f} | tar_aux_' + task_name + ': {:.3f}'
                print_string = 'Epoch {:>2} | iter {:>4} | loss:{:.3f} |  acc: {:.3f} | src_main: {:.3f} |' + printt + '{:4.2f} s/it'
                src_aux_loss_all = [loss.item() for loss in src_aux_loss.values()]
                tar_aux_loss_all = [loss.item() for loss in tar_aux_loss.values()]
                self.logger.info(print_string.format(epoch, self.start_iter,
                                                     losses.avg,
                                                     top1.avg,
                                                     main_loss.avg,
                                                     *src_aux_loss_all,
                                                     *tar_aux_loss_all,
                                                     batch_time.avg))
                self.writer.add_scalar('losses/all_loss', losses.avg, self.start_iter)
                self.writer.add_scalar('losses/src_main_loss', src_main_loss, self.start_iter)
                for task_name in self.config.aux_task_names:
                    if task_name == 'domain_classifier':
                        # self.writer.add_scalar('losses/src_aux_loss_'+task_name, src_aux_loss[task_name], i_iter)
                        self.writer.add_scalar('losses/tar_aux_loss_' + task_name, tar_aux_loss[task_name],
                                               self.start_iter)
                    else:
                        self.writer.add_scalar('losses/src_aux_loss_' + task_name, src_aux_loss[task_name],
                                               self.start_iter)
                        self.writer.add_scalar('losses/tar_aux_loss_' + task_name, tar_aux_loss[task_name],
                                               self.start_iter)
        self.wandb.log({"Train Loss": main_loss.avg})
        self.scheduler.step()

        # del loss, src_class_loss, src_aux_loss, tar_aux_loss, tar_entropy_loss
        # del src_aux_logits, src_class_logits
        # del tar_aux_logits, tar_class_logits

    def train(self, src_loader, tar_loader, val_loader, test_loader):
        num_batches = len(src_loader)
        print_freq = max(num_batches // self.config.training_num_print_epoch, 1)
        start_epoch = self.start_iter // num_batches
        num_epochs = self.config.num_epochs
        for epoch in range(start_epoch, num_epochs):
            if len(self.config.task_names) == 1:
                self.train_epoch_main_task(src_loader, tar_loader, epoch, print_freq)
            else:
                self.train_epoch_all_tasks(src_loader, tar_loader, epoch, print_freq)
            self.logger.info('learning rate: %f ' % get_lr(self.optimizer))
            # validation
            self.save(self.config.model_dir, 'last')

            if val_loader is not None:
                self.logger.info('validating...')
                class_acc, AUC = self.test(val_loader)
                # self.writer.add_scalar('val/aux_acc', class_acc, i_iter)
                self.writer.add_scalar('val/class_acc', class_acc, self.start_iter)
                if class_acc > self.best_acc:
                    self.best_acc = class_acc
                    self.save(self.config.best_model_dir, 'best_acc')
                if AUC > self.best_AUC:
                    self.best_AUC = AUC
                    self.save(self.config.best_model_dir, 'best_AUC')
                    # todo copy current model to best model
                self.logger.info('Best validation accuracy: {:.2f} %'.format(self.best_acc))

            if test_loader is not None:
                self.logger.info('testing...')
                class_acc = self.test(test_loader)
                # self.writer.add_scalar('test/aux_acc', class_acc, i_iter)
                self.writer.add_scalar('test/class_acc', class_acc, self.start_iter)
                # if class_acc > self.best_acc:
                #     self.best_acc = class_acc
                # todo copy current model to best model
                self.logger.info('Best testing accuracy: {:.2f} %'.format(class_acc))

        self.logger.info('Best validation accuracy: {:.2f} %'.format(self.best_acc))
        self.logger.info('Finished Training.')

    def save(self, path, ext):
        state = {"iter": self.start_iter + 1,
                 "model_state": self.model.state_dict(),
                 "optimizer_state": self.optimizer.state_dict(),
                 "scheduler_state": self.scheduler.state_dict(),
                 "best_acc": self.best_acc,
                 }
        save_path = os.path.join(path, f'model_{ext}.pth')
        self.logger.info('Saving model to %s' % save_path)
        torch.save(state, save_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.logger.info('Loaded model from: ' + path)

        if self.config.mode == 'train':
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_iter = checkpoint['iter']
            self.best_acc = checkpoint['best_acc']
            self.logger.info('Start iter: %d ' % self.start_iter)

    def test(self, val_loader):
        val_loader_iterator = iter(val_loader)
        num_val_iters = len(val_loader)
        tt = tqdm(range(num_val_iters), total=num_val_iters, desc="Validating")
        loss = AverageMeter()
        kk = 1
        aux_correct = 0
        class_correct = 0
        total = 0
        soft_labels = np.zeros((1, 2))
        true_labels = []
        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:
                data = next(val_loader_iterator)
                data = to_device(data, self.device)
                imgs, cls_lbls, _, _, _, _,_ = data
                # Get the inputs
                logits = self.model(imgs, 'main_task')
                test_loss = self.class_loss_func(logits, cls_lbls)
                loss.update(test_loss.item(), imgs.size(0))
                if self.config.save_output == True:
                    smax = nn.Softmax(dim=1)
                    smax_out = smax(logits)
                    soft_labels = np.concatenate((soft_labels, smax_out.cpu().numpy()), axis=0)
                    true_labels = np.append(true_labels, cls_lbls.cpu().numpy())
                    pred_trh = smax_out.cpu().numpy()[:, 1]
                    pred_trh[pred_trh >= 0.5] = 1
                    pred_trh[pred_trh < 0.5] = 0
                    compare = cls_lbls.cpu().numpy() - pred_trh
                    FP_idx = np.where(compare == -1)
                    FN_idx = np.where(compare == 1)
                    # FP_imgs = imgs.cpu().numpy()[FP_idx, ...]
                    # FN_imgs = imgs.cpu().numpy()[FN_idx, ...]
                    # save_output_img(FP_imgs[0, ...], 'FP_images', 'FP', kk * imgs.shape[0])
                    # save_output_img(FN_imgs[0, ...], 'FN_images', 'FN', kk * imgs.shape[0])
                    kk += 1

                _, cls_pred = logits.max(dim=1)
                # _, aux_pred = aux_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls)
                # aux_correct += torch.sum(aux_pred == aux_lbls.data)
                total += imgs.size(0)

            tt.close()
        self.wandb.log({
            "Test Loss": loss.avg})
        # if self.config.save_output == True:
        soft_labels = soft_labels[1:, :]
            # np.save('pred_' + self.config.mode + '_main3.npy', soft_labels)
            # np.save('true_' + self.config.mode + '_main3.npy', true_labels)
        AUC = stats(soft_labels, true_labels, opt_thresh=0.5)

        # aux_acc = 100 * float(aux_correct) / total
        class_acc = 100 * float(class_correct) / total
        self.logger.info('class_acc: {:.2f} %'.format(class_acc))
        self.wandb.log({
            "Test acc": class_acc,
            "Test AUC": 100*AUC})
        return class_acc, AUC
