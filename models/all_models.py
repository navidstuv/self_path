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
from torch.optim.lr_scheduler import _LRScheduler

# custom modules
from schedulers import get_scheduler

from optimizers import get_optimizer
from models.model import get_model
from utils.metrics import AverageMeter, accuracy
from utils.utils import to_device, make_inf_dl

# summary
from tensorboardX import SummaryWriter

class LinearRampdown(_LRScheduler):
    def __init__(self, opt, rampdown_from=1000, rampdown_till=1200, last_epoch=-1):
        self.rampdown_from = rampdown_from
        self.rampdown_till = rampdown_till
        super(LinearRampdown, self).__init__(opt, last_epoch)

    def ramp(self, e):
        if e > self.rampdown_from:
            f = (e-self.rampdown_from)/(self.rampdown_till-self.rampdown_from)
            return 1 - f
        else:
            return 1.0

    def get_lr(self):
        factor = self.ramp(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

class AuxModel:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = SummaryWriter(config.log_dir)
        cudnn.enabled = True

        # set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model, self.g_model = get_model(config)
        self.d_model = self.d_model.to(self.device)
        self.g_model = self.g_model.to(self.device)


        if config.mode == 'train':
            # set up optimizer, lr scheduler and loss functions

            lr = config.lr
            self.best_acc = 0
            self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=lr, betas=(.5, .999))
            self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=lr, betas=(.5, .999))
            self.g_scheduler = LinearRampdown(self.g_optimizer, rampdown_from=1000, rampdown_till=1200)
            self.d_scheduler = LinearRampdown(self.d_optimizer, rampdown_from=1000, rampdown_till=1200)

            self.class_loss_func = nn.CrossEntropyLoss()
            self.softplus = nn.Softplus()

            self.start_iter = 0

            # resume
            if config.training_resume:
                self.load(config.model_dir + '/' + config.training_resume)

            cudnn.benchmark = True

        elif config.mode == 'val':
            self.load(os.path.join(config.model_dir, config.validation_model))
        else:
            self.load(os.path.join(config.model_dir, config.testing_model))

    def entropy_loss(self, x):
        return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()

    def train(self, src_loader, tar_loader, val_loader, test_loader):

        num_batches = len(src_loader)
        print_freq = max(num_batches // self.config.training_num_print_epoch, 1)
        i_iter = self.start_iter
        start_epoch = i_iter // num_batches
        num_epochs = self.config.num_epochs

        for epoch in range(start_epoch, num_epochs):
            self.d_model.train()
            self.g_model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()

            for it, (src_batch, tar_batch) in enumerate(zip(src_loader, itertools.cycle(tar_loader))):
                t = time.time()

                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                src = src_batch
                tar = tar_batch
                src = to_device(src, self.device)
                tar = to_device(tar, self.device)
                src_imgs, src_cls_lbls, src_aux_imgs, src_aux_lbls = src
                tar_imgs, tar_aux_lbls = tar

                r = torch.randperm(2*self.config.src_batch_size)
                src_tar_imgs = torch.cat((src_imgs, tar_imgs), dim=0)
                src_tar_imgs = src_tar_imgs[r,:,:,:]
                src_tar_img = src_tar_imgs[:self.config.src_batch_size,:,:,:]

                src_tar_lbls = torch.cat((torch.zeros((self.config.src_batch_size)), torch.ones((self.config.src_batch_size))), dim=0)
                src_tar_lbls = src_tar_lbls[r]
                src_tar_lbls = src_tar_lbls[:self.config.src_batch_size]
                src_tar_lbls = to_device(src_tar_lbls.long(), self.device)



                z = torch.randn(self.config.src_batch_size, self.config.gan_latent_dim)
                z = z.cuda()


                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                gen_inp = self.g_model(z)

                lab_logits = self.d_model(src_imgs, 'main_task')
                unlab_logits = self.d_model(tar_imgs, 'main_task')
                fake_logits = self.d_model(gen_inp.detach(), 'main_task')

                unlab_lse = torch.logsumexp(unlab_logits, dim=1)
                fake_lse = torch.logsumexp(fake_logits, dim=1)
                loss_lab = self.class_loss_func(lab_logits, src_cls_lbls)
                loss_unlab = -.5 * unlab_lse.mean() + .5 * self.softplus(unlab_lse).mean() + .5 * self.softplus(fake_lse).mean()
                loss_disc = loss_unlab + loss_lab

                tar_aux_loss = {}
                src_aux_loss = {}

                src_tar_logits = self.d_model(src_tar_img, 'domain_classifier')
                tar_aux_logits = self.d_model(tar_imgs, 'magnification')
                src_aux_logits = self.d_model(src_aux_imgs, 'magnification')
                tar_aux_loss['magnification'] = self.class_loss_func(tar_aux_logits, tar_aux_lbls)
                src_aux_loss['magnification'] = self.class_loss_func(src_aux_logits, src_aux_lbls)
                tar_aux_loss['domain_classifier'] = self.class_loss_func(src_tar_logits, src_tar_lbls)

                loss_disc += src_aux_loss['magnification'] * self.config.loss_weight['magnification'] # todo: magnification weight
                loss_disc += tar_aux_loss['magnification'] * self.config.loss_weight['magnification'] # todo: main task weight
                loss_disc += tar_aux_loss['domain_classifier'] * self.config.loss_weight['domain_classifier']

                loss_disc.backward()
                self.d_optimizer.step()

                #Train Generator
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                __, fake_layer = self.d_model(gen_inp, 'main_task', req_inter_layer=True)
                __, real_layer = self.d_model(tar_imgs, 'main_task', req_inter_layer=True)
                m1 = fake_layer.mean(dim=0)
                # m2 = real_layer.detach().mean(dim=0)
                m2 = real_layer.mean(dim=0)
                loss_gen = torch.abs(m1 - m2).mean()
                loss_gen.backward()
                self.g_optimizer.step()






                precision1_train, precision2_train = accuracy(lab_logits, src_cls_lbls, topk=(1, 2))
                top1.update(precision1_train[0], src_imgs.size(0))


                losses.update(loss_disc.item(), src_imgs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - t)

                i_iter += 1

                # if i_iter % print_freq == 0:
                print= ''
                for task_name in self.config.aux_task_names:
                    if task_name=='domain_classifier':
                        print = print +' tar_aux_' + task_name +': {:.3f}'
                    else:
                        print = print + 'src_aux_' + task_name + ': {:.3f} | tar_aux_' + task_name + ': {:.3f}'
                print_string = 'Epoch {:>2} | iter {:>4} | loss:{:.3f} |  acc: {:.3f}| src_main: {:.3f} |loss_g:{:.3f}|' + print +  '|{:4.2f} s/it'

                src_aux_loss_all = [loss.item() for loss in src_aux_loss.values()]
                tar_aux_loss_all = [loss.item() for loss in tar_aux_loss.values()]
                self.logger.info(print_string.format(epoch, i_iter,
                    losses.avg,
                    top1.avg,
                    loss_lab.item(),
                    loss_gen.item(),
                    *src_aux_loss_all,
                    *tar_aux_loss_all,
                    batch_time.avg))
                self.writer.add_scalar('losses/all_loss', losses.avg, i_iter)
                self.writer.add_scalar('losses/src_main_loss', loss_lab, i_iter)
                for task_name in self.config.aux_task_names:
                    if task_name=='domain_classifier':
                        # self.writer.add_scalar('losses/src_aux_loss_'+task_name, src_aux_loss[task_name], i_iter)
                        self.writer.add_scalar('losses/tar_aux_loss_'+task_name, tar_aux_loss[task_name], i_iter)
                    else:
                        self.writer.add_scalar('losses/src_aux_loss_'+task_name, src_aux_loss[task_name], i_iter)
                        self.writer.add_scalar('losses/tar_aux_loss_'+task_name, tar_aux_loss[task_name], i_iter)

            # del loss, src_class_loss, src_aux_loss, tar_aux_loss, tar_entropy_loss
            # del src_aux_logits, src_class_logits
            # del tar_aux_logits, tar_class_logits

            # validation
            self.save(self.config.model_dir, i_iter)

            if val_loader is not None:
                self.logger.info('validating...')
                class_acc = self.test(val_loader)
                # self.writer.add_scalar('val/aux_acc', class_acc, i_iter)
                self.writer.add_scalar('val/class_acc', class_acc, i_iter)
                if class_acc > self.best_acc:
                    self.best_acc = class_acc
                    self.save(self.config.best_model_dir, i_iter)
                    # todo copy current model to best model
                self.logger.info('Best testing accuracy: {:.2f} %'.format(self.best_acc))

            if test_loader is not None:
                self.logger.info('testing...')
                class_acc = self.test(test_loader)
                # self.writer.add_scalar('test/aux_acc', class_acc, i_iter)
                self.writer.add_scalar('test/class_acc', class_acc, i_iter)
                if class_acc > self.best_acc:
                    self.best_acc = class_acc
                    # todo copy current model to best model
                self.logger.info('Best testing accuracy: {:.2f} %'.format(self.best_acc))

        self.logger.info('Best testing accuracy: {:.2f} %'.format(self.best_acc))
        self.logger.info('Finished Training.')

    def save(self, path, i_iter):
        state = {"iter": i_iter + 1,
                "d_model_state": self.d_model.state_dict(),
                 "g_model_state": self.g_model.state_dict(),
                "d_optimizer_state": self.d_optimizer.state_dict(),
                 "g_optimizer_state": self.g_optimizer.state_dict(),
                "d_scheduler_state": self.d_scheduler.state_dict(),
                 "g_scheduler_state": self.g_scheduler.state_dict(),
                 'best_acc': self.best_acc
                }
        save_path = os.path.join(path, 'model_{:06d}.pth'.format(i_iter))
        self.logger.info('Saving model to %s' % save_path)
        torch.save(state, save_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.d_model.load_state_dict(checkpoint['d_model_state'])
        self.logger.info('Loaded model from: ' + path)

        if self.config.mode == 'train':
            # self.d_model.load_state_dict(checkpoint['d_model_state'])
            self.g_model.load_state_dict(checkpoint['g_model_state'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
            self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state'])
            self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state'])
            self.start_iter = checkpoint['iter']
            self.best_acc = checkpoint['best_acc']
            self.logger.info('Start iter: %d ' % self.start_iter)


    def test(self, val_loader):
        val_loader_iterator = iter(val_loader)
        num_val_iters = len(val_loader)
        tt = tqdm(range(num_val_iters), total=num_val_iters, desc="Validating")

        aux_correct = 0
        class_correct = 0
        total = 0
        soft_labels = np.zeros((1, 2))
        true_labels = []

        self.d_model.eval()
        with torch.no_grad():
            for cur_it in tt:

                data = next(val_loader_iterator)
                data = to_device(data, self.device)
                imgs, cls_lbls, _, _ = data
                # Get the inputs

                logits = self.d_model(imgs, 'main_task')

                if self.config.save_output==True:
                    smax = nn.Softmax(dim=1)
                    smax_out = smax(logits)
                    soft_labels = np.concatenate((soft_labels, smax_out.cpu().numpy()), axis=0)
                    true_labels = np.append(true_labels, cls_lbls.cpu().numpy())

                _, cls_pred = logits.max(dim=1)
                # _, aux_pred = aux_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls)
                # aux_correct += torch.sum(aux_pred == aux_lbls.data)
                total += imgs.size(0)

            tt.close()
        if self.config.save_output==True:
            soft_labels = soft_labels[1:, :]
            np.save('pred_cam1234.npy', soft_labels)
            np.save('true_cam1234.npy', true_labels)

        # aux_acc = 100 * float(aux_correct) / total
        class_acc = 100 * float(class_correct) / total
        self.logger.info('class_acc: {:.2f} %'.format( class_acc))
        return class_acc