# import
import argparse
import os
import glob
import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import scipy.io as scio
import shutil
from loguru import logger
# inner import
from models import *
from utils.dataset_tfrecord import get_dataset, get_dataset_multicoil
from utils.tools import mse, calc_SNR, calc_PSNR, ifft2c_mri, mriF, rsos, loss_SLR_ISTA
from utils.mask_generator import generate_mask, load_mask


class Solver(object):
    def __init__(self, args):
        self.datadir = args.datadir
        self.dataset = args.dataset
        
        self.start_epoch = args.start_epoch
        self.end_epoch = args.end_epoch
        
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.masktype = args.masktype
        self.acc = args.acc
        self.ModelName = args.ModelName
        self.weight = args.weight
        
        self.debug = args.debug
        self.multicoil = args.multicoil
        
        # specify network
        self.net = eval(self.ModelName+'()')
        # if self.ModelName == 'LplusS_Net':
        #     self.net = LplusS_Net()
        # elif self.ModelName == 'SLR_Net':
        #     self.net = SLR_Net()
        # elif self.ModelName == 'ISTA_Net_plus':
        #     self.net = ISTA_Net_plus()
        # elif self.ModelName == 'DCCNN':
        #     self.net = DCCNN()
        self.param_num = 0 # initialize it 0, later calc it
        
        self.archive()
        
        
    def train(self):
        # prepare dataset
        if self.multicoil:
            dataset_train = get_dataset_multicoil('train', self.datadir, self.batch_size, shuffle=True)
            dataset_val = get_dataset_multicoil('val', self.datadir, 1, shuffle=False)
        else:
            dataset_train = get_dataset('train', self.datadir, self.batch_size, shuffle=True)
            dataset_val = get_dataset('val', self.datadir, 1, shuffle=False)
        logger.info('dataset loaded.')

        # load pre-weight
        if self.weight is not None:
            logger.info('load weights.')
            self.net.load_weights(self.weight)
        logger.info('network initialized.')

        # define lr and optimizer
        learning_rate = self.learning_rate
        learning_rate_decay = 0.95
        learning_rate = learning_rate * learning_rate_decay ** (self.start_epoch - 1)
        optimizer = tf.optimizers.Adam(learning_rate)

        # Iterate over epochs.
        total_step = 0
        loss = 0
        
        # calculate the base validate PSNR
        # start epoch equals to 1 means no pre-trained weights, so we calc psnr using the original undersampled data
        # else we calc psnr using the reconstructed data from net using the pre-trained weights
        if self.weight is not None:
            self.net.load_weights(os.path.split(self.weight)[0]+'/weight-best')
            val_psnr_best, _ = self.val(dataset_val, is_first=(self.start_epoch==1))
            self.net.load_weights(self.weight)
        else:
            val_psnr_best, _ = self.val(dataset_val, is_first=(self.start_epoch==1))
        logger.info(20*'*')
        logger.info('the best val psnr is /%.3f/' % val_psnr_best)
        logger.info(20*'*')

        for epoch in range(self.start_epoch, self.end_epoch+1):
            for step, sample in enumerate(dataset_train):
                # forward
                t0 = time.time()
                k0 = None
                with tf.GradientTape() as tape:
                    if self.multicoil:
                        k0, csm = sample
                        nb, nc, nt, nx, ny = k0.get_shape()
                        k0 = k0 / tf.cast(tf.math.reduce_max(tf.abs(k0)),tf.complex64)
                        label = ifft2c_mri(k0)
                        label = rsos(label)
                        F = mriF(csm)
                        # for multicoil, usually using the VISTA (fixed). of course, random is also ok
                        if self.masktype == 'vista':
                            mask = load_mask('train', self.masktype, self.acc, self.datadir)
                        else:
                            mask = generate_mask([nx, ny, nt], self.acc, self.masktype)
                    else:
                        k0, label = sample
                        nb, nt, nx, ny = k0.get_shape()
                        F = mriF()
                        # generate under-sampling mask (random)
                        if self.masktype == 'vista':
                            mask = load_mask('train', self.masktype, self.acc, self.datadir)
                        else:
                            mask = generate_mask([nx, ny, nt], self.acc, self.masktype)

                    mask = np.transpose(mask, (2, 0, 1))
                    mask = tf.constant(np.complex64(mask + 0j))

                    # generate the undersampled data k0
                    k0 = k0 * mask

                    # feed the data
                    recon, loss = self.run_infer(k0, F, mask, label)
                    psnr = calc_PSNR(recon, label)
                    
                # sum all the losses and avarage them to write the summary when epoch ends
                psnr_epoch = psnr_epoch + psnr if step != 0 else psnr
                loss_epoch = loss_epoch + loss.numpy() if step != 0 else loss.numpy()

                # backward
                grads = tape.gradient(loss, self.net.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.net.trainable_weights)) 
                
                if self.param_num == 0:
                    self.param_num = np.sum([np.prod(v.get_shape()) for v in self.net.trainable_variables])
                    logger.info('params: %d' % self.param_num)

                # log output
                if step % 100 == 0: 
                    logger.info('Epoch %d/%d, Step %d, Loss=%.3e, PSNR=%.2f, time=%.2f, lr=%.4e' % (epoch, self.end_epoch, step, loss.numpy(), psnr, (time.time() - t0), learning_rate))
                total_step += 1
                
            # At the end of epoch, print one message
            logger.info('Epoch %d/%d, Step %d, Loss=%.3e, PSNR=%.2f, time=%.2f, lr=%.4e' % (epoch, self.end_epoch, step, loss.numpy(), psnr, (time.time() - t0), learning_rate))
            
            # record loss
            with self.train_writer.as_default():
                tf.summary.scalar('loss/loss', loss_epoch/(step+1), step=epoch)
                tf.summary.scalar('PSNR', psnr_epoch/(step+1), step=epoch)

            # learning rate decay for each epoch
            learning_rate = learning_rate * learning_rate_decay
            optimizer = tf.optimizers.Adam(learning_rate)
            
            # validate
            val_psnr, val_loss = self.val(dataset_val)
            with self.val_writer.as_default():
                tf.summary.scalar('loss/loss', val_loss, step=epoch)
                tf.summary.scalar('PSNR', val_psnr, step=epoch)
                
            # save model
            # if validate PSNR is better than the best PSNR, save the best model
            if val_psnr > val_psnr_best:
                self.net.save_weights(self.weightdir+'/weight-best')
                logger.info(20*'*')
                logger.info('epoch %d, best PSNR = %.2f' % (epoch,val_psnr))
                logger.info(20*'*')
                val_psnr_best = val_psnr
            # save the latest epoch weights for continued training
            self.net.save_weights(self.weightdir+'/weight-latest')
            # every 10 epoches, we save the weights
            if epoch % 10 == 0 or epoch == 1:
                self.net.save_weights(self.weightdir+'/weight-'+str(epoch)) 
        
        self.test(training=True)

    
    
    def val(self, dataset_val, is_first = False):
        masks = load_mask('val', self.masktype, self.acc, self.datadir)
        for step, sample in enumerate(dataset_val):
            if self.multicoil:
                k0, csm = sample
                nb, nc, nt, nx, ny = k0.get_shape()
                k0 = k0 / tf.cast(tf.math.reduce_max(tf.abs(k0)),tf.complex64)
                label = ifft2c_mri(k0)
                label = rsos(label)
                F = mriF(csm)
            else:
                k0, label = sample
                nb, nt, nx, ny = k0.get_shape()
                F = mriF()

            # generate under-sampling mask (fix for val)S
            mask = masks[masks.files[step]]
            mask = tf.constant(np.complex64(mask + 0j))

            # generate the undersampled data k0
            k0 = k0 * mask

            # feed the data
            if is_first:
                recon = F.TH(k0)
                recon = tf.abs(recon) if self.multicoil else recon
                loss_all = 0
            else:
                recon, loss = self.run_infer(k0, F, mask, label)
                loss_all = loss_all + loss.numpy() if step != 0 else loss.numpy()
                
            psnr_all = psnr_all + calc_PSNR(recon, label) if step != 0 else calc_PSNR(recon, label)

        val_loss = loss_all/(step+1)
        val_psnr = psnr_all/(step+1)
            
        return val_psnr, val_loss


    def test(self, training=False, autoload=False):
        if self.multicoil:
            dataset_test = get_dataset_multicoil('test', self.datadir, 1, shuffle=False)
        else:
            dataset_test = get_dataset('test', self.datadir, 1, shuffle=False)
        if training:
            self.net.load_weights(self.weightdir+'/weight-best')
        else:
            if self.weight is not None:
                logger.info('loading weights...')
                self.net.load_weights(self.weight)
        if autoload:
            fn = '-'.join([self.ModelName, self.masktype, str(self.acc)])
            print(fn)
            logger.debug(str(*glob.glob('./weights-'+self.dataset+'/' + fn + '*')) + '/weight-best')
            self.net.load_weights(str(*glob.glob('./weights-'+self.dataset+'/' + fn + '*')) + '/weight-best')
        logger.info('net initialized, testing...')
        SNRs = []
        PSNRs = []
        MSEs = []
        SSIMs = []
        masks = load_mask('test', self.masktype, self.acc, self.datadir)
        for step, sample in enumerate(dataset_test):
            if self.multicoil:
                k0, csm = sample
                nb, nc, nt, nx, ny = k0.get_shape()
                k0 = k0 / tf.cast(tf.math.reduce_max(tf.abs(k0)),tf.complex64)
                label = ifft2c_mri(k0)
                label = rsos(label)
                F = mriF(csm)
            else:
                k0, label = sample
                nb, nt, nx, ny = k0.get_shape()
                F = mriF()

            # generate under-sampling mask (fix for test)
            mask = masks[masks.files[step]]
            mask = tf.constant(np.complex64(mask + 0j))

            # generate the undersampled data k0
            k0 = k0 * mask
            
            # feed the data
            t0 = time.time()
            recon, loss = self.run_infer(k0, F, mask, label)
            t = time.time() - t0
            
            # if step == 8:
            #     scio.savemat(self.ModelName+'.mat', {'recon': recon.numpy()})
            #     scio.savemat('us.mat', {'us': ifft2c_mri(k0).numpy()})
            
            # calc the metrics
            SNR_ = calc_SNR(recon, label)
            PSNR_ = calc_PSNR(recon, label)
            MSE_ = mse(recon, label)
            SSIM_ = tf.image.ssim(tf.transpose(tf.abs(recon), [0, 2, 3, 1]), tf.transpose(tf.abs(label), [0, 2, 3, 1]), max_val=1.0)
            SNRs.append(SNR_)
            PSNRs.append(PSNR_)
            MSEs.append(MSE_)
            SSIMs.append(SSIM_)
            logger.info('data %d --> SER = \%.3f\, PSNR = \%.3f\, SSIM = \%.3f\, MSE = {%.3e}, t = %.2f' % (step, SNR_, PSNR_, SSIM_, MSE_, t))
            
        SNRs = np.array(SNRs)
        PSNRs = np.array(PSNRs)
        MSEs = np.array(MSEs)
        logger.info('SER = %.3f(%.3f), PSNR = %.3f(%.3f), SSIM = %.3f(%.3f), MSE = %.3e(%.3e)' % (np.mean(SNRs), np.std(SNRs), np.mean(PSNRs), np.std(PSNRs), np.mean(SSIMs), np.std(SSIMs), np.mean(MSEs), np.std(MSEs)))


    def test_spec(self, num, autoload=False):
        if self.multicoil:
            dataset_test = get_dataset_multicoil('test', self.datadir, 1, shuffle=False)
        else:
            dataset_test = get_dataset('test', self.datadir, 1, shuffle=False)
        if self.weight is not None:
            logger.info('loading weights...')
            self.net.load_weights(self.weight)
        if autoload:
            fn = '-'.join([self.ModelName, self.masktype, str(self.acc)])
            logger.debug(str(*glob.glob('./weights-'+self.dataset+'/' + fn + '*')) + '/weight-best')
            self.net.load_weights(str(*glob.glob('./weights-'+self.dataset+'/' + fn + '*')) + '/weight-best')
        logger.info('net initialized, testing...')
        masks = load_mask('test', self.masktype, self.acc, self.datadir)
        
        sample = tuple(dataset_test.skip(num).take(1))[0]
        
        if self.multicoil:
            k0, csm = sample
            nb, nc, nt, nx, ny = k0.get_shape()
            k0 = k0 / tf.cast(tf.math.reduce_max(tf.abs(k0)),tf.complex64)
            label = ifft2c_mri(k0)
            label = rsos(label)
            F = mriF(csm)
        else:
            k0, label = sample
            nb, nt, nx, ny = k0.get_shape()
            F = mriF()
        # generate under-sampling mask (fix for test)
        mask = masks[masks.files[num]]
        mask = tf.constant(np.complex64(mask + 0j))

        # generate the undersampled data k0
        k0 = k0 * mask
        
        # feed the data
        t0 = time.time()
        recon, loss = self.run_infer(k0, F, mask, label)
        t = time.time() - t0
        
        fn = '-'.join([self.ModelName, self.masktype, str(self.acc)])
        scio.savemat('./out/'+'-'.join([self.ModelName, self.masktype, str(self.acc)])+'-%d.mat'%num, {'recon': recon.numpy()})
        us = tf.abs(F.TH(k0)) if self.multicoil else F.TH(k0)
        scio.savemat('./out/'+'-'.join(['us', self.masktype, str(self.acc)])+'-%d.mat'%num, {'us': us.numpy()})
        scio.savemat('./out/'+'-'.join(['label', self.masktype, str(self.acc)])+'-%d.mat'%num, {'label': label.numpy()})
        
        # calc the metrics
        SNR_ = calc_SNR(recon, label)
        PSNR_ = calc_PSNR(recon, label)
        MSE_ = mse(recon, label)
        SSIM_ = tf.image.ssim(tf.transpose(tf.abs(recon), [0, 2, 3, 1]), tf.transpose(tf.abs(label), [0, 2, 3, 1]), max_val=1.0)
        logger.info('data %d --> SER = \%.3f\, PSNR = \%.3f\, SSIM = \%.3f\, MSE = {%.3e}, t = %.2f' % (num, SNR_, PSNR_, SSIM_, MSE_, t))
      
            
    def run_infer(self, k0, F, mask, label):
        if self.ModelName == 'JotlasNet':
            recon, X_SYM = self.net(k0, F, mask)
            recon = tf.abs(recon) if self.multicoil else recon
            loss = mse(recon, label)
        elif self.ModelName == 'T2LR_Net':
            recon, X_SYM = self.net(k0, F, mask)
            recon = tf.abs(recon) if self.multicoil else recon
            loss = mse(recon, label)
        elif self.ModelName == 'LplusS_Net':
            L_recon, S_recon, LSrecon = self.net(k0, F, mask)
            recon = L_recon + S_recon
            recon = tf.abs(recon) if self.multicoil else recon
            loss = mse(recon, label)
        elif self.ModelName == 'SLR_Net':
            recon, X_SYM = self.net(k0, F, mask)
            recon = tf.abs(recon) if self.multicoil else recon
            loss = loss_SLR_ISTA(recon, label, X_SYM)
        elif self.ModelName == 'ISTA_Net_plus':
            recon, X_SYM = self.net(k0, F, mask)
            recon = tf.abs(recon) if self.multicoil else recon
            loss = loss_SLR_ISTA(recon, label, X_SYM)
        elif self.ModelName == 'DCCNN':
            recon = self.net(k0, F, mask)
            recon = tf.abs(recon) if self.multicoil else recon
            loss = mse(recon, label)  
        return recon, loss
    

    def archive(self):
        if not self.debug:
            # give the log dir and the model dir
            name_seq = [str(self.ModelName), str(self.masktype), str(self.acc), str(self.channels), str(self.factor), str(self.batch_size), str(self.learning_rate)]
            model_id = '-'.join([name_seq[i] for i in [0,1,2,3]]) # it can be chosen flexiably
            TIMESTAMP = "{0:%Y%m%dT%H%M%S}".format(datetime.now())
            
            os.makedirs('./archive') if not os.path.exists('./archive') else None
            target =  './archive/' + model_id + '-' + TIMESTAMP
            os.makedirs(target) if not os.path.exists(target) else None
            
            # log
            train_logdir = target+'/logs/train'
            val_logdir = target+'/logs/val'
            self.train_writer = tf.summary.create_file_writer(train_logdir)
            self.val_writer = tf.summary.create_file_writer(val_logdir)
            # print logger
            logger.remove(handler_id=None) # 清除之前的设置
            logger.add(sink=target+'/log.log', level='INFO') 
                
            # model
            self.weightdir = target+'/weights/'
            os.makedirs(self.weightdir) if not os.path.exists(self.weightdir) else None

            # adding exception handling
            try:
                shutil.copy('./main.py', target)
                shutil.copy('./Solver.py', target)
                shutil.copytree('./models', target+'/models')
            except IOError as e:
                logger.error("Unable to copy file. %s" % e)
            except:
                logger.error("Unexpected error:", sys.exc_info())
        elif self.debug:
            logger.remove()
            logger.add(sink = sys.stderr, level='DEBUG')
            # log
            train_logdir = './logs/train'
            val_logdir = './logs/val'
            self.train_writer = tf.summary.create_file_writer(train_logdir)
            self.val_writer = tf.summary.create_file_writer(val_logdir)
            # model
            self.weightdir = './weights/'
            os.makedirs(self.weightdir) if not os.path.exists(self.weightdir) else None
            

