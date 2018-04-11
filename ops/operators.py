import sys, os
import glob
import time
import datetime
import numpy as np
import scipy.misc as scm
from function.img_read import *
from function.img_gen import *
from PIL import Image

def inverse_image(img):
    img = (img + 0.5) * 255.
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[..., ::-1] # bgr to rgb
    return img

class Operator:

    # trainの定義
    def train(self, train_flag):

        # パラメータの設定
        self.learning_rate = 0.0001
        self.niter = 50
        self.niter_snapshot = 2000/self.batch_size

        # load data
        data_path = "./Data/cifar-10-batches-py/data_batch_1"
        d = unpickle(data_path)
        self.data = d["data"].reshape(-1, 32, 32, 3)

        # initial parameter
        start_time = time.time()
        kt = np.float32(0.)
        lr = np.float32(self.learning_rate)
        self.count = 0
        self.j = 0

        for epoch in range(self.niter):

            batch_idxs = len(self.data) // self.batch_size

            for idx in range(0, batch_idxs):

                # 乱数と学習データの取得
                batch_x = np.random.uniform(-1., 1., size=[self.batch_size, self.input_size])
                batch_mask = np.random.choice(len(self.data), self.batch_size)
                batch_data = self.data[batch_mask]
                batch_data = batch_data/255

                # opt & feed list (different with paper)
                g_opt = [self.opt_g, self.g_loss, self.d_real_loss, self.d_fake_loss]
                d_opt = [self.opt_d, self.d_loss]
                feed_dict = {self.x: batch_x, self.y: batch_data, self.kt: kt, self.lr: lr}

                # run tensorflow
                _, loss_g, d_real_loss, d_fake_loss = self.sess.run(g_opt, feed_dict=feed_dict)
                _, loss_d = self.sess.run(d_opt, feed_dict=feed_dict)

                if self.count % self.niter_snapshot == 0:
                    samples = self.sess.run(self.recon_gen, feed_dict={self.x: batch_x})
                    fig = plot(samples)
                    plt.savefig('output/{}.png'.format(str(self.j).zfill(5)), bbox_inches='tight')
                    self.j += 1
                    plt.close(fig)

                # update kt, m_global
                kt = np.maximum(np.minimum(1., kt + self.lamda * (self.gamma * d_real_loss - d_fake_loss)), 0.)
                m_global = d_real_loss + np.abs(self.gamma * d_real_loss - d_fake_loss)
                loss = loss_g + loss_d

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "
                      "loss: %.4f, loss_g: %.4f, loss_d: %.4f, d_real: %.4f, d_fake: %.4f, kt: %.8f, M: %.8f"
                      % (epoch, idx, batch_idxs, time.time() - start_time,
                         loss, loss_g, loss_d, d_real_loss, d_fake_loss, kt, m_global))

                self.count += 1
