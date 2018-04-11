from layer.layers import *
from ops.operators import *

class began(Operator): # Operatorを継承

    # initialize
    def __init__(self, sess):

        self.sess = sess

        #パラメータの設定
        self.batch_size = 16
        self.input_size = 32
        self.data_size = 32
        self.filter_number = 32
        self.embedding = 32
        self.lamda = 0.001
        self.gamma = 0.5
        self.mm = 0.5


        # placeholderの設定
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size], name='x')
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_size, self.data_size, 3], name='y')
        self.kt = tf.placeholder(tf.float32, name='kt')
        self.lr = tf.placeholder(tf.float32, name='lr')

        # Generator
        self.recon_gen = self.generator(self.x)

        # Discriminator (Encode -> Decode)
        d_real = self.decoder(self.encoder(self.y))
        d_fake = self.decoder(self.encoder(self.recon_gen, reuse=True), reuse=True)
        self.recon_dec = self.decoder(self.x, reuse=True)

        # DiscriminatorのLossをそれぞれ計算
        self.d_real_loss = l1_loss(self.y, d_real)
        self.d_fake_loss = l1_loss(self.recon_gen, d_fake)

        # Loss
        self.d_loss = self.d_real_loss - self.kt * self.d_fake_loss
        self.g_loss = self.d_fake_loss
        self.m_global = self.d_real_loss + tf.abs(self.gamma * self.d_real_loss - self.d_fake_loss)

        # Variables
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gen_")
        d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "disc_")

        # Optimizer
        self.opt_g = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.g_loss, var_list=g_vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr, self.mm).minimize(self.d_loss, var_list=d_vars)

        # initializer
        self.sess.run(tf.global_variables_initializer())

    # Generator
    def generator(self, x, reuse=None):
        with tf.variable_scope('gen_') as scope:
            if reuse:
                scope.reuse_variables()

            w = self.data_size
            f = self.filter_number
            p = "SAME"

            x = fc(x, 8 * 8 * f, name='fc')
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv1_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv1_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w / 2) # アップサンプリング(w/4 -> w/2)

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv2_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv2_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w)    # アップサンプリング(w/2 -> w)

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv3_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv3_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, 3], stride=1,  padding=p,name='conv4_a')
        return x

    # Encoder (Discriminatorの前半)
    def encoder(self, x, reuse=None):
        with tf.variable_scope('disc_') as scope:
            if reuse:
                scope.reuse_variables()

            f = self.filter_number
            h = self.embedding
            p = "SAME"

            x = conv2d(x, [3, 3, 3, f], stride=1,  padding=p,name='conv1_enc_a')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv2_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv2_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, f, 2 * f], stride=1,  padding=p,name='conv3_enc_0')
            x = pool(x, r=2, s=2) # ダウンサンプリング(w  -> w/2)

            x = conv2d(x, [3, 3, 2 * f, 2 * f], stride=1,  padding=p,name='conv3_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 2 * f, 2 * f], stride=1,  padding=p,name='conv3_enc_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [1, 1, 2 * f, 3 * f], stride=1,  padding=p,name='conv4_enc_0')
            x = pool(x, r=2, s=2) # ダウンサンプリング(w/2 -> w/4)

            x = conv2d(x, [3, 3, 3* f, 3 * f], stride=1,  padding=p,name='conv4_enc_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, 3 * f, 3 * f], stride=1,  padding=p,name='conv4_enc_b')
            x = tf.nn.elu(x)

            x = fc(x, h, name='enc_fc')
        return x

    # Decoder (Discriminatorの後半)
    def decoder(self, x, reuse=None):
        with tf.variable_scope('disc_') as scope:
            if reuse:
                scope.reuse_variables()


            w = self.data_size
            f = self.filter_number
            p = "SAME"

            x = fc(x, 8 * 8 * f, name='fc')
            x = tf.reshape(x, [-1, 8, 8, f])

            x = conv2d(x, [3, 3, f, f], stride=1, padding=p, name='conv1_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv1_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w / 2) # アップサンプリング(w/4 -> w/2)

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv2_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p, name='conv2_b')
            x = tf.nn.elu(x)

            x = resize_nn(x, w)     # アップサンプリング(w/2 -> w)

            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv3_a')
            x = tf.nn.elu(x)
            x = conv2d(x, [3, 3, f, f], stride=1,  padding=p,name='conv3_b')
            x = tf.nn.elu(x)

            x = conv2d(x, [3, 3, f, 3], stride=1,  padding=p,name='conv4_a')
        return x
