import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.BEGAN import began
from function.img_gen import plot

# session の定義
sess = tf.Session()

""" beganネットワークのセットアップ """
model = began(sess)

""" 実行 """
model.train(train_flag=True)
