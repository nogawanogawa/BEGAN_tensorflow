import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import _pickle as cPickle


def unpickle(f):
    fo = open(f, 'rb')
    d = cPickle.load(fo, encoding='latin1')
    fo.close()
    return d
