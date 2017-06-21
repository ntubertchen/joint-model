import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from sap_w2v import DataPrepare
from slu_model import slu_model
import argparse
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

FLAGS = flags.FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--glove', help='glove path')
parser.add_argument('--train_p', help='training data path')
parser.add_argument('--test_p', help='testing data path')
parser.add_argument('--test', action='store_true', help='test or train')
parser.add_argument('--slot_l', help='slot tag path')
parser.add_argument('--intent_l', help='intent path')
parser.add_argument('--intent', action='store_false', help='intent training')
parser.add_argument('--slot', action='store_false', help='slot training')
args = parser.parse_args()

def load_pretrain_glove(gloveFile):
    f = open(gloveFile, 'r')
    embedding_matrix = list()
    for line in f:
        splitLine = line.strip('\n').split()
        embedding = [float(val) for val in splitLine[1:]]
        embedding_matrix.append(embedding)
    print "Done.", len(embedding_matrix)," words loaded!"
    return np.array(embedding_matrix)

if __name__ == '__main__':
    self.sess = tf.Session(config=config)
    model = slu_model()
    # read in the glove embedding matrix
    embedding = load_pretrain_glove('glove.txt')
    self.sess.run(model.init_embedding, feed_dict={model.read_embedding_matrix:embedding})
    for i in range(epoch):
        # get the data
        x, y, hist = data.get_batch()
        self.sess.run(model.train_op, feed_dict={model.input_x:hist, model.input_nl:x, model.labels:y})
    self.sess.close()
