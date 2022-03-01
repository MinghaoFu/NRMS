import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
from model import model
tf.get_logger().setLevel('ERROR') # only show error messages


from recommenders.models.deeprec.deeprec_utils import (download_deeprec_resources, cal_metric)
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

seed = 42

# Options: demo, small, large
MIND_type = 'large'
data_path = os.path.join('data_large')

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(os.path.join(data_path, 'train')):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

if not os.path.exists(os.path.join(data_path, 'valid')):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)

if not os.path.exists(os.path.join(data_path, 'utils')):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/',
                               os.path.join(data_path, 'utils'), mind_utils)

# hyperparameter
yaml_file = os.path.join(r'config.yaml')
hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file)

model = model(hparams, MINDIterator, seed)
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

imp_indices, labels, predictions = model.evaluate(valid_news_file, valid_behaviors_file)
eval_info = cal_metric(labels, predictions, hparams.metrics)
print(eval_info)

model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "nrms_model_weight"))

imp_indices, labels, predictions = model.evaluate(valid_news_file, valid_behaviors_file)

with open(os.path.join(data_path, 'prediction.txt'), 'w') as f:
    for imp_index, prediction in tqdm(zip(imp_indices, predictions)):
        imp_index += 1
        rank = (np.argsort(np.argsort(prediction)[::-1]) + 1).tolist()
        rank = '[' + ','.join([str(i) for i in rank]) + ']'
        f.write(' '.join([str(imp_index), rank])+ '\n')


f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')
f.close()