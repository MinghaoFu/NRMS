import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import time
import logging
import numpy as np
from tqdm import tqdm
from recommenders.models.deeprec.deeprec_utils import cal_metric

from attention import multiHeadSelfAttention, additiveAttention


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class model:
    def __init__(
        self,
        hparams,
        data_loader,
        test_data_loader,
        seed
    ):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(self.seed)
        
        if tf.config.experimental.list_physical_devices('GPU'):
            gpuOptions = tf.compat.v1.GPUOptions(allow_growth=True)
            session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuOptions))
            tf.compat.v1.keras.backend.set_session(session)
            logger.info('Tensorflow is using GPU')
        else:
            session = tf.compat.v1.Session()
            tf.compat.v1.keras.backend.set_session(session)
            logger.info('Tensorflow is not using GPU')

        self.hparams = hparams
        self.word_embedding = np.load(self.hparams.wordEmb_file)
        
        # MINDiterator
        self.train_data_loader = data_loader(hparams,hparams.npratio,col_spliter='\t')
        self.test_data_loader = test_data_loader(hparams,col_spliter='\t')

        self.model, self.scorer = self.nrms_graph(self.nrms_core)
        #self.model, self.scorer = self.nrms_core()
        # categorical cross entropy, adam optimizer
        adam_optimizer = keras.optimizers.Adam(learning_rate=self.hparams.learning_rate)
        self.model.compile(loss=self.hparams.loss, optimizer=adam_optimizer)

    def build_news_encoder(self, embedding_layer):
        news_title = keras.Input(shape=(self.hparams.title_size,), dtype='int32')
        # first layer
        embedded_news = embedding_layer(news_title)
        # dropout 0.25, avoid over-fitting
        x = layers.Dropout(self.hparams.dropout)(embedded_news)
        # second layer
        x = multiHeadSelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)([x, x, x])
        x = layers.Dropout(self.hparams.dropout)(x)
        # last
        news_presentation = additiveAttention(self.hparams.attention_hidden_dim, seed=self.seed)(x)

        return keras.Model(news_title, news_presentation, name='build_news_encoder')

    def build_user_encoder(self, news_encoder):
        user_click_history = keras.Input(shape=(self.hparams.his_size, self.hparams.title_size), dtype='int32')
        
        x = layers.TimeDistributed(news_encoder)(user_click_history)
        x = multiHeadSelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)([x, x, x])
        user_presentation = additiveAttention(self.hparams.attention_hidden_dim, seed=self.seed)(x)
        
        return keras.Model(user_click_history, user_presentation, name='build_user_encoder')

    def nrms_core(self):
        # nrms core logic, guideline NRMS
        embedding_layer = layers.Embedding(
            self.word_embedding.shape[0],
            self.word_embedding.shape[1],
            weights=[self.word_embedding],
            trainable=True,
        )
        self.news_encoder = self.build_news_encoder(embedding_layer)
        self.user_encoder = self.build_user_encoder(self.news_encoder)

        user_click_history = keras.Input(shape=(self.hparams.his_size, self.hparams.title_size), dtype='int32')
        # negative samples and poss
        candidate_samples = keras.Input(shape=(self.hparams.npratio + 1, self.hparams.title_size), dtype='int32')
        candidate_one = keras.Input(shape=(1, self.hparams.title_size, ), dtype='int32')

        user_rep = self.user_encoder(user_click_history)
        candidate_rep = layers.TimeDistributed(self.news_encoder)(candidate_samples)
        candidate_one_rep = self.news_encoder(layers.Reshape((self.hparams.title_size,))(candidate_one))

        # dot product
        scores = layers.Dot(axes=-1)([candidate_rep, user_rep])
        scores = layers.Activation(activation='softmax')(scores)

        scores_one = layers.Dot(axes=-1)([candidate_one_rep, user_rep])
        scores_one = layers.Activation(activation='sigmoid')(scores_one)

        model = keras.Model([user_click_history, candidate_samples], scores)
        scorer = keras.Model([user_click_history, candidate_one], scores_one)

        return model, scorer

    def nrms_graph(self, core):
        return core()
    
    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
    ):
        # train model, adam, batch 32
        logger.info('start training, total epochs: {0:d}, batch size: {1:d}'
                    .format(self.hparams.epochs, self.hparams.batch_size))
        for epoch in range(0, self.hparams.epochs):
            train_start = time.time()
            steps = 0
            total_loss = 0

            progress_bar = tqdm(self.train_data_loader.load_data_from_file(train_news_file, train_behaviors_file))
            for batch_data in progress_bar:
                batch_loss = self.model.train_on_batch(
                    [batch_data['clicked_title_batch'], batch_data['candidate_title_batch']],
                    batch_data['labels']
                )
                total_loss += batch_loss
                steps += 1
                if steps % self.hparams.show_step == 0:
                    progress_bar.set_description('epoch {3:d}, step {0:d}, average loss: {1:.6f}, step loss: {2:.6f}, total iteration: {4:d}'.format(steps, total_loss / steps, batch_loss, epoch + 1, len(progress_bar)))

            train_end = time.time()

            eval_start = time.time()

            imp_indices, labels, predictions = self.evaluate(valid_news_file, valid_behaviors_file)

            eval_end = time.time()
            eval_info = cal_metric(labels, predictions, self.hparams.metrics)

            print('epoch {0:d}, training time: {1:.3f}s, evaluating time: {2:.3f}s, evaluate info: {3}'
                  .format(epoch + 1, train_end - train_start, eval_end - eval_start, eval_info))

        return self

    def build_news_rep_dict(self, news_file):
        # build newsId-newsRepresentation dictionary for evaluate
        logger.info('build newsId-newsRepresentation dictionary')
        news_indices = []
        news_reps = []

        for batch_data in tqdm(self.test_data_loader.load_news_from_file(news_file)):
            news_titles_batch = batch_data['candidate_title_batch']
            news_reps_batch = self.news_encoder.predict_on_batch(news_titles_batch)
            news_indices_batch = batch_data['news_index_batch']

            news_reps.extend(news_reps_batch)
            news_indices.extend(news_indices_batch)

        return dict(zip(news_indices, news_reps))

    def build_imp_urep_dict(self, news_file, behaviors_file):
        # build impressionId-userRepresentation dictionary for evaluate
        logger.info('build impressionId-userRepresentation dictionary')
        imp_indices = []
        user_reps = []

        for batch_data in tqdm(self.test_data_loader.load_test_user_from_file(news_file, behaviors_file)):

            user_history_batch = batch_data['clicked_title_batch']
            user_reps_batch = self.user_encoder.predict_on_batch(user_history_batch)
            imp_indices_batch = batch_data['impr_index_batch']

            imp_indices.extend(imp_indices_batch)
            user_reps.extend(user_reps_batch)

        return dict(zip(imp_indices, user_reps))


    def evaluate(self, news_file, behaviors_file):
        news_rep_dict = self.build_news_rep_dict(news_file)
        imp_urep_dict = self.build_imp_urep_dict(news_file, behaviors_file)

        imp_indices = []
        labels = []
        predictions = []

        for (
                imp_index,
                news_index,
                user_index,
                label,
        ) in tqdm(self.test_data_loader.load_impression_from_file(behaviors_file)):
            imp_predictions = np.dot([news_rep_dict[i] for i in news_index], imp_urep_dict[imp_index])
            imp_indices.append(imp_index)
            labels.append(label)
            predictions.append(imp_predictions)

        return imp_indices, labels, predictions


    def predict(self, news_file, behaviors_file):
        # get labels and pridictions to call metrics
        news_rep_dict = self.build_news_rep_dict(news_file)
        imp_urep_dict = self.build_imp_urep_dict(news_file, behaviors_file)

        imp_indices = []
        predictions = []

        for (
            imp_index,
            news_index,
            user_index,
        ) in tqdm(self.test_data_loader.load_test_impression_from_file(behaviors_file)):
            imp_predictions = np.dot([news_rep_dict[i] for i in news_index], imp_urep_dict[imp_index])
            imp_indices.append(imp_index)
            predictions.append(imp_predictions)

        return imp_indices, predictions



    




