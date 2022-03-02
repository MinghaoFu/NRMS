import tensorflow as tf
import numpy as np
from tqdm import tqdm

from recommenders.models.newsrec.io.mind_iterator import MINDIterator


__all__ = ["MINDTestIterator"]


class MINDTestIterator(MINDIterator):
    def __init__(
        self,
        hparams,
        npratio=-1,
        col_spliter="\t",
        ID_spliter="%",
    ):
        self.num = 0
        super(MINDTestIterator, self).__init__(hparams, npratio, col_spliter, ID_spliter)

    def init_test_behaviors(self, behaviors_file):
        print('init test behaviors data')
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in tqdm(rd):
                self.num += 1
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[
                    : self.his_size
                ]

                impr_news = [self.nid2index[i] for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1


    def load_test_user_from_file(self, news_file, behavior_file):
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        self.init_test_behaviors(behavior_file)

        user_indexes = []
        impr_indexes = []
        click_title_indexes = []
        cnt = 0

        for index in range(len(self.impr_indexes)):
            click_title_indexes.append(self.news_title_index[self.histories[index]])
            user_indexes.append(self.uindexes[index])
            impr_indexes.append(self.impr_indexes[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_user_data(
                    user_indexes,
                    impr_indexes,
                    click_title_indexes,
                )
                user_indexes = []
                impr_indexes = []
                click_title_indexes = []
                cnt = 0

        if cnt > 0:
            yield self._convert_user_data(
                user_indexes,
                impr_indexes,
                click_title_indexes,
            )

    def load_test_impression_from_file(self, behaivors_file):
        if not hasattr(self, "histories"):
            self.init_test_behaviors(behaivors_file)

        indexes = np.arange(self.num)

        for index in indexes:
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
            )



