from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_curve, auc, mean_absolute_error

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Nadam
from keras.models import load_model

from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import pyphen
# import kenlm

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from neural_network_utils import auc

import keras.backend as K
import tensorflow as tf


class Instance():
    """
    Each line represents a sentence with one complex word annotation and
    relevant information, each separated by a TAB character.
        - The first column shows the HIT ID of the sentence. All sentences with
        the same ID belong to the same HIT.
        - The second column shows the actual sentence where there exists a
        complex phrase annotation.
        - The third and fourth columns display the start and end offsets of the
        target word in this sentence.
        - The fifth column represents the target word.
        - The sixth and seventh columns show the number of native annotators
        and the number of non-native annotators who saw the sentence.
        - The eighth and ninth columns show the number of native annotators and
        the number of non-native annotators who marked the target word as
        difficult.
        - The tenth and eleventh columns show the gold-standard label for the
        binary and probabilistic classification tasks.
    """

    def __init__(self, sentence, test=False):
        self.hit_id = sentence[0]
        self.sentence = sentence[1]
        self.offset = [int(sentence[2]), int(sentence[3])]
        self.target_chars = sentence[4]
        self.annotators = [int(sentence[5]), int(sentence[6])]
        self.tokens, self.target = self.tokenize()
        self.target.sort()
        if not test:
            self.label = [int(sentence[9]), float(sentence[10])]
            self.difficult = [int(sentence[7]), int(sentence[8])]
        else:
            self.label = None
            self.difficult = None

    def __str__(self):
        string = "HIT ID: %s\nSENTENCE: %s\nTOKENS: %s\nOFFSET: %s\nTARGET_CHARS: %s\nTARGET_TOKENS: %s\nANNOTATORS: %s\nDIFFICULT: %s\nLABEL: %s"
        data = (self.hit_id,
                self.sentence,
                self.tokens,
                '%s --> %s' % (self.offset,
                               self.sentence[self.offset[0]:self.offset[1]]),
                self.target_chars,
                '%s --> %s' % (self.target,
                               [self.tokens[i] for i in self.target]),
                self.annotators,
                self.difficult,
                self.label)
        return string % data

    def tokenize(self):
        tokens = []
        target = {}
        endings = ['?', '”', '"', '!', ')', '.']
        start = 0
        for i in range(len(self.sentence)):
            if i in range(self.offset[0], self.offset[1]):
                target[len(tokens)] = True
            if self.sentence[i] == ' ':
                if self.sentence[start:i]:
                    tokens.append(self.sentence[start:i].lower())
                start = i + 1
            elif self.sentence[i] == '\'':
                tokens.append(self.sentence[start:i].lower())
                start = i
            elif self.sentence[i] in (endings + [',', ':', ';', '(', ')', '', '…', '[', ']']):
                if i - start > 0:
                    tokens.append(self.sentence[start:i].lower())
                tokens.append(self.sentence[i:i+1].lower())
                start = i + 1
            elif self.sentence[i] in ['\"', '“']:
                if i - start >= 0:
                    tokens.append(self.sentence[i:i + 1].lower())
                start = i + 1
            if i == len(self.sentence)-1 and self.sentence[i] not in endings:
                tokens.append(self.sentence[start:].lower())
        return tokens, list(target.keys())

class Data():
    """Docstring."""

    def __init__(self, datasets, is_test=False):
        self.instances = []
        self.y = []
        self.y_prob = []
        self.test = is_test
        self.load_data(datasets)

    def load_data(self, datasets):
        for dataset in datasets:
            with open(dataset, encoding = 'utf-8') as fp:
                lines = [line.split('\t') for line in fp.read().splitlines()]
            # assert data consistance
            for line in lines:
                print('\n\n\n',line,'\n\n\n')
                if not self.test:
                    assert len(line) == 11, "Missing field: %s" % line
                else:
                    assert len(line) == 7, "Missing field: %s" % line
            # create instances
            for line in lines:
                self.instances.append(Instance(line, test=self.test))
        if not self.test:
            self.y = np.array([instance.label[0] for instance in self.instances])
            self.y_prob = np.array([instance.label[1] for instance in self.instances])

    def statistics(self):
        print('Instances: %d' % len(self.instances))
        unique_insts = len(set([i.hit_id for i in self.instances]))
        unique_targets = len(set([i.target_chars for i in self.instances]))
        sents_len = [len(i.sentence) for i in self.instances]
        tokens_len = [len(i.tokens) for i in self.instances]
        print('Unique instances: %d' % unique_insts)
        print('Unique targets: %d' % unique_targets)
        print('Sentences char length: %.2f (±%.2f)' % (np.mean(sents_len),
                                                       np.std(sents_len)))
        print('Sentences token length: %.2f (±%.2f)' % (np.mean(tokens_len),
                                                        np.std(tokens_len)))

class FeatureExtractor():
    """Docstring."""

    def __init__(self, psycho_path=None, lm_books_path=None, lm_news_path=None, embedding_model_path=None):
        """Docstring"""

        if psycho_path:
            self.df = pd.read_csv(psycho_path, sep='\t')
            self.df_mean = self.df.mean(axis=0)
        if lm_books_path:
            self.lm_books = kenlm.LanguageModel(lm_books_path)
        if lm_news_path:
            self.lm_news = kenlm.LanguageModel(lm_news_path)
        if embedding_model_path:
            self.embeddings = KeyedVectors.load_word2vec_format(embedding_model_path, binary=True)
        self.syllables = pyphen.Pyphen(lang='en')

    def lexical(self, words):
        """Extract lexical features."""

        lexicals = []
        for word in words:
            dic = {'chars':0, 'syllables':0}
            dic['chars'] = len(word)
            dic['syllables'] = len(self.syllables.positions(word)) + 1
            lexicals.append(pd.Series(dic, index=dic.keys()))
        df = pd.DataFrame(lexicals)
        df = df.rolling(df.shape[0]).agg(['mean', 'std', 'min', 'max'])[-1:]
        df.columns = df.columns.map('_'.join)
        return df

    def wordnet(self, words):
        """Extract wordnet features."""

        wordnets = []
        for word in words:
            dic = {'synsets':0, 'hypernyms':0, 'hyponyms':0}
            syns = wn.synsets(word)
            dic['synsets'] = len(syns)
            for syn in syns:
                dic['hypernyms'] += len(syn.hypernyms())
                dic['hyponyms'] += len(syn.hyponyms())
            wordnets.append(pd.Series(dic, index=dic.keys()))
        df = pd.DataFrame(wordnets)
        df = df.rolling(df.shape[0]).agg(['mean', 'std', 'min', 'max'])[-1:]
        df.columns = df.columns.map('_'.join)
        return df

    def psycholinguistics(self, words):
        """Extract psycholinguistic features."""

        psychos = []
        for word in words:
            psycho = {'Familiarity': 0, 'Age_of_Acquisition':0, 'Concreteness':0,'Imagery': 0}
            infos = self.df[self.df.Word == word]
            if not infos.empty:
                for key in psycho.keys():
                    psycho[key] += infos[key].values[0]
            else:
                for key in psycho.keys():
                    psycho[key] += self.df_mean[key]
            psychos.append(pd.Series(psycho, index=psycho.keys()))
        df = pd.DataFrame(psychos)
        df = df.rolling(df.shape[0]).agg(['mean', 'std', 'min', 'max'])[-1:]
        df.columns = df.columns.map('_'.join)
        return df

    def language_model(self, tokens):
        """Extract language model features."""

        model = {'LM-Book_log10':0, 'LM-News_log10':0}
        model['LM-Book_log10'] = self.lm_books.score(' '.join(tokens), bos=False, eos=False)
        model['LM-News_log10'] = self.lm_news.score(' '.join(tokens), bos=False, eos=False)
        return model

    def predict_average_embeddings(self, instances):
        """Extract average value of target words embeddings."""

        data_embeddings = []
        for index, instance in enumerate(instances):
            words = []
            for i in instance.target:
                if instance.tokens[i] in self.embeddings:
                    words.append(self.embeddings[instance.tokens[i]])
            if len(words) == 0:
                words.append(self.embeddings['unk'])
            data_embeddings.append(np.average(words, axis=0))
        return np.asarray(data_embeddings, )

    def predict_linguistics(self, instances):
        """Extract features for every instance."""

        df = pd.DataFrame()
        for instance in instances:
            features = dict()
            tokens = [instance.tokens[i] for i in instance.target]

            features.update(self.psycholinguistics(tokens))
            features.update(self.language_model(tokens))
            features.update(self.wordnet(tokens))
            features.update(self.lexical(tokens))
            df = df.append(pd.DataFrame(features))
            df = df.reset_index().drop('index', axis=1)
            df.fillna(0, inplace=True)
        return df


def create_nn():
    """Create a simple neural network"""

    model = Sequential()
    model.add(Dense(100, input_shape=(100,), activation='relu'))
    model.add(Dense(100, input_shape=(100,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def grid_search_classifiers(model, params):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, scoring='roc_auc')
    grid_result = grid_search.fit(x_train, y_train)
    print(grid_result.best_params_)


def grid_search_regressors(model, params):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, scoring='neg_mean_absolute_error')
    grid_result = grid_search.fit(x_train, y_train)
    print(grid_result.best_params_)

def print_roc(pred, y, message='Roc curve'):
    fpr, tpr, _ = roc_curve(y.ravel(), pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(message)
    plt.legend(loc="lower right")
    plt.show()

def best_threshold(pred, y, metric='accuracy'):
    best = 0
    th = 0
    if metric == 'precision':
        metric = precision_score
    elif metric == 'recall':
        metric = recall_score
    elif metric == 'f1':
        metric = f1_score
    else:
        metric = accuracy_score
    for i in np.arange(pred.min(), pred.max(), 0.01):
        aux = pred.copy()
        aux[aux >= i] = 1
        aux[aux < i] = 0
        if metric == f1_score:
            new = metric(aux, y, average='macro')
        else:
            new = metric(aux, y)
        if new > best:
            best = new
            th = i
    return th

def evaluate(pred_train, pred_test, y_train, y_test, optimize='accuracy', label='ROC Curve'):
    print('--Optimizing %s--' % optimize)
    th = best_threshold(pred_train, y_train, optimize)
    print('Threshold: %.2f' % th)
    aux = pred_test.copy()
    aux[aux >= th] = 1
    aux[aux < th] = 0
    print('--Scores--')
    print('Accuracy: %.2f' % accuracy_score(aux, y_test))
    print('Precision: %.2f' % precision_score(aux, y_test))
    print('Recall: %.2f' % recall_score(aux, y_test))
    print('F1: %.2f' % f1_score(aux, y_test))
    print('--Confusion matrix:--\n %s' % confusion_matrix(aux, y_test))
    # print_roc(pred_test, y_test, label)



path = 'dataset/'

embeddings_path = 'glove.100d.bin'
neural_network_model_path = 'model.h5'

training_data = [path + 'News_Train.tsv',
                 path + 'WikiNews_Train.tsv',
                 path + 'Wikipedia_Train.tsv',
                ]

dev_data = [path + 'News_Dev.tsv',
            path + 'WikiNews_Dev.tsv',
            path + 'Wikipedia_Dev.tsv'
            ]

dev_data_news = [path + 'News_Dev.tsv']
dev_data_wikinews = [path + 'WikiNews_Dev.tsv']
dev_data_wikipedia = [path + 'Wikipedia_Dev.tsv']

test_data = [path + 'News_Test.tsv',
             path + 'WikiNews_Test.tsv',
             path + 'Wikipedia_Test.tsv'
            ]

test_data_news = [path + 'News_Test.tsv']
test_data_wikinews = [path + 'WikiNews_Test.tsv']
test_data_wikipedia = [path + 'Wikipedia_Test.tsv']


def main():

    data_train = Data(training_data)
    data_dev = Data(dev_data)
    data_test = Data(test_data, is_test=True)

    # data.load_data(data_train)
    # print('=====STATISTICS=====')
    # data.statistics()
    # print('=====EXAMPLE=====')
    # print(data.instances[0])

    # DEV
    data_dev_news = Data(dev_data_news)
    data_dev_wikinews = Data(dev_data_wikinews)
    data_dev_wikipedia = Data(dev_data_wikipedia)

    # TEST
    data_test_news = Data(test_data_news, is_test=True)
    data_test_wikinews = Data(test_data_wikinews, is_test=True)
    data_test_wikipedia = Data(test_data_wikipedia, is_test=True)

    fe = FeatureExtractor(embedding_model_path=embeddings_path)

    x_train = fe.predict_average_embeddings(data_train.instances)
    y_train, y_train_prob = data_train.y, data_train.y_prob

    x_dev = fe.predict_average_embeddings(data_dev.instances)
    y_dev, y_dev_prob = data_dev.y, data_dev.y_prob

    x_test = fe.predict_average_embeddings(data_test.instances)
    y_test, y_test_prob = data_test.y, data_test.y_prob

    # DEV
    x_dev_news = fe.predict_average_embeddings(data_dev_news.instances)
    y_dev_news, y_dev_news_prob = data_dev_news.y, data_dev_news.y_prob

    x_dev_wikinews = fe.predict_average_embeddings(data_dev_wikinews.instances)
    y_dev_wikinews, y_dev_wikinews_prob = data_dev_wikinews.y, data_dev_wikinews.y_prob

    x_dev_wikipedia = fe.predict_average_embeddings(data_dev_wikipedia.instances)
    y_dev_wikipedia, y_dev_wikipedia_prob = data_dev_wikipedia.y, data_dev_wikipedia.y_prob

    # TEST
    x_test_news = fe.predict_average_embeddings(data_test_news.instances)
    x_test_wikinews = fe.predict_average_embeddings(data_test_wikinews.instances)
    x_test_wikipedia = fe.predict_average_embeddings(data_test_wikipedia.instances)


    # UNCOMMENT TO TRAIN MODEL
    # model = create_nn()
    # # optimizer = Adam(lr=1e-6)
    # optimizer = Nadam(lr=2e-3, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
    # callbacks_list = [ModelCheckpoint(path + 'model/trainedModel.h5', monitor='val_auc', verbose=1, save_best_only=True,
    #                                   mode='max'), ]
    # model.compile(optimizer=optimizer,
    #               loss='binary_crossentropy',
    #               metrics=[auc])
    #
    # model.fit(
    #     x_train,
    #     y_train,
    #     validation_data=(x_dev, y_dev),
    #     epochs=20,
    #     batch_size=32,
    #     callbacks=callbacks_list)

    # USING BUILT MODEL
    model = load_model(neural_network_model_path, custom_objects={'auc': auc})

    pred_train = model.predict(x_train)
    pred_dev = model.predict(x_dev)
    pred_test = model.predict(x_test)
    evaluate(pred_train, pred_dev, y_train, y_dev, optimize='f1', label='Average Embedding Neural Network')

if __name__=="__main__":
    main()