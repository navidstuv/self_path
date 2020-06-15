import os
import sys
import torch
import logging
import argparse
import datetime
import collections
from torch.autograd import Function
import random
from skimage.io import imsave
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, \
    f1_score, accuracy_score, roc_curve, average_precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from inspect import signature
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import random_projection
import hdbscan


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def cls_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval_seg(label_preds, label_trues, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lp, lt, n_class)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    return mean_iou


def fast_hist(label_pred, label_true, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    return np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2).reshape(n_class, n_class)


def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def make_inf_dl(dl):
    while True:
        try:
            data_iter = iter(dl)
            yield next(data_iter)
        except StopIteration:
            del (data_iter)
            data_iter = iter(dl)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def f11_score(precision, recall):
    f1 = 2 * np.divide(np.multiply(precision, recall), np.add(precision, recall))
    return f1


def stats(soft_labels, true_labels, opt_thresh = 0.5):
    '''
    prediction should be soft labels

    :param pred:
    :param true:
    :return:
    '''

    tumour_class = [x[1] for x in soft_labels]

    def thresh_fuc(input, thres_value):
        if input < thres_value:
            input = 0
        else:
            input = 1
        return input

    # pred_labels = tumour_class.copy()
    pred_labels = [thresh_fuc(i, opt_thresh) for i in tumour_class]

    # pred_labels = np.argmax(soft_labels, axis=-1)
    # GT = test_generator.classes
    # tumour_class=  np.array(tumour_class)
    # threshold = 0.5
    fpr, tpr, thresholds = roc_curve(true_labels, tumour_class, pos_label=1)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Tumor')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # ---------------------------------------------

    precision, recall, thresholds = precision_recall_curve(true_labels,
                                                           tumour_class)
    f1 = f11_score(precision, recall)
    nan_places = np.isnan(f1)
    f1[nan_places] = 0

    print('max f1 score:{} optimal thresh: {}'.format(np.amax(f1), thresholds[np.where(f1 == np.amax(f1))]))
    average_precision = average_precision_score(true_labels, tumour_class)

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.9, num=5)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    l, = plt.plot(recall, precision, color='turquoise', lw=2)
    lines.append(l)
    labels.append('Precision-recall for class Tumour (area = {:0.4f})'.format(average_precision))

    l, = plt.plot(recall, f1, color='cornflowerblue', lw=2)
    lines.append(l)
    labels.append(
        'max f1 score:{:0.2f} optimal thresh: {:0.2f}'.format(np.amax(f1), thresholds[np.where(f1 == np.amax(f1))][0]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    # ---------------------------------------------
    precision, recall, _ = precision_recall_curve(true_labels, tumour_class)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision_score(true_labels, tumour_class)))
    plt.show()
    Auc = roc_auc_score(true_labels, tumour_class)
    # tumour_class[tumour_class>threshold] = 1
    # tumour_class[tumour_class <= threshold] = 0

    F1 = f1_score(true_labels, pred_labels, pos_label=1)
    ACC = accuracy_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    print('f1 score is: {}'.format(F1))
    print('Accuracy score is: {}'.format(ACC))
    print('Auc is: {}'.format(Auc))
    print(f'averag precision: {average_precision}')
    print('conf_matrix is: {}'.format(conf_matrix))
    print('Precision is {}'.format(precision))
    print('recall is {}'.format(recall))
    return Auc

def show_images(images, iter, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(10, 10))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        a.set_title(str(title))
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    plt.savefig('../patches/'+str(iter)+'.png')
    # plt.close()

def save_output_img(imgs,path, prefix, num):
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(imgs.shape[0]):
        imsave(os.path.join(path, prefix + '_' + str(i + 1 + num ) + '.png'), np.transpose(imgs[i, :, :, :], (1, 2, 0)) )

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def kmean(train_features, test_feature):
    kmeans = KMeans(n_clusters=2, random_state=0)
    preds = kmeans.fit_predict(train_features)
    return preds
def GMM(train_features, test_feature):
    gmm = GaussianMixture(n_components=2, random_state=0).fit(train_features)
    probs = gmm.predict_proba(test_feature)
    return probs
def HDBscan(train_features):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
    preds = clusterer.fit_predict(train_features)
    return preds
def random_proj(features):
    transformer = random_projection.GaussianRandomProjection(n_components=2)
    features_new = transformer.fit_transform(features)
    return features_new
