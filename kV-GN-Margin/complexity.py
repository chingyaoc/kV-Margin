import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from sklearn.metrics import pairwise_distances
import ot

tf.enable_eager_execution()

def gn_margin_loss(model, img, label, layer_idx):
    with tf.GradientTape(persistent=True) as tape:
        intermediateVal = [tf.convert_to_tensor(img)]
        tape.watch(intermediateVal)
        for l, layer_ in enumerate(model.layers):
            intermediateVal.append(layer_(intermediateVal[-1]))
        output = intermediateVal[-1]
        intermediateVal = intermediateVal[layer_idx]

        outputs = []
        margins = []
        for i in range(len(img)):
            # margins
            output_y = output[i][label[i]]
            output_max = np.max(np.concatenate([output[i][:label[i]], output[i][label[i]+1:]]))
            margins.append(output_y - output_max)
            # outputs
            y_ = np.argmax(np.concatenate([output[i][:label[i]], np.array([-10000]), output[i][label[i]+1:]]))
            outputs.append(tf.stack([output[i, label[i]], output[i, y_]]))
        margins = np.stack(margins)
        outputs = tf.stack(outputs)

        grad_1 = tape.gradient(outputs[:,0], intermediateVal)
        grad_2 = tape.gradient(outputs[:,1], intermediateVal)
        grad_1 = grad_1.numpy().reshape(grad_1.shape[0], -1)
        grad_2 = grad_2.numpy().reshape(grad_2.shape[0], -1)

        grad_norm = LA.norm(grad_1 - grad_2, ord=2, axis=1)

        return margins / (grad_norm + 1e-6)


def W1_distance(feats_0, feats_1):
    feat_dim = feats_0.shape[-1]
    k = len(feats_0)
    M = pairwise_distances(feats_0, feats_1)
    uni = np.zeros(k) + 1. / k
    kv = ot.emd2(uni, uni, M)
    return kv

def get_features(model, img, layer_idx):
    output = model(img)
    feat = model.layers[layer_idx]._last_seen_input
    feat = feat.numpy()
    return feat.reshape(feat.shape[0], -1)

def complexity(model, dataset, shallow=):
    np.random.seed(0)

    if shallow:
        layers = [2,3,4,5]
    else:
        layers = [8,9,6,7,5,4,3,2]

    # check layer
    for l in layers:
        try:
            c = [*model.get_layer(index=l).get_config().keys()]
        except:
            continue
        if 'strides' in c:     # cnn layers
            layer_idx = l
            break

    # get dataset
    images = []
    labels = []
    for image, label in dataset.take(-1):
        images.append(image.numpy())
        labels.append(label.numpy())
    images = np.stack(images)
    labels = np.stack(labels)

    # estimate class distribution
    class_size = np.max(labels) + 1
    cls_num = [len(np.where(labels == c)[0]) for c in range(class_size)]
    w = [cls_num[c] / np.sum(cls_num) for c in range(class_size)]

    # sample subset for estimation
    data_size = min(len(images), class_size*200)
    idx = np.random.choice(np.where(labels > -1)[0], data_size, replace=False)
    images = images[idx]
    labels = labels[idx]

    # margin score (drop last)
    margins = []
    for i in range(int(data_size / 200)):
        margin_i = gn_margin_loss(model, images[200*i:200*(i+1)], labels[200*i:200*(i+1)], layer_idx)
        margins.append(margin_i)
    margin = np.concatenate(margins)
    margin_score = np.median(margin)

    # determine the k-variance normalizer
    gamma = 0.
    for c in range(class_size):
        # estimate k_variance
        idx = np.random.permutation(np.where(labels == c)[0])
        k = int(len(idx)/2)
        feature = get_features(model, images[idx], layer_idx)
        kv = W1_distance(feature[:k], feature[k:k*2])
        gamma += w[c] * kv

    score = -margin_score / gamma

    return score
