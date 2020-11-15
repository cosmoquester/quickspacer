import tensorflow as tf
from tensorflow.keras import backend as K


def learning_rate_scheduler(num_epochs, max_learning_rate, min_learninga_rate=1e-7):
    lr_delta = (max_learning_rate - min_learninga_rate) / num_epochs

    def _scheduler(epoch, lr):
        return lr - lr_delta

    return _scheduler


def f1_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    TP = tf.reduce_sum(y_true * y_pred, axis=0)
    FP = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    FN = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(f1)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_socre = recall(y_true, y_pred)
    return 2 * (precision_score * recall_socre) / (precision_score + recall_socre + K.epsilon())
