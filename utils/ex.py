import tensorflow as tf
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
target = []
prediction = []

tp = 0
for t in target:
    for p in prediction:
        if t == p:
            tp += 1
            break
fp = len(prediction) - tp
fn = len(target) - tp

print(tp, fp, fn)
precision = float(tp) / (tp + fp)
recall = float(tp) / (tp + fn)
f1 = 2*precision*recall / (precision + recall)

print precision
print recall
print f1
