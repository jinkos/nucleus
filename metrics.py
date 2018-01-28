from __future__ import print_function

from skimage.morphology import label
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# displays a list of images
def display_grid_from_list(nw,nh,img_list):
 
  # Set figure width to 12 and height to 9
  fig_size = plt.rcParams["figure.figsize"]
  fig_size[0] = 12
  fig_size[1] = 9
  plt.rcParams["figure.figsize"] = fig_size

  for h in range(nh):
    for w in range(nw):
      index = h * nw + w
      if index >= len(img_list): break
      plt.subplot(nh,nw,index+1)
      plt.imshow(img_list[index])
    if index >= len(img_list): break
  
  plt.show()

def display_labels(_labels):
  np_labels = np.array(_labels,dtype=int)
  patch_list = []
  for uni in np.unique(_labels):
    patch = np.where(np_labels==uni,1.0,0.0)
    patch_list.append(patch)
    
  display_grid_from_list(5,4,patch_list)

# not sure how half of this works as it is stolen from Kaggle Kernals
# y_true_in: greyscale image
# y_pred_in: greyscale image
def iou_metric(y_true_in, y_pred_in, print_table=False, do_plots = False):

  # convert greyscale to B&W
  # label() will split B&W images into a list of regions
  labels = label(y_true_in > 0.5)
  y_pred = label(y_pred_in > 0.5)
  
  n_true_objects = len(np.unique(labels))
  n_pred_objects = len(np.unique(y_pred))

  if do_plots:
    print("n_true_objects",n_true_objects)
    print("n_pred_objects",n_pred_objects)
    print("labels.shape",labels.shape)
    print("y_pred.shape",y_pred.shape)

    display_labels(labels)
    display_labels(y_pred)

  intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(n_true_objects, n_pred_objects))[0]

  # Compute areas (needed for finding the union between all objects)
  area_true = np.histogram(labels, bins = n_true_objects)[0]
  area_pred = np.histogram(y_pred, bins = n_pred_objects)[0]
  area_true = np.expand_dims(area_true, -1)
  area_pred = np.expand_dims(area_pred, 0)

  # Compute union
  union = area_true + area_pred - intersection

  # Exclude background from the analysis
  intersection = intersection[1:,1:]
  union = union[1:,1:]
  union[union == 0] = 1e-9

  # Compute the intersection over union
  iou = intersection / union

  # Precision helper function
  def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

  # Loop over IoU thresholds
  prec = []
  if print_table:
    print("Thresh\tTP\tFP\tFN\tPrec.")
  for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    if (tp + fp + fn) > 0:
      p = tp / (tp + fp + fn)
    else:
      p = 0
    if print_table:
      print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)
  
  if print_table:
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
  return np.mean(prec)
  
def iou_metric_batch(y_true_in, y_pred_in):
  batch_size = y_true_in.shape[0]
  metric = []
  for batch in range(batch_size):
    value = iou_metric(y_true_in[batch], y_pred_in[batch])
    metric.append(value)
  return np.array(np.mean(metric), dtype=np.float32)

# tf callable version of iou_metric
def my_iou_metric(label, pred):
  metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
  return metric_value
