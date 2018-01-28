from __future__ import print_function

from skimage.morphology import label
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
import numpy as np
import os

import image_wrapper
import arg_config
import our_models
import metrics

shear_tag_list = ['sh','shu','shd','shl','shr']
val_split = 6

# valid and train dataset lists
idstr_train = []
idstr_valid = []
idstr_test = []
n_idstr_train = 0
n_idstr_valid = 0
n_idstr_test = 0

def load_test_data():

  global idstr_train
  global idstr_valid
  global idstr_test

  global n_idstr_train
  global n_idstr_valid
  global n_idstr_test

    # only load once
  if n_idstr_train > 0: return
  
  # Get train IDs from directory names
  train_path = arg_config.cfg['train_path']
  idstrs = next(os.walk(train_path))[1]
  
  # remove duff training examples
  if '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80' in idstrs:
    print('ignoring 7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
    idstrs.remove('7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
    
  do_short = False
  if do_short:
    idstrs = idstrs[:10]

  n_idstr = len(idstrs)
  
  # build the valid and train dataset lists
  for idx in range(n_idstr):
    if idx % val_split == 0:
      idstr_valid.append(idstrs[idx])
    else:
      idstr_train.append(idstrs[idx])

  n_idstr_train = len(idstr_train)
  n_idstr_valid = len(idstr_valid)

  # Get test IDs from directory names
  test_path = arg_config.cfg['test_path']
  idstrs = next(os.walk(test_path))[1]
  
  do_short = False
  if do_short:
    idstrs = idstrs[:10]

  n_idstr = len(idstrs)
  
  # build the valid and train dataset lists
  for idx in range(n_idstr):
    idstr_test.append(idstrs[idx])

  n_idstr_test = len(idstr_test)
  
  # print stats
  print('load_data()...')
  print("n_idstr_train / val / test",n_idstr_train,n_idstr_valid,n_idstr_test)
  print()

def get_XY_from_idstr(idstr):

  train_path = arg_config.cfg['train_path']
  path_X = os.path.join(train_path,idstr,'images' , idstr + '.png')
  path_Y = os.path.join(train_path,idstr,'masks','all_' + idstr + '.png')
  img_X = imread(path_X)[:,:,:3]
  img_Y = imread(path_Y)
  
  return img_X,img_Y

def get_test_X_from_idstr(idstr):

  train_path = arg_config.cfg['test_path']
  path_X = os.path.join(train_path,idstr,'images' , idstr + '.png')
  img_X = imread(path_X)[:,:,:3]
  
  return img_X

# displays 2x2 grid of images and prints some stats
def display4img(img1,img2,img3,img4):
  
  print("img1.shape",img1.shape)
  print("img2.shape",img2.shape)
  print("img3.shape",img3.shape)
  print("img4.shape",img4.shape)

  img1 = np.squeeze(img1)
  img2 = np.squeeze(img2)
  img3 = np.squeeze(img3)
  img4 = np.squeeze(img4)
  
  print('img1',np.min(img1),np.max(img1))
  print('img2',np.min(img2),np.max(img2))
  print('img3',np.min(img3),np.max(img3))
  print('img4',np.min(img4),np.max(img4))

  # Get current size
  fig_size = plt.rcParams["figure.figsize"]
   
  # Prints: [8.0, 6.0]
  print("Current size:", fig_size)
   
  # Set figure width to 12 and height to 9
  fig_size[0] = 12
  fig_size[1] = 9
  plt.rcParams["figure.figsize"] = fig_size

  plt.subplot(221)
  plt.imshow(img1)
  plt.subplot(222)
  plt.imshow(img2)
  plt.subplot(223)
  plt.imshow(img3)
  plt.subplot(224)
  plt.imshow(img4)
  
  plt.show()

def crop_warp(img):
  
  if len(img.shape)  == 2:
    img = np.expand_dims(img, axis=-1)

  (h,w,_) = img.shape

  # width of the original center rounded to the nearest 16    
  # then add a border of 16 (32)

  original_crop = [0,0,w,h]

  img_wrap = image_wrapper.wrapify(img)
  original_crop[0] = w
  original_crop[1] = h
  
  crop_w = int(round(w/16))*16 + 64
  crop_h = int(round(h/16))*16 + 64
  crop_x = (3*w - crop_w) // 2
  crop_y = (3*h - crop_h) // 2

  img_crop = img_wrap[crop_y:crop_y+crop_h,
              crop_x:crop_x+crop_w,
              :]
    
  original_crop[0] = w - crop_x
  original_crop[1] = h - crop_y
  
  return img_crop,original_crop

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def restore_crop(img,original_crop):

  img_crop = img[original_crop[1]:original_crop[1]+original_crop[3],
              original_crop[0]:original_crop[0]+original_crop[2],
              :]
  
  return img_crop
    
if __name__ == "__main__":
 
  # always start with this 
  arg_config.arg_config(do_print=True)

  # load test data
  # but also some train data for double checking and validation
  load_test_data()

  # flags to say what data sets you want to look at
  do_valid = True
  do_test = False

  model_name = 'Unet32'

  model, weights_fname, submit_fname = our_models.get_model_data(model_name)

  print('model.load_weights("{}")'.format(weights_fname))
  model.load_weights(weights_fname)
  
  # validation with bells and whistles
  if do_valid:
    for idstr in idstr_valid:
      # get the validation X,Y_true
      img_X,Y_true = get_XY_from_idstr(idstr)
      
      # int to float conversion
      img_X = np.array(img_X) / 255.0
      Y_true = np.array(Y_true) / 255.0

      # get a version of X,Y with a 32 pixel border with mirrored edges
      crop_X,original_crop_X = crop_warp(img_X)
      crop_Y,original_crop_Y = crop_warp(Y_true)

      # check their dims
      print(crop_Y.shape)
      # the original_crop is used to find the image without the border
      print(original_crop_X)

      # grab a cropped version of the prediction
      # this should be identical to img_X
      org_X = restore_crop(crop_X,original_crop_X)
      
      # get ready to pass to the model
      x_batch = np.expand_dims(crop_X, axis=0)

      # predict!!!
      print(x_batch.shape)
      pred_Y = model.predict_on_batch(x_batch)

      # get rid of unwanted borders and dimentions
      pred_Y = np.squeeze(pred_Y,axis = 0)
      pred_Y = restore_crop(pred_Y,original_crop_X)
      pred_Y = np.squeeze(pred_Y)
      
      print(pred_Y.shape)
      print(Y_true.shape)
      # get the iou_metric with extra plots and text
      mean_iou = metrics.iou_metric(Y_true,
                                    pred_Y,
                                    print_table=True,
                                    do_plots=True)
      
      # lets compare what when it to what came out!!!
      bw_Y = np.where(pred_Y>0.5,1.0,0.0)
      display4img(img_X,Y_true,pred_Y,bw_Y)
    
  # submission time
  if do_test:
    new_test_ids = []
    rles = []
    
    for index,idstr in enumerate(idstr_test):
      
      # grab the test image and prepare it for prediction
      img_X = get_test_X_from_idstr(idstr)
      img_X = np.array(img_X) / 255.0
      crop_X,original_crop_X = crop_warp(img_X)
      print(original_crop_X)
      org_X = restore_crop(crop_X,original_crop_X)
      x_batch = np.expand_dims(crop_X, axis=0)

      # predict !!!
      pred_Y = model.predict_on_batch(x_batch)

      # get rid of unwanted borders and dimentions
      pred_Y = np.squeeze(pred_Y,axis = 0)
      pred_Y = restore_crop(pred_Y,original_crop_X)
      pred_Y = np.squeeze(pred_Y)

      # greyscale -> B&W      
      pred_Y = pred_Y[pred_Y > 0.5]
      # split the prediction into individual blobs
      lab_img = label(pred_Y)

      # run length encode the blobs
      rle = []
      for i in range(1, lab_img.max() + 1):
        rle.append(rle_encoding(lab_img == i))

      # add that to the BIG list of ALL blobs in ALL images
      rles.extend(rle)
      new_test_ids.extend([idstr] * len(rle))

      # tell the user how we are getting on      
      if index % 10 == 0:
        print(index,n_idstr_test)
      
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('submit_fname', index=False)
  
  print("Done!")
