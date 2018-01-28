from __future__ import print_function

from skimage.transform import rotate
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import transform
import numpy as np
import random
import os

import arg_config

# for finding all the different types of shearing files
shear_tag_list = ['sh','shu','shd','shl','shr']
# val:test split is 1:5
val_split = 6

# valid and train dataset lists
idstr_train = []
idstr_valid = []
n_idstr_train = 0
n_idstr_valid = 0

# sets up idstr_train, idstr_valid, n_idstr_train and n_idstr_valid
# idstr_valid is a list of id strings
# idstr_train is a list of different combinations of aumentation
#   which are used to construct the corresponding file name
#   and other proprocessing ops like flipping
def load_data():

  global n_idstr_train
  global n_idstr_valid
  global val_split
  global idstr_train
  global idstr_valid
  
  # only load once
  if n_idstr_train > 0: return
  
  # Get train ID Strings from directory names
  train_path = arg_config.cfg['train_path']
  idstrs = next(os.walk(train_path))[1]
  
  # remove duff training examples
  if '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80' in idstrs:
    print('ignoring 7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
    idstrs.remove('7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
  
  do_short = False
  # short version on mac (always)
  if arg_config.args.machine == 'james_mac':
    do_short = True
  if do_short:
    idstrs = idstrs[:10]

  # stats
  n_idstr = len(idstrs)

  # build the valid and train dataset lists
  for idx in range(n_idstr):
    if idx % val_split == 0:
      # for the valid set we don't want augmented data
      idstr_valid.append(idstrs[idx])
    else:
      # we want all the angles x24
      for angle in range(0,360,15):
        # all the shears x5
        for shear in shear_tag_list:
          # and all filps x2
          for flip in range(2):
            idstr_train.append((idstrs[idx],angle,shear,flip))

  n_idstr_train = len(idstr_train)
  n_idstr_valid = len(idstr_valid)
  
  # shuffle the valid and train dataset lists
  random.Random(2018).shuffle(idstr_train)
  random.Random(2018).shuffle(idstr_valid)
  # print stats
  print('load_data()...')
  print("n_train_ids / train / val",n_idstr,n_idstr_train,n_idstr_valid)
  print()

# returns X (3 channel input image), y (truth mask) and whether the
# image should be flipped
def gen_XY(is_train=True):
  global idstr_train
  global idstr_valid
  
  load_data()

  train_path = arg_config.cfg['train_path']
  
  if not is_train:
    # VALIDATIONS SET
    for idstr in idstr_valid:
      
      path_X = os.path.join(train_path,idstr,'images','aug_0_sh_' + idstr + '.png')
      path_Y = os.path.join(train_path,idstr,'masks','aug_0_sh_' + idstr + '.png')
      img_X = imread(path_X)[:,:,:3]
      img_Y = imread(path_Y)
      
      # convert uint8 to float64
      img_X = np.array(img_X) / 255.0
      img_Y = np.array(img_Y) / 255.0

      (xh,xw,_) = img_X.shape
      (yh,yw) = img_Y.shape
      
      assert yh == xh, "mask is not same shape as image"
      assert yw == xw, "mask is not same shape as image"
      
      # last field is about flipping - ignore
      yield img_X,img_Y,-1
    return
  
  while(True):
    # TRAINING SET
    for idstr,angle,shear,do_flip, in idstr_train:
      path_X = os.path.join(train_path,idstr,
                            'images',
                            'aug_' + str(angle) + '_' + shear + '_' + idstr + '.png')
      path_Y = os.path.join(train_path,idstr,
                            'masks',
                            'aug_' + str(angle) + '_' + shear + '_' + idstr + '.png')
      img_X = imread(path_X)[:,:,:3]
      img_Y = imread(path_Y)
      
      img_X = np.array(img_X) / 255.0
      img_Y = np.array(img_Y) / 255.0

      (xh,xw,_) = img_X.shape
      (yh,yw) = img_Y.shape
      
      # convert uint8 to float64
      assert yh == xh, "mask is not same shape as image"
      assert yw == xw, "mask is not same shape as image"
      
      yield img_X,img_Y,do_flip

# takes two images (X,Y) and crops them in the middle
# the middle is shifted randomly according to 'shift'
# if one of the images is empty then it is ignored
def do_random_crop(img_X,img_Y,crop_w,crop_h,shift):

  if len(img_X) > 0:
    h,w = img_X.shape[:2]
  else:
    h,w = img_Y.shape[:2]

  crop_x = (w - crop_w) // 2
  crop_y = (h - crop_h) // 2

  crop_x += random.randint(-shift,shift)
  crop_y += random.randint(-shift,shift)

  if len(img_X) > 0:
    img_X = img_X[crop_y:crop_y+crop_h,
                crop_x:crop_x+crop_w,
                :]
  if len(img_Y) > 0:
    img_Y = img_Y[crop_y:crop_y+crop_h,
                crop_x:crop_x+crop_w]

  return img_X, img_Y

# rotates X & Y by the same angle
def do_rotate(img_X,img_Y,rotation_angle):

    img_X = rotate(img_X, rotation_angle, order=3)
    img_Y = rotate(img_Y, rotation_angle, order=0)
    
    return img_X,img_Y

# this shears the X & Y around the central point
# if one of the images is empty then it is ignored
def do_shear(img_X,img_Y,sh,horz):

  if len(img_X) > 0:
    h,w = img_X.shape[:2]
  else:
    h,w = img_Y.shape[:2]

  matrix = np.zeros((3,3))
  matrix[0,0] = 1.0
  matrix[1,1] = 1.0
  matrix[2,2] = 1.0

  if horz:
    matrix[0,1] = sh
    matrix[0,2] = -np.sin(sh)* w / 2
  else:
    matrix[1,0] = sh
    matrix[1,2] = -np.sin(sh)* h / 2

  # Create Afine transform
  afine_tf = transform.AffineTransform(matrix)
  #print(afine_tf.params)
  # Apply transform to image data
  if len(img_X) > 0:
    img_X = transform.warp(img_X, inverse_map=afine_tf,mode='edge', order=3)
  if len(img_Y) > 0:
    img_Y = transform.warp(img_Y, inverse_map=afine_tf,mode='edge', order=0)

  return img_X, img_Y

# this returns X&Y after doing some any necesarry preprocessing
def gen_XY_aug(is_train=True):

  XY_gen = gen_XY(is_train=is_train)
  
  for img_X,img_Y,do_flip in XY_gen:
    
    (h,w,_) = img_X.shape  

    # width of the original center rounded to the nearest 16    
    # then add a border of 16 (32)
    # we like a bit of random jitter of the traing data
    # consequently we can't use the whole 32 pixel border
    # contained in the augmented images
    # we can only have a SAFE border of 16 pixels
    w_crop = int(round((w-64)/16))*16 + 32
    h_crop = int(round((h-64)/16))*16 + 32
    
    if is_train:
      # we like a bit of random jitter of the traing data
      aug_X, aug_Y = do_random_crop(img_X,img_Y,w_crop,h_crop,16)  

      # flipping of requested    
      if do_flip:
          img_X = np.fliplr(img_X)
          img_Y = np.fliplr(img_Y)
    else:
      # return cropped images with no jitter or augmentation
      aug_X,aug_Y = do_random_crop(img_X,img_Y,w_crop,h_crop,0)  

    # get the dimentions ready for the NN
    aug_X = np.expand_dims(aug_X, axis=0)
    aug_Y = np.expand_dims(aug_Y, axis=0)
    aug_Y = np.expand_dims(aug_Y, axis=-1)
    
    aug_X = np.array(aug_X,dtype=np.float32)
    aug_Y = np.array(aug_Y,dtype=np.float32)
            
    yield aug_X,aug_Y
  
  
if __name__ == "__main__":

  arg_config.arg_config(do_print=True)

  XY_gen = gen_XY(is_train=True)
  
  for img_X,img_Y,do_flip in XY_gen:
    
    (h,w,_) = img_X.shape  
    
    # width of the original center rounded to the nearest 16    
    # then add a border of 16 (32)
    w_crop = int(round((w-64)/16))*16 + 32
    h_crop = int(round((h-64)/16))*16 + 32

    aug_X, aug_Y = do_random_crop(img_X,img_Y,w_crop,h_crop,16)  
  
    if do_flip:
        img_X = np.fliplr(img_X)
        img_Y = np.fliplr(img_Y)

    img_X,img_Y = do_random_crop(img_X,img_Y,w_crop,h_crop,0)  

    print('img_X',np.min(img_X),np.max(img_X))
    print('img_Y',np.min(img_Y),np.max(img_Y))
    print('aug_X',np.min(aug_X),np.max(aug_X))
    print('aug_Y',np.min(aug_Y),np.max(aug_Y))

    plt.subplot(221)
    plt.imshow(img_X)
    plt.subplot(222)
    plt.imshow(img_Y)
    
    plt.subplot(223)
    plt.imshow(aug_X)
    plt.subplot(224)
    plt.imshow(aug_Y)
    plt.show()
    