from __future__ import print_function

from skimage.io import imread, imshow
from skimage.transform import rotate
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
from tqdm import tqdm
import numpy as np
import sys
import os

import arg_config
import nuclear_gen

use_multiprocessing = True
n_processes = 4

### DEBUG FLAGS FOR augment(id_)
# set to true for lots of plots to see what's going on
do_show = False
# print lots of debugging data
do_data = False
# set this to False if you are experimenting and don't want to
# overwrite your previous images 
do_save = True
    

# copies a numpy image (src)
# to an x,y position of another (dst)
def np_blit(dst,x,y,src):
  
  (dh,dw,_) = dst.shape
  (sh,sw,_) = src.shape

  dx = x
  sx = 0
  w = sw
  
  # clip left
  if dx < 0:
    w += dx
    sx -= dx
    dx = 0
    
  # clip right
  if x + w > dx + dw:
    w = dx + dw - x
  
  
  dy = y
  sy = 0
  h = sh

  # clip right
  if dy < 0:
    h += dy
    sy -= dy
    dy = 0
    
  # clip top
  if y + h > dy + dh:
    h = dy + dh - y

  if w <= 0 or h <= 0: return -1
  
  dst[dy:dy+h,dx:dx+w,:] = src[sy:sy+h,sx:sx+w,:]
  
  return 0

# takes an input image (src)
# returns a larger image mage of 9 sub-images
# central image is the src
# the other eight mirror the central image
def wrapify(src):

  (h,w,c) = src.shape

  # will return this
  img_wrap = np.empty((h*3,w*3,c),dtype=src.dtype)

  # center
  np_blit(img_wrap, w, h, src)
  # center top and bottom
  img_ud = np.flipud(src)
  np_blit(img_wrap, w, 0, img_ud)
  np_blit(img_wrap, w, 2*h, img_ud)
  # center left and right
  img_lr = np.fliplr(src)
  np_blit(img_wrap, 0, h, img_lr)
  np_blit(img_wrap, 2*w, h, img_lr)
  # corners
  im_lrud = np.fliplr(img_ud)
  np_blit(img_wrap, 0, 0, im_lrud)
  np_blit(img_wrap, 2*w, 0, im_lrud)
  np_blit(img_wrap, 0, 2*h, im_lrud)
  np_blit(img_wrap, 2*w, 2*h, im_lrud)

  return img_wrap

def imsave(fname,img):

  pil_img = Image.fromarray(img)
  pil_img.save(fname)

# creates aumented files for a given idstr
def augment(multi_arg):
  
  idstr,train_path,n = multi_arg

  if (use_multiprocessing):
    print("doing {} : {}".format(n,idstr))
  
  # load png image from 'image' folder
  path = train_path + idstr
  img = imread(path + '/images/' + idstr + '.png')[:,:,:3]
  (h,w,_) = img.shape

  # U-Net will need images whose dimentions are divisible by 16
  # otherwise skip-connections don't line up
  # also we want a 32 pixel border around the original image position
  # save these crop sizes for later
  w_crop = int(round(w/16))*16 + 64
  h_crop = int(round(h/16))*16 + 64
  
  # create the wrapped image
  img_wrap = wrapify(img)
  
  if do_data:
    print('img',np.min(img),np.max(img))
    print('img_wrap',np.min(img_wrap),np.max(img_wrap))
  
  if do_show:
    imshow(img)
    plt.show()
    imshow(img_wrap)
    plt.show()
  
  if do_save:
    # try each 15 degrees of rotation
    for rotation_angle in range(0,360,15):
      # this is SLOW!!!
      # preserve_range stops it from expanding the rand of the colors
      # order=3 is something like bi-cubic color fitting
      img_rot = rotate(img_wrap, rotation_angle, order=3,preserve_range=True)
      if do_data:
        print('img_rot',np.min(img_rot),np.max(img_rot))
        
      # shear the original rotated image
      img_shear,_ = nuclear_gen.do_shear(img_rot, [], -0.2, True)
      if do_data:
        print('img_shear',np.min(img_shear),np.max(img_shear))

      # crop the middle with a 32 pixel border
      img_crop,_ = nuclear_gen.do_random_crop(img_shear, [], w_crop, h_crop, 0)
      if do_data:
        print('img_crop',np.min(img_crop),np.max(img_crop))

      # make sure it's a 24-bit image
      img_crop = np.array(img_crop,dtype=np.uint8)
      if do_data:
        print('img_crop',np.min(img_crop),np.max(img_crop))

      # save it
      imsave(path + '/images/aug_' + str(rotation_angle) + "_shu_" + idstr + '.png',img_crop)

      # repeat shear-crop-save with a different shear
      img_shear,_ = nuclear_gen.do_shear(img_rot, [], 0.2, True)
      img_crop,_ = nuclear_gen.do_random_crop(img_shear, [], w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/images/aug_' + str(rotation_angle) + "_shd_" + idstr + '.png',img_crop)

      # repeat shear-crop-save with a different shear
      img_shear,_ = nuclear_gen.do_shear(img_rot, [], 0.2, False)
      img_crop,_ = nuclear_gen.do_random_crop(img_shear, [], w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/images/aug_' + str(rotation_angle) + "_shl_" + idstr + '.png',img_crop)

      # repeat shear-crop-save with a different shear
      img_shear,_ = nuclear_gen.do_shear(img_rot, [], -0.2, False)
      img_crop,_ = nuclear_gen.do_random_crop(img_shear, [], w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/images/aug_' + str(rotation_angle) + "_shr_" + idstr + '.png',img_crop)

      # repeat with NO shear
      img_crop,_ = nuclear_gen.do_random_crop(img_rot, [], w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/images/aug_' + str(rotation_angle) + "_sh_" + idstr + '.png',img_crop)

  # create single mask from all mask files
  img = np.zeros((h, w, 1), dtype=np.bool)
  for mask_file in next(os.walk(path + '/masks/'))[2]:
      # ignore file name of already augmented masks
      # a bit UGLY - sorry
      if mask_file[:5] == 'wrap_': continue 
      if mask_file[:4] == 'all_': continue
      if mask_file[:4] == 'aug_': continue
      mask_ = imread(path + '/masks/' + mask_file)
      if do_data:
        print('Ymask_',np.min(mask_),np.max(mask_))
      mask_ = np.expand_dims(mask_, axis=-1)
      img = np.maximum(img, mask_)

  # create the wrapped image
  img_wrap = wrapify(img)
  if do_data:
    print('Yimg_wrap',np.min(img_wrap),np.max(img_wrap))

  # generally we like our greyscale images to have 0 channels
  # instead of 1 channel
  img_wrap = np.squeeze(img_wrap)
  img = np.squeeze(img)
  
  if do_show:
    imshow(img)
    plt.show()
    imshow(img_wrap)
    plt.show()

  # save the np.maximum(all masks)
  if do_save:
    imsave(path + '/masks/all_' + idstr + '.png',img)
    
    # this block of code is same as for color image input
    # but for B&W mask
    for rotation_angle in range(0,360,15):
      # order=0 is for nearest neighbour
      # we want B&W images not greyscale
      img_rot = rotate(img_wrap, rotation_angle, order=0,preserve_range=True)
      if do_data:
        print('Yimg_rot',np.min(img_rot),np.max(img_rot))
      _,img_shear = nuclear_gen.do_shear([], img_rot, -0.2, True)
      if do_data:
        print('Yimg_shear',np.min(img_shear),np.max(img_shear))
      _,img_crop = nuclear_gen.do_random_crop([], img_shear, w_crop, h_crop, 0)
      if do_data:
        print('Yimg_crop',np.min(img_crop),np.max(img_crop))
      img_crop = np.array(img_crop,dtype=np.uint8)
      if do_data:
        print('Yimg_crop',np.min(img_crop),np.max(img_crop))
      imsave(path + '/masks/aug_' + str(rotation_angle) + "_shu_" + idstr + '.png',img_crop)

      _,img_shear = nuclear_gen.do_shear([], img_rot, 0.2, True)
      _,img_crop = nuclear_gen.do_random_crop([], img_shear, w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/masks/aug_' + str(rotation_angle) + "_shd_" + idstr + '.png',img_crop)

      _,img_shear = nuclear_gen.do_shear([], img_rot, 0.2, False)
      _,img_crop = nuclear_gen.do_random_crop([], img_shear, w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/masks/aug_' + str(rotation_angle) + "_shl_" + idstr + '.png',img_crop)

      _,img_shear = nuclear_gen.do_shear([], img_rot, -0.2, False)
      _,img_crop = nuclear_gen.do_random_crop([], img_shear, w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/masks/aug_' + str(rotation_angle) + "_shr_" + idstr + '.png',img_crop)

      _,img_crop = nuclear_gen.do_random_crop([], img_rot, w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/masks/aug_' + str(rotation_angle) + "_sh_" + idstr + '.png',img_crop)

if __name__ == "__main__":

  # allways call this first
  arg_config.arg_config(do_print=True)

  train_path = arg_config.cfg['train_path']
  test_path = arg_config.cfg['test_path']

  # Get train and test IDs from directory names
  train_ids = next(os.walk(train_path))[1]
  test_ids = next(os.walk(test_path))[1]

  # radically shrinks the dataset to make the program quicker
  do_short = True 
  if do_short:
    train_ids = train_ids[:10]
    test_ids = test_ids[:10]

  # useful stats
  n_train_ids = len(train_ids)
  n_test_ids = len(test_ids)
  print("n_train_ids",n_train_ids)
  print("n_test_ids",n_test_ids)

  print("first few ids will be:")
  for train_id in train_ids[:5]:
    print(train_id)  

  ### DEAL WITH TRAINING SET
  if use_multiprocessing:
    train_paths = [train_path] * len(train_ids)
    counters = range(len(train_ids))
    p = multiprocessing.Pool(8)
    multi_args = zip(train_ids,train_paths,counters)
    p.map(augment, multi_args)
  else:
    # loop through the training images with tqdm
    sys.stdout.flush()
    for n, idstr in tqdm(enumerate(train_ids), total=n_train_ids):
      augment((idstr,train_path,n))
    
  ### DEAL WITH TEST SET
  # we're just going to save the cropped mirrored original
  # no augmentation
  print('Wrapify test images ... ')
  sys.stdout.flush()
  for n, idstr in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = test_path + idstr
    img = imread(path + '/images/' + idstr + '.png')[:,:,:3]
    (h,w,_) = img.shape
    w_crop = int(round(w/16))*16 + 64
    h_crop = int(round(h/16))*16 + 64
  
    img_wrap = wrapify(img)
  
    if do_show:
      imshow(img)
      plt.show()
      imshow(img_wrap)
      plt.show()
  
    if do_save:
      img_crop,_ = nuclear_gen.do_random_crop(img_wrap, [], w_crop, h_crop, 0)
      img_crop = np.array(img_crop,dtype=np.uint8)
      imsave(path + '/images/wrap_' + idstr + '.png',img_crop)
