from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import keras
import time

import nuclear_gen
import arg_config
import our_models
import metrics

if __name__ == "__main__":
  
  # always do this first
  arg_config.arg_config(do_print=True)
  report_step = int(arg_config.cfg['report_step'])
  validation_step = int(arg_config.cfg['validation_step'])
  save_step = int(arg_config.cfg['save_step'])

  # which model?
  model_name = 'Unet32'
  # get the model data
  model, weights_fname, _ = our_models.get_model_data(model_name)
  # print the model
  model.summary(line_length=110)
  # set up the optimizer
  opt = keras.optimizers.sgd(0.001, momentum=0.99)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[metrics.my_iou_metric])
  
  # re-load weights if --load is set
  if arg_config.args.load:
    print('model.load_weights("{}")'.format(weights_fname))
    model.load_weights(weights_fname)
  
  # stats
  sample_counter = 0  
  total_loss = 0
  total_mean_iou = 0
  start_time = time.time()
  report_time = time.time()
  deep_time = time.time()
  deep_total = 0
  not_deep_total = 0
  
  # NUCLEAR GENERATOR!
  my_train_gen = nuclear_gen.gen_XY_aug(is_train=True)

  # more stats
  do_print_train = False
  total_train_mean = 0
  total_train_counter = 0
  is_first_train_loop = True

  # loop through the data
  for img_X,img_Y in my_train_gen:

    # stats
    not_deep_total += time.time() - deep_time
    deep_time = time.time()

    # TRAIN!!!
    losses = model.train_on_batch(img_X,img_Y)

    # stats
    deep_total += time.time() - deep_time
    deep_time = time.time()
    sample_counter += 1
    total_train_counter += 1
    total_loss += losses[0]
    total_mean_iou += losses[1]
    total_train_mean += losses[1]
    
    # print intermediate training stats before the first validation step
    # and thereafter only if do_print_train == True
    if sample_counter % report_step == 0:
      if do_print_train or is_first_train_loop:
        print("{:5d}s {:3d}e {:6.3f}l {:6.3f}iou {:3.0f}sps {:5.3f}".format(sample_counter,
                                                                    sample_counter // nuclear_gen.n_idstr_train,
                                                                    total_loss / report_step,
                                                                    total_mean_iou / report_step,
                                                                    report_step / (time.time()-report_time),
                                                                    deep_total / (deep_total + not_deep_total)))
        
      total_loss = 0
      total_mean_iou = 0
      report_time = time.time()
      deep_total = 0
      not_deep_total = 0

    # save weights occationally
    if arg_config.args.save and sample_counter % save_step == 0:
      print('model.save_weights("{}")'.format(weights_fname))
      model.save_weights(weights_fname)

    # validation time...
    if sample_counter % validation_step == 0:
      
      # get the training data generator
      my_valid_gen = nuclear_gen.gen_XY_aug(is_train=False)

      # stats
      total_mean_iou = 0
      is_first = True
      is_first_train_loop = False

      # loop through the validation set
      for img_X,true_Y in my_valid_gen:
        
        # PREDICTIONS
        pred_Y = model.predict_on_batch(img_X)

        # unwanted dimentions
        pred_Y = np.squeeze(pred_Y)
        true_Y = np.squeeze(true_Y)

        # sometimes I like to see exactly what is going on
        if is_first and arg_config.args.machine == "james_mac":
          print("img_X.shape",np.squeeze(img_X).shape)
          print("true_Y.shape",true_Y.shape)
          print("pred_Y.shape",pred_Y.shape)
          plt.ion()
          plt.subplot(221)
          plt.imshow(np.squeeze(img_X))
          plt.subplot(222)
          plt.imshow(true_Y)
          plt.subplot(223)
          plt.imshow(pred_Y)
          plt.show()
          plt.pause(0.001)
          is_first = False

        # Ah yes - those funny metrics        
        mean_iou = metrics.iou_metric(true_Y, pred_Y,print_table=False)
        total_mean_iou += mean_iou

      train_mean = total_train_mean / total_train_counter
      total_train_mean = 0
      total_train_counter = 0

      # lets print out how we have done      
      print("{:5d}s {:3d}e val:{:6.3f} train:{:6.3f}".format(sample_counter,
                                      sample_counter // nuclear_gen.n_idstr_train,
                                      total_mean_iou / nuclear_gen.n_idstr_valid,
                                      train_mean))
    deep_time = time.time()

  print('Done!')
