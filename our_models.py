import unet
import os

def get_model_data(model_name):

  if model_name == 'Unet16':
    model = unet.Unet(None,None,16)
  elif model_name == 'Unet32':
    model = unet.Unet(None, None,32)
  elif model_name == 'Unet64':
    model = unet.Unet(None,None,64)
  else:
    os.sys.exit("bollcocks")

  weights_fname = '{}.h5'.format(model_name)
  submit_fname = '{}.csv'.format(model_name)

  return model, weights_fname, submit_fname
