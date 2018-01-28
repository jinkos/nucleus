from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout

# two convolutional layers with some dropout
def drop_conv(x,size,drop):

  x = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
  x = Dropout(drop)(x)
  x = Conv2D(size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
  return x

# two convolutional layers with some dropout
# followed by a maxpool
def drop_pool(x,size,drop):

  c = drop_conv(x, size, drop)
  p = MaxPooling2D((2, 2))(c)
  return c,p,size*2

# I have seen th following used instead of 'Conv2DTranspose' and 'concatenate'
# are they equivalent in Keras?
'''
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
'''
def drop_up(x,skip_add,size,drop):
  
  # upsample
  u = Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same')(x)
  # concat with skip layers
  u = concatenate([u, skip_add])
  # two convolutional layers with some dropout
  c = drop_conv(u,size,drop)
  
  return c, size//2

def Unet(rows,columns,size=16):
  
  inputs = Input((rows,columns,3))

  c1,p1,size = drop_pool(inputs, size, 0.1)
  c2,p2,size = drop_pool(p1, size, 0.1)
  c3,p3,size = drop_pool(p2, size, 0.2)
  c4,p4,size = drop_pool(p3, size, 0.2)
  c5 = drop_conv(p4, size, 0.3)

  size //= 2

  c6,size = drop_up(c5,c4,size,0.2)
  c7,size = drop_up(c6,c3,size,0.2)
  c8,size = drop_up(c7,c2,size,0.1)
  c9,size = drop_up(c8,c1,size,0.1)

  outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

  model = Model(inputs=[inputs], outputs=[outputs])

  return model
  