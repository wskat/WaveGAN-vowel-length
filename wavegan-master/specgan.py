import tensorflow as tf_root
from tensorflow.keras import layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def conv2d_transpose(inputs, filters, kernel_len, stride=2, padding='same', upsample='zeros'):
  # Normalize shorthand upsample names
  if upsample == 'lin':
    upsample = 'linear'
  elif upsample == 'cub':
    upsample = 'cubic'
  if upsample == 'zeros':
    return layers.Conv2DTranspose(filters,
                                  kernel_size=kernel_len,
                                  strides=stride,
                                  padding=padding,
                                  use_bias=True)(inputs)
  elif upsample in ['nn', 'linear', 'cubic']:
    shape = inputs.get_shape().as_list()
    h, w = shape[1], shape[2]
    if h is None or w is None:
      dyn = tf.shape(inputs)
      h = dyn[1]
      w = dyn[2]
    if upsample == 'nn':
      method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif upsample == 'linear':
      method = tf.image.ResizeMethod.BILINEAR
    else:
      method = tf.image.ResizeMethod.BICUBIC
    x = tf.image.resize(inputs, [h * stride, w * stride], method=method, align_corners=False)
    return layers.Conv2D(filters,
                         kernel_size=kernel_len,
                         strides=1,
                         padding=padding,
                         use_bias=True)(x)
  else:
    raise NotImplementedError('Unknown upsample mode {}'.format(upsample))


def SpecGANGenerator(z,
                     kernel_len=5,
                     dim=64,
                     use_batchnorm=False,
                     upsample='zeros',
                     train=False):
  batch_size = tf.shape(z)[0]
  def batchnorm(x, scope):
    if not use_batchnorm:
      return x
    return layers.BatchNormalization(name=scope)(x, training=train)

  output = z
  with tf.variable_scope('z_project'):
    output = layers.Dense(4 * 4 * dim * 16, name='fc')(output)
    output = tf.reshape(output, [batch_size, 4, 4, dim * 16])
    output = batchnorm(output, 'bn')
  output = tf.nn.relu(output)

  with tf.variable_scope('upconv_0'):
    output = conv2d_transpose(output, dim * 8, kernel_len, 2, upsample=upsample)
    output = batchnorm(output, 'bn')
  output = tf.nn.relu(output)

  with tf.variable_scope('upconv_1'):
    output = conv2d_transpose(output, dim * 4, kernel_len, 2, upsample=upsample)
    output = batchnorm(output, 'bn')
  output = tf.nn.relu(output)

  with tf.variable_scope('upconv_2'):
    output = conv2d_transpose(output, dim * 2, kernel_len, 2, upsample=upsample)
    output = batchnorm(output, 'bn')
  output = tf.nn.relu(output)

  with tf.variable_scope('upconv_3'):
    output = conv2d_transpose(output, dim, kernel_len, 2, upsample=upsample)
    output = batchnorm(output, 'bn')
  output = tf.nn.relu(output)

  with tf.variable_scope('upconv_4'):
    output = conv2d_transpose(output, 1, kernel_len, 2, upsample=upsample)
  output = tf.nn.tanh(output)

  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    if update_ops:
      with tf.control_dependencies(update_ops):
        output = tf.identity(output)

  return output


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def SpecGANDiscriminator(x,
                         kernel_len=5,
                         dim=64,
                         use_batchnorm=False):
  batch_size = tf.shape(x)[0]
  def batchnorm(t, scope):
    if not use_batchnorm:
      return t
    return layers.BatchNormalization(name=scope)(t, training=True)

  output = x
  with tf.variable_scope('downconv_0'):
    output = layers.Conv2D(dim, kernel_size=kernel_len, strides=2, padding='same', name='conv')(output)
  output = lrelu(output)

  with tf.variable_scope('downconv_1'):
    output = layers.Conv2D(dim * 2, kernel_size=kernel_len, strides=2, padding='same', name='conv')(output)
    output = batchnorm(output, 'bn')
  output = lrelu(output)

  with tf.variable_scope('downconv_2'):
    output = layers.Conv2D(dim * 4, kernel_size=kernel_len, strides=2, padding='same', name='conv')(output)
    output = batchnorm(output, 'bn')
  output = lrelu(output)

  with tf.variable_scope('downconv_3'):
    output = layers.Conv2D(dim * 8, kernel_size=kernel_len, strides=2, padding='same', name='conv')(output)
    output = batchnorm(output, 'bn')
  output = lrelu(output)

  with tf.variable_scope('downconv_4'):
    output = layers.Conv2D(dim * 16, kernel_size=kernel_len, strides=2, padding='same', name='conv')(output)
    output = batchnorm(output, 'bn')
  output = lrelu(output)

  output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])
  with tf.variable_scope('output'):
    output = layers.Dense(1, name='fc')(output)[:, 0]
  return output
