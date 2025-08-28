import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from specgan import SpecGANGenerator, SpecGANDiscriminator

def main():
    z = tf.random_uniform([4, 100], -1., 1.)
    with tf.variable_scope('G'):
        G = SpecGANGenerator(z, train=False, kernel_len=5, dim=64, use_batchnorm=False, upsample='zeros')
    print('Generator tensor:', G)

    x_fake = tf.random_uniform([4, 128, 128, 1], -1., 1.)
    with tf.variable_scope('D'):
        D = SpecGANDiscriminator(x_fake, kernel_len=5, dim=64, use_batchnorm=False)
    print('Discriminator tensor:', D)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        G_out, D_out = sess.run([G, D])
        print('Generated batch shape:', G_out.shape)
        print('Discriminator output shape:', D_out.shape)

if __name__ == '__main__':
    main()
