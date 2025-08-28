"""fiwGAN extensions: structured latent codes + mutual information head (InfoGAN style).

Initial minimal implementation that reuses WaveGANGenerator and implements a discriminator
with auxiliary Q heads for categorical codes. Keeps original WaveGAN design unchanged.

Assumptions:
  - All structured codes are categorical (including binary length encoded as 2-way softmax)
  - Provided as one-hot vectors; concatenated with noise z before generator
  - Code specification is a list of (name, dim) pairs

Provides convenience functions to sample codes and build losses.
"""
from __future__ import annotations

import tensorflow as tf
import collections
from wavegan import WaveGANGenerator as BaseGenerator, lrelu, apply_phaseshuffle


CodeSpec = collections.namedtuple('CodeSpec', ['name', 'dim'])


def sample_codes(batch_size, code_specs):
    """Sample list of one-hot codes; returns concatenated code tensor and dict of tensors."""
    codes = []
    code_tensors = {}
    for spec in code_specs:
        logits = tf.random_uniform([batch_size, spec.dim])  # random uniform then softmax to one-hot via argmax trick
        one_hot = tf.one_hot(tf.argmax(logits, axis=1), spec.dim)
        codes.append(one_hot)
        code_tensors[spec.name] = one_hot
    c = tf.concat(codes, axis=1, name='c_concat') if codes else tf.zeros([batch_size, 0])
    return c, code_tensors


def FiwGANGenerator(z_plus_c, **g_kwargs):
    return BaseGenerator(z_plus_c, **g_kwargs)


def FiwGANDiscriminatorQ(x, code_specs, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
    """Discriminator + Q heads.

    Returns:
      D_logit: adversarial scalar logit
      Q_logits: dict name->logits tensor
    """
    batch_size = tf.shape(x)[0]
    slice_len = int(x.get_shape()[1])

    if use_batchnorm:
        batchnorm = lambda t: tf.layers.batch_normalization(t, training=True)
    else:
        batchnorm = lambda t: t

    if phaseshuffle_rad > 0:
        phaseshuffle = lambda t: apply_phaseshuffle(t, phaseshuffle_rad)
    else:
        phaseshuffle = lambda t: t

    output = x
    with tf.variable_scope('downconv_0'):
        output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
    output = lrelu(output)
    output = phaseshuffle(output)

    with tf.variable_scope('downconv_1'):
        output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output); output = phaseshuffle(output)

    with tf.variable_scope('downconv_2'):
        output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output); output = phaseshuffle(output)

    with tf.variable_scope('downconv_3'):
        output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output); output = phaseshuffle(output)

    with tf.variable_scope('downconv_4'):
        output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    if slice_len == 32768:
        with tf.variable_scope('downconv_5'):
            output = tf.layers.conv1d(output, dim * 32, kernel_len, 2, padding='SAME')
            output = batchnorm(output)
        output = lrelu(output)
    elif slice_len == 65536:
        with tf.variable_scope('downconv_5'):
            output = tf.layers.conv1d(output, dim * 32, kernel_len, 4, padding='SAME')
            output = batchnorm(output)
        output = lrelu(output)

    feat = tf.reshape(output, [batch_size, -1], name='flatten')
    with tf.variable_scope('output'):
        D_logit = tf.layers.dense(feat, 1)[:, 0]

    Q_logits = {}
    with tf.variable_scope('q_heads'):
        for spec in code_specs:
            Q_logits[spec.name] = tf.layers.dense(feat, spec.dim, name=f'logits_{spec.name}')

    return D_logit, Q_logits


def mutual_information_loss(code_specs, code_true, q_logits):
    """Cross-entropy for each categorical code.

    code_true: dict name->one-hot actual code
    q_logits: dict name->logits predicted
    Returns scalar loss and per-code accuracies dict.
    """
    losses = []
    accs = {}
    for spec in code_specs:
        logits = q_logits[spec.name]
        labels = code_true[spec.name]
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        losses.append(tf.reduce_mean(ce))
        preds = tf.argmax(logits, axis=1)
        gold = tf.argmax(labels, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, gold), tf.float32))
        accs[spec.name] = acc
    total = tf.add_n(losses, name='mi_loss') if losses else tf.constant(0., name='mi_loss')
    return total, accs


def build_fiwgan_graph(x, args, code_specs):
    """Construct fiwGAN graph (generator, discriminator, Q) using WGAN-GP + MI.

    Expects args to carry wavegan_g_kwargs, wavegan_d_kwargs, fiwgan parameters:
      fiwgan_z_dim, fiwgan_mi_weight
    """
    batch_size = tf.shape(x)[0]
    # Sample noise & codes
    z = tf.random_uniform([args.train_batch_size, args.fiwgan_z_dim], -1., 1.)
    c, code_true = sample_codes(args.train_batch_size, code_specs)
    zc = tf.concat([z, c], axis=1, name='z_plus_c')
    with tf.variable_scope('G'):
        G_z = FiwGANGenerator(zc, train=True, **args.wavegan_g_kwargs)
    with tf.name_scope('D_x'), tf.variable_scope('D'):
        D_x, _ = FiwGANDiscriminatorQ(x, code_specs, **args.wavegan_d_kwargs)
    with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
        D_G_z, q_logits = FiwGANDiscriminatorQ(G_z, code_specs, **args.wavegan_d_kwargs)

    # Adversarial losses (WGAN-GP)
    G_loss_adv = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
    # Gradient penalty
    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
    differences = G_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
        D_interp, _ = FiwGANDiscriminatorQ(interpolates, code_specs, **args.wavegan_d_kwargs)
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += 10 * gradient_penalty

    # Mutual information loss
    mi_loss, accs = mutual_information_loss(code_specs, code_true, q_logits)
    G_loss = G_loss_adv + args.fiwgan_mi_weight * mi_loss
    Q_loss = mi_loss  # optimize Q to minimize CE

    summaries = [
        tf.summary.scalar('G_loss_adv', G_loss_adv),
        tf.summary.scalar('D_loss', D_loss),
        tf.summary.scalar('gradient_penalty', gradient_penalty),
        tf.summary.scalar('MI_loss', mi_loss),
        tf.summary.scalar('G_loss_total', G_loss)
    ]
    for name, acc in accs.items():
        summaries.append(tf.summary.scalar(f'acc_{name}', acc))
    summary_op = tf.summary.merge(summaries, name='fiwgan_summaries')

    return {
        'G_z': G_z,
        'z': z,
        'c': c,
        'code_true': code_true,
        'D_x': D_x,
        'D_G_z': D_G_z,
        'G_loss': G_loss,
        'D_loss': D_loss,
        'Q_loss': Q_loss,
        'summary_op': summary_op,
        'accs': accs,
    }
