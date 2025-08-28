"""Training entry point for fiwGAN (WaveGAN + structured latent codes).

This keeps original WaveGAN code untouched. You can run this script with arguments similar to
train_wavegan.py plus fiwGAN-specific flags.

Example:
  python fiwgan_train.py --data_dir ../code_of_miss/Cantonese --train_dir ./fiwgan_run \
      --fiwgan_code_names vowel,length --fiwgan_code_dims 10,2

Note: metadata integration (mapping real examples to codes) is deferred; codes are sampled.
Later we can incorporate supervised or semi-supervised signals using the metadata CSV.
"""
from __future__ import print_function

import argparse
import glob
import os
import tensorflow as tf

import loader
from fiwgan import CodeSpec, build_fiwgan_graph


def parse_code_specs(names_str, dims_str):
    if not names_str:
        return []
    names = [n.strip() for n in names_str.split(',') if n.strip()]
    dims = [int(x) for x in dims_str.split(',') if x.strip()]
    if len(names) != len(dims):
        raise ValueError('fiwgan_code_names and fiwgan_code_dims must align in length')
    return [CodeSpec(n, d) for n, d in zip(names, dims)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train'], default='train')
    ap.add_argument('--train_dir', required=True)
    ap.add_argument('--data_dir', required=True)
    # Data args (subset)
    ap.add_argument('--data_sample_rate', type=int, default=16000)
    ap.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536], default=16384)
    ap.add_argument('--data_num_channels', type=int, default=1)
    ap.add_argument('--data_first_slice', action='store_true')
    ap.add_argument('--data_overlap_ratio', type=float, default=0.)
    ap.add_argument('--data_pad_end', action='store_true')
    ap.add_argument('--data_fast_wav', action='store_true')
    ap.add_argument('--data_prefetch_gpu_num', type=int, default=0)
    # WaveGAN base
    ap.add_argument('--wavegan_kernel_len', type=int, default=25)
    ap.add_argument('--wavegan_dim', type=int, default=64)
    ap.add_argument('--wavegan_batchnorm', action='store_true')
    ap.add_argument('--wavegan_disc_phaseshuffle', type=int, default=2)
    # Training
    ap.add_argument('--train_batch_size', type=int, default=32)
    ap.add_argument('--train_save_secs', type=int, default=300)
    ap.add_argument('--train_summary_secs', type=int, default=120)
    # fiwGAN code specs
    ap.add_argument('--fiwgan_z_dim', type=int, default=64)
    ap.add_argument('--fiwgan_code_names', type=str, default='length')
    ap.add_argument('--fiwgan_code_dims', type=str, default='2')
    ap.add_argument('--fiwgan_mi_weight', type=float, default=1.0)

    args = ap.parse_args()

    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    # Collect files
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    if len(fps) == 0:
        raise RuntimeError('No audio files found in data_dir')
    print(f'Found {len(fps)} audio files')

    # Data pipeline
    with tf.name_scope('loader'):
        x = loader.decode_extract_and_batch(
            fps,
            batch_size=args.train_batch_size,
            slice_len=args.data_slice_len,
            decode_fs=args.data_sample_rate,
            decode_num_channels=args.data_num_channels,
            decode_fast_wav=args.data_fast_wav,
            decode_parallel_calls=4,
            slice_randomize_offset=False if args.data_first_slice else True,
            slice_first_only=args.data_first_slice,
            slice_overlap_ratio=0. if args.data_first_slice else args.data_overlap_ratio,
            slice_pad_end=True if args.data_first_slice else args.data_pad_end,
            repeat=True,
            shuffle=True,
            shuffle_buffer_size=4096,
            prefetch_size=args.train_batch_size * 4,
            prefetch_gpu_num=args.data_prefetch_gpu_num)[:, :, 0]

    # Set kwargs matching original wavegan
    setattr(args, 'wavegan_g_kwargs', {
        'slice_len': args.data_slice_len,
        'nch': args.data_num_channels,
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'upsample': 'zeros'
    })
    setattr(args, 'wavegan_d_kwargs', {
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
    })

    code_specs = parse_code_specs(args.fiwgan_code_names, args.fiwgan_code_dims)
    print('Code specs:', [(c.name, c.dim) for c in code_specs])

    graph_objs = build_fiwgan_graph(x, args, code_specs)

    # Optimizers
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
    # Q heads live under D/q_heads so include them in separate var list
    Q_vars = [v for v in D_vars if 'q_heads/' in v.name]
    D_main_vars = [v for v in D_vars if 'q_heads/' not in v.name]

    G_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    D_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    Q_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)

    global_step = tf.train.get_or_create_global_step()

    G_train_op = G_opt.minimize(graph_objs['G_loss'], var_list=G_vars, global_step=global_step)
    D_train_op = D_opt.minimize(graph_objs['D_loss'], var_list=D_main_vars)
    Q_train_op = Q_opt.minimize(graph_objs['Q_loss'], var_list=Q_vars)

    summary_op = graph_objs['summary_op']
    summary_writer = tf.summary.FileWriter(args.train_dir)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.train_dir, save_checkpoint_secs=args.train_save_secs) as sess:
        print('fiwGAN training started')
        step = 0
        while True:
            # One D + Q update, then G update
            sess.run([D_train_op, Q_train_op])
            sess.run(G_train_op)
            step += 1
            if step % 50 == 0:
                summ, g_loss, d_loss = sess.run([summary_op, graph_objs['G_loss'], graph_objs['D_loss']])
                summary_writer.add_summary(summ, step)
                print(f'step {step}  G_loss={g_loss:.4f}  D_loss={d_loss:.4f}')


if __name__ == '__main__':
    main()
