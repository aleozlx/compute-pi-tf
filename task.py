import os, sys, time

#[playbook(main)]
def main(ctx):
    import numpy as np
    import tensorflow as tf
    STEPS = ctx.num_samples // ctx.batch_size

    tf.reset_default_graph()
    with tf.device("gpu:0"):
        darts = tf.random_uniform((ctx.batch_size, 2))
        counter = tf.reduce_sum(tf.cast(tf.less(tf.norm(darts, axis=1), 1), tf.int32))
        pi_estimate = counter * 4 / ctx.batch_size
        pi_avg, pi_avg_update = tf.metrics.mean(pi_estimate)
        init = tf.local_variables_initializer()
    t0 = time.time()
    sess_config = tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)
    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        for _ in range(STEPS):
            sess.run([pi_estimate, pi_avg_update])
        print('Estimate of pi:', sess.run(pi_avg))
    print('Completed {:.2f} darts/s'.format(ctx.num_samples / (time.time()-t0)))

