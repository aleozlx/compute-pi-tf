import os, sys
import numpy as np
import tensorflow as tf

N_SAMPLES = 100000000
BATCH_SIZE = 512
STEPS = N_SAMPLES // BATCH_SIZE

tf.reset_default_graph()
darts = tf.random_uniform((BATCH_SIZE, 2))
counter = tf.reduce_sum(tf.cast(tf.less(tf.norm(darts, axis=1), 1), tf.int32))
pi_estimate = counter * 4 / BATCH_SIZE
pi_avg, pi_avg_update = tf.metrics.mean(pi_estimate)
init = tf.local_variables_initializer()

sess_config = tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)
with tf.Session(config=sess_config) as sess:
    sess.run(init)
    for _ in range(STEPS):
        sess.run([pi_estimate, pi_avg_update])
    print(sess.run(pi_avg))

