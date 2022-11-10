from collections import Counter, namedtuple
import numpy as np
import joblib
import os
import time
import random

from common_layers import *
from data_loaders import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Disables AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_preprocessed_data(data_path, dataset='Ballroom', valid_fold=0, symbolic=False):
    print('get preprocessed data...')
    global hp
    if dataset == 'Ballroom':
        hp = hp._replace(sequence_length=1700, batch_size=64)
    elif dataset == 'Hainsworth':
        hp = hp._replace(sequence_length=4096, batch_size=40)
    elif dataset == 'GTZAN':
        hp = hp._replace(sequence_length=1600, batch_size=100)
    elif dataset == 'ASAP':
        hp = hp._replace(sequence_length=2048, batch_size=80)
    else:
        print('Invalid dataset.')
        exit(1)

    if dataset in ['Ballroom', 'Hainsworth', 'GTZAN']:
        with open(data_path + '/train_dict_fold_' + str(valid_fold) + '.pickle', 'rb') as f:
            train_dict = joblib.load(f)
        with open(data_path + '/valid_dict_fold_' + str(valid_fold) + '.pickle', 'rb') as f:
            valid_dict = joblib.load(f)
    elif dataset == 'ASAP':
        filename = '_symbolic.pickle' if symbolic else '_audio.pickle'
        with open(data_path + '/train_dict' + filename, 'rb') as f:
            train_dict = joblib.load(f)
        with open(data_path + '/test_dict' + filename, 'rb') as f:
            valid_dict = joblib.load(f)
    return train_dict, valid_dict


def train_audio_beat_downbeat_estimation_baseline():

    dataset = 'Ballroom' # ['Ballroom', 'Hainsworth', 'GTZAN', 'ASAP']
    valid_fold = 0
    data_path = './ballroom_preprocessed_data_reshape'
    graph_location = './model'
    train_dict, valid_dict = get_preprocessed_data(data_path, dataset=dataset, valid_fold=valid_fold)

    n_train_samples = train_dict['spec'].shape[0]
    n_valid_samples = valid_dict['spec'].shape[0]
    n_iterations_per_epoch = math.ceil(n_train_samples / hp.batch_size)
    print('n_train_samples =', n_train_samples)
    print('n_valid_samples =', n_valid_samples)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp)

    with tf.name_scope('placeholder'):
        x_spec = tf.placeholder(tf.float32, [None, hp.sequence_length, 81, 2], name='spec')
        x_len = tf.placeholder(tf.int32, [None], name='valid_lengths')
        y_db = tf.placeholder(tf.int32, [None, hp.sequence_length], name='downbeat')
        y_b = tf.placeholder(tf.int32, [None, hp.sequence_length], name='beat')
        y_hop = tf.placeholder(tf.int32, [None], name='hop_size')
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        P_db, R_db, F1_db = tf.placeholder(dtype=tf.float32, name='P_db'), tf.placeholder(dtype=tf.float32, name='R_db'), tf.placeholder(dtype=tf.float32, name='F1_db')
        P_b, R_b, F1_b = tf.placeholder(dtype=tf.float32, name='P_b'), tf.placeholder(dtype=tf.float32, name='R_b'), tf.placeholder(dtype=tf.float32, name='F1_b')

    with tf.name_scope('model'):
        seq_mask_bool = tf.sequence_mask(x_len, hp.sequence_length, dtype=tf.bool) # Sequence mask
        seq_mask_float = tf.cast(seq_mask_bool, tf.float32)
        expanding_sizes = 3087 // y_hop # 3087 = 0.07*44100
        y_db_half = label_expanding(y_db, expanding_sizes, epsilon=0.5, mask=seq_mask_float)
        y_b_half = label_expanding(y_b, expanding_sizes, epsilon=0.5, mask=seq_mask_float)
        logits_db, logits_b = audio_joint_beat_downbeat_estimation_baseline(x_spec, x_len, dropout, is_training, hp)

    with tf.variable_scope('loss'):
        n_valid = tf.reduce_sum(tf.cast(seq_mask_bool, tf.float32))

        # Cross entropy
        ce_db = 2 * tf.losses.sigmoid_cross_entropy(y_db_half, logits_db, scope='ce_db')
        ce_b = tf.losses.sigmoid_cross_entropy(y_b_half, logits_b, scope='ce_b')

        # Total loss
        loss = ce_db + ce_b
    summary_loss = tf.Variable([0.0 for _ in range(3)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + n_valid * [loss, ce_db, ce_b])
    update_valid = tf.assign(summary_valid, summary_valid + n_valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('CE_db', summary_loss[1])
    tf.summary.scalar('CE_b', summary_loss[2])

    with tf.name_scope('evaluation'):
        # Downbeat
        prob_db = tf.sigmoid(logits_db) * seq_mask_float
        pred_db = tf.cast(tf.greater(prob_db, hp.threshold), tf.int32)
        prob_db_pool = output_probability_pooling(prob_db, pool_size=7) # pooling by local maximums
        pred_db_pool = tf.cast(tf.greater(prob_db_pool, hp.threshold), tf.int32)
        # Beat
        prob_b = tf.sigmoid(logits_b) * seq_mask_float
        pred_b = tf.cast(tf.greater(prob_b, hp.threshold), tf.int32)
        prob_b_pool = output_probability_pooling(prob_b, pool_size=7) # pooling by local maximums
        pred_b_pool = tf.cast(tf.greater(prob_b_pool, hp.threshold), tf.int32)
    tf.summary.scalar('Precision_db', P_db)
    tf.summary.scalar('Recall_db', R_db)
    tf.summary.scalar('F1_db', F1_db)
    tf.summary.scalar('Precision_b', P_b)
    tf.summary.scalar('Recall_b', R_b)
    tf.summary.scalar('F1_b', F1_b)

    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.hidden_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        train_op = optimizer.minimize(loss)

    # Graph location and summary writers
    print('Saving graph to: %s' % graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(graph_location + '/train')
    valid_writer = tf.summary.FileWriter(graph_location + '/valid')
    train_writer.add_graph(tf.get_default_graph())
    valid_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()

    # Training
    print('\nTrain the model...')
    np.set_printoptions(suppress=True, linewidth=180)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_scores = [0.0, 0.0]
        in_succession = 0
        best_epoch = 0
        indices = np.arange(n_train_samples)
        batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]
        step = 1
        for epoch in range(1, hp.n_training_epochs+1):
            # Training
            if epoch > 2:
                # Shuffle training data
                indices = np.array(random.sample(range(n_train_samples), n_train_samples))
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]
            # Batch-wise feeding
            train_pred_db_pool_list, train_pred_b_pool_list = [], []
            for i_b, batch_idx in enumerate(batch_indices):
                batch = {'spec': train_dict['spec'][batch_idx],
                         'len': train_dict['len'][batch_idx],
                         'downbeat': train_dict['downbeat'][batch_idx],
                         'beat': train_dict['beat'][batch_idx],
                         'id': train_dict['recovery_id'][batch_idx],
                         'hop': train_dict['hop'][batch_idx]}

                train_run_list = [train_op, update_valid, update_loss,
                                  loss, ce_db, ce_b,
                                  prob_db, prob_b, pred_db, pred_b, pred_db_pool, pred_b_pool,
                                  seq_mask_bool, y_db_half, y_b_half]
                train_feed_fict = {x_spec: batch['spec'],
                                   x_len: batch['len'],
                                   y_db: batch['downbeat'],
                                   y_b: batch['beat'],
                                   y_hop: batch['hop'],
                                   dropout: hp.drop,
                                   is_training: True,
                                   global_step: step}
                _, _, _, \
                train_loss, train_ce_db, train_ce_b, \
                train_prob_db, train_prob_b, train_pred_db, train_pred_b, train_pred_db_pool, train_pred_b_pool, \
                train_mask, train_y_db, train_y_b = sess.run(train_run_list, feed_dict=train_feed_fict)

                # Store batch-wise results
                train_pred_db_pool_list.append(train_pred_db_pool)
                train_pred_b_pool_list.append(train_pred_b_pool)
                # show the loss information
                if step == 1:
                    print('*~ total loss %.4f, ce(db %.4f, b %.4f) ~*' % (train_loss, train_ce_db, train_ce_b))
                step +=1

            # Concat stored results
            train_pred_db_pool_all = np.concatenate(train_pred_db_pool_list, axis=0)
            train_pred_b_pool_all = np.concatenate(train_pred_b_pool_list, axis=0)

            # Recovery ordering
            gather_id = [np.where(indices == ord)[0][0] for ord in range(n_train_samples)]
            train_pred_db_pool_all = train_pred_db_pool_all[gather_id, :]
            train_pred_b_pool_all = train_pred_b_pool_all[gather_id, :]

            # Evaluate training predictions
            # Downbeat
            score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(train_pred_db_pool_all, train_dict, f_measure_threshold=0.07, target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1, item = (key, scores), scores=(P, R, F1, len(pred), len(label))
            n_pred_beats_db, n_true_beats_db = sum([v[3] for v in score_dict_db_pool.values()]), sum([v[4] for v in score_dict_db_pool.values()])
            mean_P_db = sum([v[0] * v[3] for v in score_dict_db_pool.values()]) / n_pred_beats_db if n_pred_beats_db > 0 else 0
            mean_R_db = sum([v[1] * v[4] for v in score_dict_db_pool.values()]) / n_true_beats_db if n_true_beats_db > 0 else 0
            mean_F1_db = (2 * mean_P_db * mean_R_db) / (mean_P_db + mean_R_db) if (mean_P_db + mean_R_db) > 0 else 0
            # Beat
            score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(train_pred_b_pool_all, train_dict, f_measure_threshold=0.07, target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_b, n_true_beats_b = sum([v[3] for v in score_dict_b_pool.values()]), sum([v[4] for v in score_dict_b_pool.values()])
            mean_P_b = sum([v[0] * v[3] for v in score_dict_b_pool.values()]) / n_pred_beats_b if n_pred_beats_b > 0 else 0
            mean_R_b = sum([v[1] * v[4] for v in score_dict_b_pool.values()]) / n_true_beats_b if n_true_beats_b > 0 else 0
            mean_F1_b = (2 * mean_P_b * mean_R_b) / (mean_P_b + mean_R_b) if (mean_P_b + mean_R_b) > 0 else 0

            # Display training log
            sess.run([mean_loss])
            train_summary, train_loss = sess.run([merged, summary_loss],
                                                 feed_dict={P_db: mean_P_db, R_db: mean_R_db, F1_db: mean_F1_db,
                                                            P_b: mean_P_b, R_b: mean_R_b, F1_b: mean_F1_b})
            train_writer.add_summary(train_summary, epoch)
            sess.run([clr_summary_valid, clr_summary_loss])
            print("==== epoch %d: train_loss: total %.4f, db %.4f, b %.4f, Downbeat: P %.4f R %.4f F1 %.4f, Beat: P %.4f R %.4f F1 %.4f ===="
                % (epoch,
                   train_loss[0], train_loss[1], train_loss[2],
                   mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b))
            print('db_best =', score_sorted_db_pool[::-1][:3])
            print('db_worst =', score_sorted_db_pool[:3])
            print('b_best =', score_sorted_b_pool[::-1][:3])
            print('b_worst =', score_sorted_b_pool[:3])
            display_len = 200
            print('op =', batch['id'][0])
            print('len =', batch['len'][0])
            print('mask'.ljust(8), ''.join([' ' if x else 'x' for x in train_mask[0, :display_len]]))
            print('label_db'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in train_y_db[0, :display_len]]))
            print('label_b'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in train_y_b[0, :display_len]]))
            print('pred_db'.ljust(8), ''.join([str(x) for x in train_pred_db[0, :display_len]]))
            print('pred_b'.ljust(8), ''.join([str(x) for x in train_pred_b[0, :display_len]]))
            print('pred_db*'.ljust(8), ''.join([str(x) for x in train_pred_db_pool[0, :display_len]]))
            print('pred_b*'.ljust(8), ''.join([str(x) for x in train_pred_b_pool[0, :display_len]]))
            #
            print('prob_db'.ljust(8), [round(prob,1) for prob, db in zip(train_prob_db[0], batch['downbeat'][0]) if db == 1][:50])
            print('prob_b'.ljust(8), [round(prob,1) for prob, b in zip(train_prob_b[0], batch['beat'][0]) if b == 1][:50])

            # Validation
            valid_prob_db_list, valid_prob_b_list = [], []
            valid_pred_db_list, valid_pred_b_list = [], []
            valid_pred_db_pool_list, valid_pred_b_pool_list = [], []
            valid_mask_list = []
            valid_y_db_list, valid_y_b_list = [], []
            valid_batch_size = hp.batch_size
            for i in range(0, n_valid_samples, valid_batch_size):
                valid_run_list = [update_valid, update_loss,
                                  prob_db, prob_b, pred_db, pred_b, pred_db_pool, pred_b_pool,
                                  seq_mask_bool, y_db_half, y_b_half]
                valid_feed_fict = {x_spec: valid_dict['spec'][i:i+valid_batch_size],
                                   x_len: valid_dict['len'][i:i+valid_batch_size],
                                   y_db: valid_dict['downbeat'][i:i+valid_batch_size],
                                   y_b: valid_dict['beat'][i:i+valid_batch_size],
                                   y_hop: valid_dict['hop'][i:i + valid_batch_size],
                                   dropout: 0.0,
                                   is_training: False,
                                   global_step: step}
                _, _, valid_prob_db, valid_prob_b, valid_pred_db, valid_pred_b, valid_pred_db_pool, valid_pred_b_pool, \
                valid_mask, valid_y_db, valid_y_b = sess.run(valid_run_list, feed_dict=valid_feed_fict)
                # Store batch-wise results
                valid_prob_db_list.append(valid_prob_db)
                valid_prob_b_list.append(valid_prob_b)
                valid_pred_db_list.append(valid_pred_db)
                valid_pred_b_list.append(valid_pred_b)
                valid_pred_db_pool_list.append(valid_pred_db_pool)
                valid_pred_b_pool_list.append(valid_pred_b_pool)
                valid_mask_list.append(valid_mask)
                valid_y_db_list.append(valid_y_db)
                valid_y_b_list.append(valid_y_b)
            valid_prob_db = np.concatenate(valid_prob_db_list, axis=0)
            valid_prob_b = np.concatenate(valid_prob_b_list, axis=0)
            valid_pred_db = np.concatenate(valid_pred_db_list, axis=0)
            valid_pred_b = np.concatenate(valid_pred_b_list, axis=0)
            valid_pred_db_pool = np.concatenate(valid_pred_db_pool_list, axis=0)
            valid_pred_b_pool = np.concatenate(valid_pred_b_pool_list, axis=0)
            valid_mask = np.concatenate(valid_mask_list, axis=0)
            valid_y_db = np.concatenate(valid_y_db_list, axis=0)
            valid_y_b = np.concatenate(valid_y_b_list, axis=0)

            #  Evaluate validation predictions
            # Downbeat
            score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(valid_pred_db_pool, valid_dict, f_measure_threshold=0.07, target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_db, n_true_beats_db = sum([v[3] for v in score_dict_db_pool.values()]), sum([v[4] for v in score_dict_db_pool.values()])
            mean_P_db = sum([v[0] * v[3] for v in score_dict_db_pool.values()]) / n_pred_beats_db if n_pred_beats_db > 0 else 0
            mean_R_db = sum([v[1] * v[4] for v in score_dict_db_pool.values()]) / n_true_beats_db if n_true_beats_db > 0 else 0
            mean_F1_db = (2 * mean_P_db * mean_R_db) / (mean_P_db + mean_R_db) if (mean_P_db + mean_R_db) > 0 else 0
            # Beat
            score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(valid_pred_b_pool, valid_dict, f_measure_threshold=0.07, target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_b, n_true_beats_b = sum([v[3] for v in score_dict_b_pool.values()]), sum([v[4] for v in score_dict_b_pool.values()])
            mean_P_b = sum([v[0] * v[3] for v in score_dict_b_pool.values()]) / n_pred_beats_b if n_pred_beats_b > 0 else 0
            mean_R_b = sum([v[1] * v[4] for v in score_dict_b_pool.values()]) / n_true_beats_b if n_true_beats_b > 0 else 0
            mean_F1_b = (2 * mean_P_b * mean_R_b) / (mean_P_b + mean_R_b) if (mean_P_b + mean_R_b) > 0 else 0
            sess.run([mean_loss])
            valid_summary, valid_loss = sess.run([merged, summary_loss],
                                                 feed_dict={P_db: mean_P_db, R_db: mean_R_db, F1_db: mean_F1_db,
                                                            P_b: mean_P_b, R_b: mean_R_b, F1_b: mean_F1_b})
            valid_writer.add_summary(valid_summary, epoch)
            sess.run([clr_summary_valid, clr_summary_loss])
            print("---- epoch %d: valid_loss: total %.4f, db %.4f, b %.4f, Downbeat: P %.4f R %.4f F1 %.4f, Beat: P %.4f R %.4f F1 %.4f ----"
                % (epoch,
                   valid_loss[0], valid_loss[1], valid_loss[2],
                   mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b))
            print('db_best =', score_sorted_db_pool[::-1][:3])
            print('db_worst =', score_sorted_db_pool[:3])
            print('b_best =', score_sorted_b_pool[::-1][:3])
            print('b_worst =', score_sorted_b_pool[:3])

            # Sample a validation result
            sample_id = random.randint(0, n_valid_samples - 1)
            print('op =', valid_dict['recovery_id'][sample_id])
            print('len =', valid_dict['len'][sample_id])
            print('mask'.ljust(8), ''.join([' ' if x else 'x' for x in valid_mask[sample_id, :display_len]]))
            print('label_db'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in valid_y_db[sample_id, :display_len]]))
            print('label_b'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in valid_y_b[sample_id, :display_len]]))
            print('pred_db'.ljust(8), ''.join([str(x) for x in valid_pred_db[sample_id, :display_len]]))
            print('pred_b'.ljust(8), ''.join([str(x) for x in valid_pred_b[sample_id, :display_len]]))
            print('pred_db*'.ljust(8), ''.join([str(x) for x in valid_pred_db_pool[sample_id, :display_len]]))
            print('pred_b*'.ljust(8), ''.join([str(x) for x in valid_pred_b_pool[sample_id, :display_len]]))
            #
            print('prob_db'.ljust(8), [round(prob,1) for prob, db in zip(valid_prob_db[sample_id], valid_dict['downbeat'][sample_id]) if db == 1][:50])
            print('prob_b'.ljust(8), [round(prob,1) for prob, b in zip(valid_prob_b[sample_id], valid_dict['beat'][sample_id]) if b == 1][:50])

            # Print worst case
            worst_case = score_sorted_db_pool[0][0]
            print('worst_case_db =', worst_case)
            print('worst_true_db =', np.round(label_op_dict_db_pool[worst_case][:25], 2))
            print('worst_pred_db =', np.round(pred_op_dict_db_pool[worst_case][:25], 2))
            print('worst_true_b =', np.round(label_op_dict_b_pool[worst_case][:25], 2))
            print('worst_pred_b =', np.round(pred_op_dict_b_pool[worst_case][:25], 2))

            # Check if early stopping
            if (mean_F1_db + mean_F1_b) > sum(best_scores):
                best_scores = [mean_F1_db, mean_F1_b]
                best_epoch = epoch
                in_succession = 0
                performance = (mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b)
                # Save variables of the model
                print('\n*saving variables...\n')
                saver.save(sess, graph_location + '/audio_beat_and_down_beat_estimation_baseline_' + dataset + '.ckpt')
            else:
                in_succession += 1
                if in_succession > hp.n_in_succession:
                    print('Early stopping.')
                    break

        elapsed_time = time.time() - startTime
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best scores =', np.round(best_scores, 4))
        print('valid_fold =', valid_fold)
        print(dataset, "Downbeat P %.4f R %.4f F1 %.4f, Beat P %.4f R %.4f F1 %.4f" % performance)


def train_audio_beat_downbeat_estimation():

    dataset = 'Ballroom' # ['Ballroom', 'Hainsworth', 'GTZAN', 'ASAP']
    valid_fold = 0
    data_path = './ballroom_preprocessed_data_reshape'
    graph_location = './model'
    train_dict, valid_dict = get_preprocessed_data(data_path, dataset=dataset, valid_fold=valid_fold)

    n_train_samples = train_dict['spec'].shape[0]
    n_valid_samples = valid_dict['spec'].shape[0]
    n_iterations_per_epoch = math.ceil(n_train_samples / hp.batch_size)
    print('n_train_samples =', n_train_samples)
    print('n_valid_samples =', n_valid_samples)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp)

    with tf.name_scope('placeholder'):
        x_spec = tf.placeholder(tf.float32, [None, hp.sequence_length, 81, 2], name='spec')
        x_len = tf.placeholder(tf.int32, [None], name='valid_lengths')
        y_db = tf.placeholder(tf.int32, [None, hp.sequence_length], name='downbeat')
        y_b = tf.placeholder(tf.int32, [None, hp.sequence_length], name='beat')
        y_hop = tf.placeholder(tf.int32, [None], name='hop_size')
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        P_db, R_db, F1_db = tf.placeholder(dtype=tf.float32, name='P_db'), \
                            tf.placeholder(dtype=tf.float32, name='R_db'), \
                            tf.placeholder(dtype=tf.float32, name='F1_db')
        P_b, R_b, F1_b = tf.placeholder(dtype=tf.float32, name='P_b'), \
                         tf.placeholder(dtype=tf.float32, name='R_b'), \
                         tf.placeholder(dtype=tf.float32, name='F1_b')

    with tf.name_scope('model'):
        seq_mask_bool = tf.sequence_mask(x_len, hp.sequence_length, dtype=tf.bool) # Sequence mask
        seq_mask_float = tf.cast(seq_mask_bool, tf.float32)
        expanding_sizes = 3087 // y_hop # 3087 = 0.07*44100
        y_db_half = label_expanding(y_db, expanding_sizes, epsilon=0.5, mask=seq_mask_float)
        y_b_half = label_expanding(y_b, expanding_sizes, epsilon=0.5, mask=seq_mask_float)
        y_xb_half = y_b_half - y_db_half
        h = audio_joint_beat_downbeat_estimation_classTCN(x_spec, x_len, dropout, is_training, hp)

    with tf.name_scope('output_layer'):
        logits = tf.layers.conv1d(h, filters=3, kernel_size=7, padding='same', name='out_conv') # [B, T, 3]
        logits = tf.layers.conv1d(logits, filters=3, kernel_size=1, padding='same', name='out_trans')  # [B, T, 3]
        act = tf.nn.softmax(logits, axis=-1) # [B, T, 3]
        prob_db, prob_xb, _ = tf.unstack(act, axis=-1) # [B, T]
        prob_b = prob_db + prob_xb # [B, T]

    with tf.name_scope('label_embedding'):
        vocabulary_db = 500
        vocabulary_b = 150
        embedding_size_db = hp.n_filters
        embedding_size_b = hp.n_filters
        y_db_period = label2period(y_db) # [B, T]
        y_b_period = label2period(y_b) # [B, T]
        y_db_period = tf.clip_by_value(y_db_period, clip_value_min=0, clip_value_max=vocabulary_db - 1)
        y_b_period = tf.clip_by_value(y_b_period, clip_value_min=0, clip_value_max=vocabulary_b - 1)
        embed_db = tf.get_variable(
            name='label_embeddings_db',
            shape=(vocabulary_db, embedding_size_db),
            dtype=tf.float32,
            trainable=True,
        ) # [vocabulary, embedding_size]
        embed_b = tf.get_variable(
            name='label_embeddings_b',
            shape=(vocabulary_b, embedding_size_b),
            dtype=tf.float32,
            trainable=True,
        ) # [vocabulary, embedding_size]
        y_db_embed = tf.nn.embedding_lookup(params=embed_db, ids=y_db_period) # [B, T, embedding_size]
        y_b_embed = tf.nn.embedding_lookup(params=embed_b, ids=y_b_period) # [B, T, embedding_size]
        y_db_b_embed = tf.concat([y_db_embed, y_b_embed], axis=-1) # [B, T, 2*embedding_size]
        y_db_b_embed *= seq_mask_float[:, :, None] # [B, T, 2*embedding_size]

        y_db_b_embed = tf.layers.conv1d(
            y_db_b_embed, filters=hp.n_filters, kernel_size=7, padding='same', name='embedding_CNN'
        ) # [B, T, 24]
        y_db_b_embed = hp.activation(y_db_b_embed)
        y_db_b_embed *= seq_mask_float[:, :, None] # [B, T, 24]

    with tf.name_scope('label_reconstruction'):
        y_db_b_rec = tf.layers.conv1d(
            y_db_b_embed, filters=embedding_size_db+embedding_size_b, kernel_size=7,
            padding='same', name='reconstruction_CNN'
        ) # [B, T, 48]
        y_db_reconstruct, y_b_reconstruct = tf.split(y_db_b_rec, num_or_size_splits=[embedding_size_db, embedding_size_b], axis=-1)

        y_db_reconstruct = tf.matmul(y_db_reconstruct,
                                     tf.tile(embed_db[None, :, :], [tf.shape(y_db_reconstruct)[0], 1, 1]),
                                     transpose_b=True) # [B, T, vocabulary_db]
        y_b_reconstruct = tf.matmul(y_b_reconstruct,
                                     tf.tile(embed_b[None, :, :], [tf.shape(y_b_reconstruct)[0], 1, 1]),
                                     transpose_b=True) # [B, T, vocabulary_b]

        y_db_reconstruct = tf.nn.softmax(y_db_reconstruct, axis=-1)
        y_b_reconstruct = tf.nn.softmax(y_b_reconstruct, axis=-1)

    with tf.variable_scope('loss'):
        n_valid = tf.reduce_sum(tf.cast(seq_mask_bool, tf.float32))

        # Dice loss
        dice_loss_db = dice_loss_from_probs(labels=y_db_half, p=prob_db, mask=seq_mask_float)
        dice_loss_b = dice_loss_from_probs(labels=y_xb_half, p=prob_xb, mask=seq_mask_float)

        # Structural Regularization
        dist = tf.norm((h - y_db_b_embed), ord='euclidean', axis=-1) # [B, T]
        dist *= (y_b_half + 0.1) # weighting
        sr_loss = 0.6 * tf.reduce_sum(dist*seq_mask_float) / tf.reduce_sum(seq_mask_float)

        # Focal loss
        y_class = jointly_supervised_class(y_b_half, y_db_half) # [B, T, 3]
        focal_alpha = [6,3,1]
        label_smoothing = 0.0
        focal_loss = 5 * categorical_focal_loss_from_probs(
            labels=y_class, probs=act, alpha=focal_alpha, label_smoothing=label_smoothing, mask=seq_mask_float
        )

        # Feature extraction L2-loss
        fe_vars = [var for var in tf.trainable_variables() if 'input_encoding' in var.name and 'bias' not in var.name and 'beta' not in var.name]
        l2_loss = 5e-4 * tf.add_n([ tf.nn.l2_loss(v) for v in fe_vars])

        # Label reconstruction loss
        db_alpha = [1]
        b_alpha = [1]
        rec_loss_db = categorical_focal_loss_from_probs(
            labels=tf.one_hot(y_db_period, vocabulary_db),
            probs=y_db_reconstruct,
            alpha=db_alpha,
            label_smoothing=0.0,
            mask=seq_mask_float)
        rec_loss_b = categorical_focal_loss_from_probs(
            labels=tf.one_hot(y_b_period, vocabulary_b),
            probs=y_b_reconstruct,
            alpha=b_alpha,
            label_smoothing=0.0,
            mask=seq_mask_float)
        rec_loss = 0.07 * (rec_loss_db + rec_loss_b)

        # Total loss
        loss = dice_loss_db + dice_loss_b + focal_loss + sr_loss + rec_loss + l2_loss
    summary_loss = tf.Variable([0.0 for _ in range(7)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + n_valid * [loss, dice_loss_db, dice_loss_b, focal_loss, sr_loss, rec_loss, l2_loss])
    update_valid = tf.assign(summary_valid, summary_valid + n_valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Dice_loss_db', summary_loss[1])
    tf.summary.scalar('Dice_loss_b', summary_loss[2])
    tf.summary.scalar('Focal_loss', summary_loss[3])
    tf.summary.scalar('SR_loss', summary_loss[4])
    tf.summary.scalar('Rec_loss', summary_loss[5])
    tf.summary.scalar('L2_loss', summary_loss[6])

    with tf.name_scope('evaluation'):
        # Downbeat
        prob_db = prob_db * seq_mask_float
        pred_db = tf.cast(tf.greater(prob_db, hp.threshold), tf.int32)
        prob_db_pool = output_probability_pooling(prob_db, pool_size=7) # pooling by local maximums
        pred_db_pool = tf.cast(tf.greater(prob_db_pool, hp.threshold), tf.int32)
        # Beat
        prob_b = prob_b * seq_mask_float
        pred_b = tf.cast(tf.greater(prob_b, hp.threshold), tf.int32)
        prob_b_pool = output_probability_pooling(prob_b, pool_size=7) # pooling by local maximums
        pred_b_pool = tf.cast(tf.greater(prob_b_pool, hp.threshold), tf.int32)
    tf.summary.scalar('Precision_db', P_db)
    tf.summary.scalar('Recall_db', R_db)
    tf.summary.scalar('F1_db', F1_db)
    tf.summary.scalar('Precision_b', P_b)
    tf.summary.scalar('Recall_b', R_b)
    tf.summary.scalar('F1_b', F1_b)

    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.hidden_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        train_op = optimizer.minimize(loss)

    # Graph location and summary writers
    print('Saving graph to: %s' % graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(graph_location + '/train')
    valid_writer = tf.summary.FileWriter(graph_location + '/valid')
    train_writer.add_graph(tf.get_default_graph())
    valid_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()

    # Training
    print('\nTrain the model...')
    np.set_printoptions(suppress=True, linewidth=180)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_scores = [0.0, 0.0]
        in_succession = 0
        best_epoch = 0
        indices = np.arange(n_train_samples)
        batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]
        step = 1
        for epoch in range(1, hp.n_training_epochs+1):
            # Training
            if epoch > 2:
                # Shuffle training data
                indices = np.array(random.sample(range(n_train_samples), n_train_samples))
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]
            # Batch-wise feeding
            train_pred_db_pool_list, train_pred_b_pool_list = [], []
            for i_b, batch_idx in enumerate(batch_indices):
                batch = {'spec': train_dict['spec'][batch_idx],
                         'len': train_dict['len'][batch_idx],
                         'downbeat': train_dict['downbeat'][batch_idx],
                         'beat': train_dict['beat'][batch_idx],
                         'id': train_dict['recovery_id'][batch_idx],
                         'hop': train_dict['hop'][batch_idx]}

                train_run_list = [train_op, update_valid, update_loss,
                                  loss, dice_loss_db, dice_loss_b, focal_loss, sr_loss, rec_loss, l2_loss,
                                  prob_db, prob_b, pred_db, pred_b, pred_db_pool, pred_b_pool, seq_mask_bool, y_db_half, y_b_half]
                train_feed_fict = {x_spec: batch['spec'],
                                   x_len: batch['len'],
                                   y_db: batch['downbeat'],
                                   y_b: batch['beat'],
                                   y_hop: batch['hop'],
                                   dropout: hp.drop,
                                   is_training: True,
                                   global_step: step}
                _, _, _, train_loss, train_dloss_db, train_dloss_b, train_floss, train_srloss, train_recloss, train_l2loss, \
                train_prob_db, train_prob_b, train_pred_db, train_pred_b, train_pred_db_pool, train_pred_b_pool, \
                train_mask, train_y_db, train_y_b = sess.run(train_run_list, feed_dict=train_feed_fict)

                # Store batch-wise results
                train_pred_db_pool_list.append(train_pred_db_pool)
                train_pred_b_pool_list.append(train_pred_b_pool)
                # show the loss information
                if step == 1:
                    print('*~ total loss %.4f, dloss(db %.4f, b %.4f), floss %.4f, srloss %.4f, recloss %.4f, l2loss %.4f) ~*'
                          % (train_loss, train_dloss_db, train_dloss_b, train_floss, train_srloss, train_recloss, train_l2loss))
                step +=1

            # Concat stored results
            train_pred_db_pool_all = np.concatenate(train_pred_db_pool_list, axis=0)
            train_pred_b_pool_all = np.concatenate(train_pred_b_pool_list, axis=0)

            # Recovery sequential order
            gather_id = [np.where(indices == ord)[0][0] for ord in range(n_train_samples)]
            train_pred_db_pool_all = train_pred_db_pool_all[gather_id, :]
            train_pred_b_pool_all = train_pred_b_pool_all[gather_id, :]

            # Evaluate training predictions
            # Downbeat
            score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(train_pred_db_pool_all, train_dict, f_measure_threshold=0.07, target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1, item = (key, scores), scores=(P, R, F1, len(pred), len(label))
            n_pred_beats_db, n_true_beats_db = sum([v[3] for v in score_dict_db_pool.values()]), sum([v[4] for v in score_dict_db_pool.values()])
            mean_P_db = sum([v[0] * v[3] for v in score_dict_db_pool.values()]) / n_pred_beats_db if n_pred_beats_db > 0 else 0
            mean_R_db = sum([v[1] * v[4] for v in score_dict_db_pool.values()]) / n_true_beats_db if n_true_beats_db > 0 else 0
            mean_F1_db = (2 * mean_P_db * mean_R_db) / (mean_P_db + mean_R_db) if (mean_P_db + mean_R_db) > 0 else 0
            # Beat
            score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(train_pred_b_pool_all, train_dict, f_measure_threshold=0.07, target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_b, n_true_beats_b = sum([v[3] for v in score_dict_b_pool.values()]), sum([v[4] for v in score_dict_b_pool.values()])
            mean_P_b = sum([v[0] * v[3] for v in score_dict_b_pool.values()]) / n_pred_beats_b if n_pred_beats_b > 0 else 0
            mean_R_b = sum([v[1] * v[4] for v in score_dict_b_pool.values()]) / n_true_beats_b if n_true_beats_b > 0 else 0
            mean_F1_b = (2 * mean_P_b * mean_R_b) / (mean_P_b + mean_R_b) if (mean_P_b + mean_R_b) > 0 else 0

            # Display training log
            sess.run([mean_loss])
            train_summary, train_loss = sess.run([merged, summary_loss],
                                                 feed_dict={P_db: mean_P_db, R_db: mean_R_db, F1_db: mean_F1_db,
                                                            P_b: mean_P_b, R_b: mean_R_b, F1_b: mean_F1_b})
            train_writer.add_summary(train_summary, epoch)
            sess.run([clr_summary_valid, clr_summary_loss])
            print("==== epoch %d: train_loss: total %.4f, dice(db %.4f, b %.4f), focal %.4f, srloss %.4f, recloss %.4f, l2loss %.4f), Downbeat: P %.4f R %.4f F1 %.4f, Beat: P %.4f R %.4f F1 %.4f ===="
                % (epoch,
                   train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], train_loss[6],
                   mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b))
            print('db_best =', score_sorted_db_pool[::-1][:3])
            print('db_worst =', score_sorted_db_pool[:3])
            print('b_best =', score_sorted_b_pool[::-1][:3])
            print('b_worst =', score_sorted_b_pool[:3])
            display_len = 200
            print('op =', batch['id'][0])
            print('len =', batch['len'][0])
            print('mask'.ljust(8), ''.join([' ' if x else 'x' for x in train_mask[0, :display_len]]))
            print('label_db'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in train_y_db[0, :display_len]]))
            print('label_b'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in train_y_b[0, :display_len]]))
            print('pred_db'.ljust(8), ''.join([str(x) for x in train_pred_db[0, :display_len]]))
            print('pred_b'.ljust(8), ''.join([str(x) for x in train_pred_b[0, :display_len]]))
            print('pred_db*'.ljust(8), ''.join([str(x) for x in train_pred_db_pool[0, :display_len]]))
            print('pred_b*'.ljust(8), ''.join([str(x) for x in train_pred_b_pool[0, :display_len]]))
            print('prob_db'.ljust(8), [round(prob,1) for prob, db in zip(train_prob_db[0], batch['downbeat'][0]) if db == 1][:50])
            print('prob_b'.ljust(8), [round(prob,1) for prob, b in zip(train_prob_b[0], batch['beat'][0]) if b == 1][:50])

            # Validation
            valid_prob_db_list, valid_prob_b_list = [], []
            valid_pred_db_list, valid_pred_b_list = [], []
            valid_pred_db_pool_list, valid_pred_b_pool_list = [], []
            valid_mask_list = []
            valid_y_db_list, valid_y_b_list = [], []
            valid_batch_size = hp.batch_size
            for i in range(0, n_valid_samples, valid_batch_size):
                valid_run_list = [update_valid, update_loss,
                                  prob_db, prob_b, pred_db, pred_b, pred_db_pool, pred_b_pool, seq_mask_bool, y_db_half, y_b_half]
                valid_feed_fict = {x_spec: valid_dict['spec'][i:i+valid_batch_size],
                                   x_len: valid_dict['len'][i:i+valid_batch_size],
                                   y_db: valid_dict['downbeat'][i:i+valid_batch_size],
                                   y_b: valid_dict['beat'][i:i+valid_batch_size],
                                   y_hop: valid_dict['hop'][i:i + valid_batch_size],
                                   dropout: 0.0,
                                   is_training: False,
                                   global_step: step}
                _, _, valid_prob_db, valid_prob_b, valid_pred_db, valid_pred_b, valid_pred_db_pool, valid_pred_b_pool, \
                valid_mask, valid_y_db, valid_y_b = sess.run(valid_run_list, feed_dict=valid_feed_fict)
                # Store batch-wise results
                valid_prob_db_list.append(valid_prob_db)
                valid_prob_b_list.append(valid_prob_b)
                valid_pred_db_list.append(valid_pred_db)
                valid_pred_b_list.append(valid_pred_b)
                valid_pred_db_pool_list.append(valid_pred_db_pool)
                valid_pred_b_pool_list.append(valid_pred_b_pool)
                valid_mask_list.append(valid_mask)
                valid_y_db_list.append(valid_y_db)
                valid_y_b_list.append(valid_y_b)
            valid_prob_db = np.concatenate(valid_prob_db_list, axis=0)
            valid_prob_b = np.concatenate(valid_prob_b_list, axis=0)
            valid_pred_db = np.concatenate(valid_pred_db_list, axis=0)
            valid_pred_b = np.concatenate(valid_pred_b_list, axis=0)
            valid_pred_db_pool = np.concatenate(valid_pred_db_pool_list, axis=0)
            valid_pred_b_pool = np.concatenate(valid_pred_b_pool_list, axis=0)
            valid_mask = np.concatenate(valid_mask_list, axis=0)
            valid_y_db = np.concatenate(valid_y_db_list, axis=0)
            valid_y_b = np.concatenate(valid_y_b_list, axis=0)

            #  Evaluate validation predictions
            # Downbeat
            score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(valid_pred_db_pool, valid_dict, f_measure_threshold=0.07, target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_db, n_true_beats_db = sum([v[3] for v in score_dict_db_pool.values()]), sum([v[4] for v in score_dict_db_pool.values()])
            mean_P_db = sum([v[0] * v[3] for v in score_dict_db_pool.values()]) / n_pred_beats_db if n_pred_beats_db > 0 else 0
            mean_R_db = sum([v[1] * v[4] for v in score_dict_db_pool.values()]) / n_true_beats_db if n_true_beats_db > 0 else 0
            mean_F1_db = (2 * mean_P_db * mean_R_db) / (mean_P_db + mean_R_db) if (mean_P_db + mean_R_db) > 0 else 0
            # Beat
            score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(valid_pred_b_pool, valid_dict, f_measure_threshold=0.07, target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_b, n_true_beats_b = sum([v[3] for v in score_dict_b_pool.values()]), sum([v[4] for v in score_dict_b_pool.values()])
            mean_P_b = sum([v[0] * v[3] for v in score_dict_b_pool.values()]) / n_pred_beats_b if n_pred_beats_b > 0 else 0
            mean_R_b = sum([v[1] * v[4] for v in score_dict_b_pool.values()]) / n_true_beats_b if n_true_beats_b > 0 else 0
            mean_F1_b = (2 * mean_P_b * mean_R_b) / (mean_P_b + mean_R_b) if (mean_P_b + mean_R_b) > 0 else 0
            sess.run([mean_loss])
            valid_summary, valid_loss = sess.run([merged, summary_loss],
                                                 feed_dict={P_db: mean_P_db, R_db: mean_R_db, F1_db: mean_F1_db,
                                                            P_b: mean_P_b, R_b: mean_R_b, F1_b: mean_F1_b})
            valid_writer.add_summary(valid_summary, epoch)
            sess.run([clr_summary_valid, clr_summary_loss])
            print("---- epoch %d: b_loss: total %.4f, dice(db %.4f, b %.4f), focal %.4f, srloss %.4f, recloss %.4f, l2loss %.4f, Downbeat: P %.4f R %.4f F1 %.4f, Beat: P %.4f R %.4f F1 %.4f ----"
                % (epoch,
                   valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], valid_loss[6],
                   mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b))
            print('db_best =', score_sorted_db_pool[::-1][:3])
            print('db_worst =', score_sorted_db_pool[:3])
            print('b_best =', score_sorted_b_pool[::-1][:3])
            print('b_worst =', score_sorted_b_pool[:3])

            # Sample a validation result
            sample_id = random.randint(0, n_valid_samples - 1)
            print('op =', valid_dict['recovery_id'][sample_id])
            print('len =', valid_dict['len'][sample_id])
            print('mask'.ljust(8), ''.join([' ' if x else 'x' for x in valid_mask[sample_id, :display_len]]))
            print('label_db'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in valid_y_db[sample_id, :display_len]]))
            print('label_b'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in valid_y_b[sample_id, :display_len]]))
            print('pred_db'.ljust(8), ''.join([str(x) for x in valid_pred_db[sample_id, :display_len]]))
            print('pred_b'.ljust(8), ''.join([str(x) for x in valid_pred_b[sample_id, :display_len]]))
            print('pred_db*'.ljust(8), ''.join([str(x) for x in valid_pred_db_pool[sample_id, :display_len]]))
            print('pred_b*'.ljust(8), ''.join([str(x) for x in valid_pred_b_pool[sample_id, :display_len]]))
            print('prob_db'.ljust(8), [round(prob,1) for prob, db in zip(valid_prob_db[sample_id], valid_dict['downbeat'][sample_id]) if db == 1][:50])
            print('prob_b'.ljust(8), [round(prob,1) for prob, b in zip(valid_prob_b[sample_id], valid_dict['beat'][sample_id]) if b == 1][:50])

            # Print worst case
            worst_case = score_sorted_db_pool[0][0]
            print('worst_case_db =', worst_case)
            print('worst_true_db =', np.round(label_op_dict_db_pool[worst_case][:25], 2))
            print('worst_pred_db =', np.round(pred_op_dict_db_pool[worst_case][:25], 2))
            print('worst_true_b =', np.round(label_op_dict_b_pool[worst_case][:25], 2))
            print('worst_pred_b =', np.round(pred_op_dict_b_pool[worst_case][:25], 2))

            # Check if early stopping
            if (mean_F1_db + mean_F1_b) > sum(best_scores):
                best_scores = [mean_F1_db, mean_F1_b]
                best_epoch = epoch
                in_succession = 0
                performance = (mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b)
                # Save variables of the model
                print('\n*saving variables...\n')
                saver.save(sess, graph_location + '/audio_beat_and_down_beat_estimation_' + dataset + '.ckpt')
            else:
                in_succession += 1
                if in_succession > hp.n_in_succession:
                    print('Early stopping.')
                    break

        elapsed_time = time.time() - startTime
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best scores =', np.round(best_scores, 4))
        print('valid_fold =', valid_fold)
        print(dataset, "Downbeat P %.4f R %.4f F1 %.4f, Beat P %.4f R %.4f F1 %.4f" % performance)


def train_symbolic_beat_downbeat_estimation():

    dataset = 'ASAP'
    data_path = './ballroom_preprocessed_data_reshape'
    graph_location = './model'
    train_dict, test_dict = get_preprocessed_data(data_path, dataset=dataset, symbolic=True)

    n_train_samples = train_dict['pianoroll'].shape[0]
    n_test_samples = test_dict['pianoroll'].shape[0]
    n_iterations_per_epoch = math.ceil(n_train_samples / hp.batch_size)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp)

    with tf.name_scope('placeholder'):
        x_pianoroll = tf.placeholder(tf.float32, [None, hp.sequence_length, 88], name='pianoroll')
        x_onset = tf.placeholder(tf.float32, [None, hp.sequence_length, 88], name='onset')
        x_IOI = tf.placeholder(tf.float32, [None, hp.sequence_length], name='IOI')
        x_flux = tf.placeholder(tf.float32, [None, hp.sequence_length], name='flux')
        x_len = tf.placeholder(tf.int32, [None], name='valid_lengths')
        y_db = tf.placeholder(tf.int32, [None, hp.sequence_length], name='downbeat')
        y_b = tf.placeholder(tf.int32, [None, hp.sequence_length], name='beat')
        y_hop = tf.placeholder(tf.int32, [None], name='hop_size')
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        P_db, R_db, F1_db = tf.placeholder(dtype=tf.float32, name='P_db'), tf.placeholder(dtype=tf.float32, name='R_db'), tf.placeholder(dtype=tf.float32, name='F1_db')
        P_b, R_b, F1_b = tf.placeholder(dtype=tf.float32, name='P_b'), tf.placeholder(dtype=tf.float32, name='R_b'), tf.placeholder(dtype=tf.float32, name='F1_b')

    with tf.name_scope('model'):
        seq_mask_bool = tf.sequence_mask(x_len, hp.sequence_length, dtype=tf.bool) # Sequence mask
        seq_mask_float = tf.cast(seq_mask_bool, tf.float32)
        expanding_sizes = 3087 // y_hop # 3087 = 0.07*44100
        y_db_half = label_expanding(y_db, expanding_sizes, epsilon=0.5, mask=seq_mask_float)
        y_b_half = label_expanding(y_b, expanding_sizes, epsilon=0.5, mask=seq_mask_float)
        y_xb_half = y_b_half - y_db_half
        h = symbolic_joint_beat_downbeat_estimation_classTCN(x_pianoroll, x_onset, x_IOI, x_flux,
                                                             x_len, dropout, is_training, hp)

    with tf.name_scope('output_layer'):
        logits = tf.layers.conv1d(h, filters=3, kernel_size=7, padding='same', name='out_conv') # [B, T, 3]
        logits = tf.layers.conv1d(logits, filters=3, kernel_size=1, padding='same', name='out_trans')  # [B, T, 3]
        act = tf.nn.softmax(logits, axis=-1) # [B, T, 3]
        prob_db, prob_xb, _ = tf.unstack(act, axis=-1) # [B, T]
        prob_b = prob_db + prob_xb # [B, T]

    with tf.variable_scope('loss'):
        n_valid = tf.reduce_sum(tf.cast(seq_mask_bool, tf.float32))

        # Dice loss
        dice_loss_db = dice_loss_from_probs(labels=y_db_half, p=prob_db, mask=seq_mask_float)
        dice_loss_b = dice_loss_from_probs(labels=y_xb_half, p=prob_xb, mask=seq_mask_float)

        # Focal loss
        y_class = jointly_supervised_class(y_b_half, y_db_half)  # [B, T, 3]
        focal_alpha = [6, 3, 1]  # [5,4,1] # [2,2,1] # [1,1,3]
        label_smoothing = 0.0  # 0.1
        focal_loss = 5 * categorical_focal_loss_from_probs(
            labels=y_class, probs=act, alpha=focal_alpha, label_smoothing=label_smoothing, mask=seq_mask_float
        )

        # Feature extraction L2-loss
        fe_vars = [var for var in tf.trainable_variables() if 'bias' not in var.name and 'beta' not in var.name]
        l2_loss = 5e-4 * tf.add_n([tf.nn.l2_loss(v) for v in fe_vars])

        # Total loss
        loss = dice_loss_db + dice_loss_b + focal_loss + l2_loss
    summary_loss = tf.Variable([0.0 for _ in range(5)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss,
                            summary_loss + n_valid * [loss, dice_loss_db, dice_loss_b, focal_loss, l2_loss])
    update_valid = tf.assign(summary_valid, summary_valid + n_valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Dice_loss_db', summary_loss[1])
    tf.summary.scalar('Dice_loss_b', summary_loss[2])
    tf.summary.scalar('Focal_loss', summary_loss[3])
    tf.summary.scalar('L2_loss_db', summary_loss[4])

    with tf.name_scope('evaluation'):
        # Downbeat
        prob_db = prob_db * seq_mask_float
        pred_db = tf.cast(tf.greater(prob_db, hp.threshold), tf.int32)
        prob_db_pool = output_probability_pooling(prob_db, pool_size=7) # pooling by local maximums
        pred_db_pool = tf.cast(tf.greater(prob_db_pool, hp.threshold), tf.int32)
        # Beat
        prob_b = prob_b * seq_mask_float
        pred_b = tf.cast(tf.greater(prob_b, hp.threshold), tf.int32)
        prob_b_pool = output_probability_pooling(prob_b, pool_size=7) # pooling by local maximums
        pred_b_pool = tf.cast(tf.greater(prob_b_pool, hp.threshold), tf.int32)
    tf.summary.scalar('Precision_db', P_db)
    tf.summary.scalar('Recall_db', R_db)
    tf.summary.scalar('F1_db', F1_db)
    tf.summary.scalar('Precision_b', P_b)
    tf.summary.scalar('Recall_b', R_b)
    tf.summary.scalar('F1_b', F1_b)

    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.hidden_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        train_op = optimizer.minimize(loss)

        # Update ops for Batch Norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])
    # Graph location and summary writers
    print('Saving graph to: %s' % graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(graph_location + '/train')
    test_writer = tf.summary.FileWriter(graph_location + '/test')
    train_writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()

    # Training
    print('\nTrain the model...')
    np.set_printoptions(suppress=True, linewidth=180)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_scores = [0.0, 0.0]
        in_succession = 0
        best_epoch = 0
        indices = np.arange(n_train_samples)
        batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]
        step = 1
        for epoch in range(1, hp.n_training_epochs+1):
            # Training
            if epoch > 2:
                # Shuffle training data
                indices = np.array(random.sample(range(n_train_samples), n_train_samples))
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]
            # Batch-wise feeding
            train_pred_db_pool_list, train_pred_b_pool_list = [], []
            for i_b, batch_idx in enumerate(batch_indices):
                batch = {'pianoroll': train_dict['pianoroll'][batch_idx],
                         'onset': train_dict['onset'][batch_idx],
                         'IOI': train_dict['IOI'][batch_idx],
                         'flux': train_dict['flux'][batch_idx],
                         'len': train_dict['len'][batch_idx],
                         'downbeat': train_dict['downbeat'][batch_idx],
                         'beat': train_dict['beat'][batch_idx],
                         'id': train_dict['recovery_id'][batch_idx],
                         'hop': train_dict['hop'][batch_idx]}

                train_run_list = [train_op, update_valid, update_loss,
                                  loss, dice_loss_db, dice_loss_b, focal_loss, l2_loss,
                                  prob_db, prob_b, pred_db, pred_b, pred_db_pool, pred_b_pool, seq_mask_bool, y_db_half, y_b_half]
                train_feed_fict = {x_pianoroll: batch['pianoroll'],
                                   x_onset: batch['onset'],
                                   x_IOI: batch['IOI'],
                                   x_flux: batch['flux'],
                                   x_len: batch['len'],
                                   y_db: batch['downbeat'],
                                   y_b: batch['beat'],
                                   y_hop: batch['hop'],
                                   dropout: hp.drop,
                                   is_training: True,
                                   global_step: step}
                _, _, _, train_loss, train_dloss_db, train_dloss_b, train_floss, train_l2loss,\
                train_prob_db, train_prob_b, train_pred_db, train_pred_b, \
                train_pred_db_pool, train_pred_b_pool,\
                train_mask, train_y_db, train_y_b = sess.run(train_run_list, feed_dict=train_feed_fict)

                # Store batch-wise results
                train_pred_db_pool_list.append(train_pred_db_pool)
                train_pred_b_pool_list.append(train_pred_b_pool)
                # show the loss information
                if step == 1:
                    print('*~ total loss %.4f, dloss(db %.4f, b %.4f), floss %.4f, l2loss %.4f ~*'
                          % (train_loss, train_dloss_db, train_dloss_b, train_floss, train_l2loss))
                step +=1

            # Concat stored results
            train_pred_db_pool_all = np.concatenate(train_pred_db_pool_list, axis=0)
            train_pred_b_pool_all = np.concatenate(train_pred_b_pool_list, axis=0)
            # Recovery sequential order
            gather_id = [np.where(indices == ord)[0][0] for ord in range(n_train_samples)]
            train_pred_db_pool_all = train_pred_db_pool_all[gather_id, :]
            train_pred_b_pool_all = train_pred_b_pool_all[gather_id, :]

            # Evaluate training predictions
            # Downbeat
            score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(train_pred_db_pool_all, train_dict, f_measure_threshold=0.07, target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1, item = (key, scores), scores=(P, R, F1, len(pred), len(label))
            n_pred_beats_db, n_true_beats_db = sum([v[3] for v in score_dict_db_pool.values()]), sum([v[4] for v in score_dict_db_pool.values()])
            mean_P_db = sum([v[0] * v[3] for v in score_dict_db_pool.values()]) / n_pred_beats_db if n_pred_beats_db > 0 else 0
            mean_R_db = sum([v[1] * v[4] for v in score_dict_db_pool.values()]) / n_true_beats_db if n_true_beats_db > 0 else 0
            mean_F1_db = (2 * mean_P_db * mean_R_db) / (mean_P_db + mean_R_db) if (mean_P_db + mean_R_db) > 0 else 0
            # Beat
            score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(train_pred_b_pool_all, train_dict, f_measure_threshold=0.07, target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_b, n_true_beats_b = sum([v[3] for v in score_dict_b_pool.values()]), sum([v[4] for v in score_dict_b_pool.values()])
            mean_P_b = sum([v[0] * v[3] for v in score_dict_b_pool.values()]) / n_pred_beats_b if n_pred_beats_b > 0 else 0
            mean_R_b = sum([v[1] * v[4] for v in score_dict_b_pool.values()]) / n_true_beats_b if n_true_beats_b > 0 else 0
            mean_F1_b = (2 * mean_P_b * mean_R_b) / (mean_P_b + mean_R_b) if (mean_P_b + mean_R_b) > 0 else 0

            # Display training log
            sess.run([mean_loss])
            train_summary, train_loss = sess.run([merged, summary_loss],
                                                 feed_dict={P_db: mean_P_db, R_db: mean_R_db, F1_db: mean_F1_db,
                                                            P_b: mean_P_b, R_b: mean_R_b, F1_b: mean_F1_b})
            train_writer.add_summary(train_summary, epoch)
            sess.run([clr_summary_valid, clr_summary_loss])
            print("==== epoch %d: train_loss: total %.4f, dice(db %.4f, b %.4f), focal %.4f, l2 %.4f, Downbeat: P %.4f R %.4f F1 %.4f, Beat: P %.4f R %.4f F1 %.4f ===="
                % (epoch,
                   train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4],
                   mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b))
            print('db_best =', score_sorted_db_pool[::-1][:3])
            print('db_worst =', score_sorted_db_pool[:3])
            print('b_best =', score_sorted_b_pool[::-1][:3])
            print('b_worst =', score_sorted_b_pool[:3])
            display_len = 200
            print('op =', batch['id'][0])
            print('len =', batch['len'][0])
            print('mask'.ljust(8), ''.join([' ' if x else 'x' for x in train_mask[0, :display_len]]))
            print('label_db'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in train_y_db[0, :display_len]]))
            print('label_b'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in train_y_b[0, :display_len]]))
            print('pred_db'.ljust(8), ''.join([str(x) for x in train_pred_db[0, :display_len]]))
            print('pred_b'.ljust(8), ''.join([str(x) for x in train_pred_b[0, :display_len]]))
            print('pred_db*'.ljust(8), ''.join([str(x) for x in train_pred_db_pool[0, :display_len]]))
            print('pred_b*'.ljust(8), ''.join([str(x) for x in train_pred_b_pool[0, :display_len]]))
            #
            print('prob_db'.ljust(8), [round(prob,1) for prob, db in zip(train_prob_db[0], batch['downbeat'][0]) if db == 1][:50])
            print('prob_b'.ljust(8), [round(prob,1) for prob, b in zip(train_prob_b[0], batch['beat'][0]) if b == 1][:50])

            # Test
            test_prob_db_list, test_prob_b_list = [], []
            test_pred_db_list, test_pred_b_list = [], []
            test_pred_db_pool_list, test_pred_b_pool_list = [], []
            test_mask_list = []
            test_y_db_list, test_y_b_list = [], []
            test_batch_size = hp.batch_size # n_test_samples #
            for i in range(0, n_test_samples, test_batch_size):
                test_run_list = [update_valid, update_loss,
                                 prob_db, prob_b, pred_db, pred_b, pred_db_pool, pred_b_pool, seq_mask_bool, y_db_half, y_b_half]
                test_feed_fict = {x_pianoroll: test_dict['pianoroll'][i:i + test_batch_size],
                                  x_onset: test_dict['onset'][i:i + test_batch_size],
                                  x_IOI: test_dict['IOI'][i:i + test_batch_size],
                                  x_flux: test_dict['flux'][i:i + test_batch_size],
                                  x_len: test_dict['len'][i:i + test_batch_size],
                                  y_db: test_dict['downbeat'][i:i + test_batch_size],
                                  y_b: test_dict['beat'][i:i + test_batch_size],
                                  y_hop: test_dict['hop'][i:i + test_batch_size],
                                  dropout: 0.0,
                                  is_training: False,
                                  global_step: step}
                _, _, test_prob_db, test_prob_b, test_pred_db, test_pred_b, test_pred_db_pool, test_pred_b_pool, \
                test_mask, test_y_db, test_y_b = sess.run(test_run_list, feed_dict=test_feed_fict)
                # Store batch-wise results
                test_prob_db_list.append(test_prob_db)
                test_prob_b_list.append(test_prob_b)
                test_pred_db_list.append(test_pred_db)
                test_pred_b_list.append(test_pred_b)
                test_pred_db_pool_list.append(test_pred_db_pool)
                test_pred_b_pool_list.append(test_pred_b_pool)
                test_mask_list.append(test_mask)
                test_y_db_list.append(test_y_db)
                test_y_b_list.append(test_y_b)
            test_prob_db = np.concatenate(test_prob_db_list, axis=0)
            test_prob_b = np.concatenate(test_prob_b_list, axis=0)
            test_pred_db = np.concatenate(test_pred_db_list, axis=0)
            test_pred_b = np.concatenate(test_pred_b_list, axis=0)
            test_pred_db_pool = np.concatenate(test_pred_db_pool_list, axis=0)
            test_pred_b_pool = np.concatenate(test_pred_b_pool_list, axis=0)
            test_mask = np.concatenate(test_mask_list, axis=0)
            test_y_db = np.concatenate(test_y_db_list, axis=0)
            test_y_b = np.concatenate(test_y_b_list, axis=0)

            #  Evaluate validation predictions
            # Downbeat
            score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(test_pred_db_pool, test_dict, f_measure_threshold=0.07, target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_db, n_true_beats_db = sum([v[3] for v in score_dict_db_pool.values()]), sum([v[4] for v in score_dict_db_pool.values()])
            mean_P_db = sum([v[0] * v[3] for v in score_dict_db_pool.values()]) / n_pred_beats_db if n_pred_beats_db > 0 else 0
            mean_R_db = sum([v[1] * v[4] for v in score_dict_db_pool.values()]) / n_true_beats_db if n_true_beats_db > 0 else 0
            mean_F1_db = (2 * mean_P_db * mean_R_db) / (mean_P_db + mean_R_db) if (mean_P_db + mean_R_db) > 0 else 0
            # Beat
            score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(test_pred_b_pool, test_dict, f_measure_threshold=0.07, target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
            score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1
            n_pred_beats_b, n_true_beats_b = sum([v[3] for v in score_dict_b_pool.values()]), sum([v[4] for v in score_dict_b_pool.values()])
            mean_P_b = sum([v[0] * v[3] for v in score_dict_b_pool.values()]) / n_pred_beats_b if n_pred_beats_b > 0 else 0
            mean_R_b = sum([v[1] * v[4] for v in score_dict_b_pool.values()]) / n_true_beats_b if n_true_beats_b > 0 else 0
            mean_F1_b = (2 * mean_P_b * mean_R_b) / (mean_P_b + mean_R_b) if (mean_P_b + mean_R_b) > 0 else 0
            sess.run([mean_loss])
            test_summary, test_loss = sess.run([merged, summary_loss],
                                                 feed_dict={P_db: mean_P_db, R_db: mean_R_db, F1_db: mean_F1_db,
                                                            P_b: mean_P_b, R_b: mean_R_b, F1_b: mean_F1_b})
            test_writer.add_summary(test_summary, epoch)
            sess.run([clr_summary_valid, clr_summary_loss])
            print("---- epoch %d: b_loss: total %.4f, dice(db %.4f, b %.4f), focal %.4f, l2 %.4f, Downbeat: P %.4f R %.4f F1 %.4f, Beat: P %.4f R %.4f F1 %.4f ----"
                % (epoch,
                   test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4],
                   mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b))
            print('db_best =', score_sorted_db_pool[::-1][:3])
            print('db_worst =', score_sorted_db_pool[:3])
            print('b_best =', score_sorted_b_pool[::-1][:3])
            print('b_worst =', score_sorted_b_pool[:3])

            # Sample a testing result
            sample_id = random.randint(0, n_test_samples - 1)
            print('op =', test_dict['recovery_id'][sample_id])
            print('len =', test_dict['len'][sample_id])
            print('mask'.ljust(8), ''.join([' ' if x else 'x' for x in test_mask[sample_id, :display_len]]))
            print('label_db'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in test_y_db[sample_id, :display_len]]))
            print('label_b'.ljust(8), ''.join([str(round(x, 1))[-1] if x != 1 else '1' for x in test_y_b[sample_id, :display_len]]))
            print('pred_db'.ljust(8), ''.join([str(x) for x in test_pred_db[sample_id, :display_len]]))
            print('pred_b'.ljust(8), ''.join([str(x) for x in test_pred_b[sample_id, :display_len]]))
            print('pred_db*'.ljust(8), ''.join([str(x) for x in test_pred_db_pool[sample_id, :display_len]]))
            print('pred_b*'.ljust(8), ''.join([str(x) for x in test_pred_b_pool[sample_id, :display_len]]))
            #
            print('prob_db'.ljust(8), [round(prob,1) for prob, db in zip(test_prob_db[sample_id], test_dict['downbeat'][sample_id]) if db == 1][:50])
            print('prob_b'.ljust(8), [round(prob,1) for prob, b in zip(test_prob_b[sample_id], test_dict['beat'][sample_id]) if b == 1][:50])

            # Print worst case
            worst_case = score_sorted_db_pool[0][0]
            print('worst_case_db =', worst_case)
            print('worst_true_db =', np.round(label_op_dict_db_pool[worst_case][:25], 2))
            print('worst_pred_db =', np.round(pred_op_dict_db_pool[worst_case][:25], 2))
            print('worst_true_b =', np.round(label_op_dict_b_pool[worst_case][:25], 2))
            print('worst_pred_b =', np.round(pred_op_dict_b_pool[worst_case][:25], 2))

            # Check if early stopping
            if (mean_F1_db + mean_F1_b) > sum(best_scores):
                best_scores = [mean_F1_db, mean_F1_b]
                best_epoch = epoch
                in_succession = 0
                performance = (mean_P_db, mean_R_db, mean_F1_db, mean_P_b, mean_R_b, mean_F1_b)
                # Save variables of the model
                print('\n*saving variables...\n')
                saver.save(sess, graph_location + '/symbolic_beat_and_down_beat_estimation_' + dataset +'.ckpt')
            else:
                in_succession += 1
                if in_succession > hp.n_in_succession:
                    print('Early stopping.')
                    break

        elapsed_time = time.time() - startTime
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best validation scores =', np.round(best_scores, 4))
        print("ASAP: Downbeat P %.4f R %.4f F1 %.4f, Beat P %.4f R %.4f F1 %.4f" % performance)


def inference():

    dataset = 'Ballroom'
    valid_fold = 0
    data_path = './ballroom_preprocessed_data_reshape'
    graph_location = './model/audio_beat_and_down_beat_estimation_Ballroom.ckpt'
    _, valid_dict = get_preprocessed_data(data_path, dataset=dataset, valid_fold=valid_fold)

    n_valid_samples = valid_dict['spec'].shape[0]
    print('n_valid_samples =', n_valid_samples)

    with tf.name_scope('placeholder'):
        x_spec = tf.placeholder(tf.float32, [None, hp.sequence_length, 81, 2], name='spec')
        x_len = tf.placeholder(tf.int32, [None], name='valid_lengths')
        y_db = tf.placeholder(tf.int32, [None, hp.sequence_length], name='downbeat')
        y_b = tf.placeholder(tf.int32, [None, hp.sequence_length], name='beat')
        y_hop = tf.placeholder(tf.int32, [None], name='hop_size')
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        # global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        # P_db, R_db, F1_db = tf.placeholder(dtype=tf.float32, name='P_db'), tf.placeholder(dtype=tf.float32, name='R_db'), tf.placeholder(dtype=tf.float32, name='F1_db')
        # P_b, R_b, F1_b = tf.placeholder(dtype=tf.float32, name='P_b'), tf.placeholder(dtype=tf.float32, name='R_b'), tf.placeholder(dtype=tf.float32, name='F1_b')

    with tf.name_scope('model'):
        seq_mask_bool = tf.sequence_mask(x_len, hp.sequence_length, dtype=tf.bool) # Sequence mask
        seq_mask_float = tf.cast(seq_mask_bool, tf.float32)
        h = audio_joint_beat_downbeat_estimation_classTCN(x_spec, x_len, dropout, is_training, hp)

    with tf.name_scope('output_layer'):
        logits = tf.layers.conv1d(h, filters=3, kernel_size=7, padding='same', name='out_conv') # [B, T, 3]
        logits = tf.layers.conv1d(logits, filters=3, kernel_size=1, padding='same', name='out_trans')  # [B, T, 3]
        act = tf.nn.softmax(logits, axis=-1) # [B, T, 3]
        prob_db, prob_xb, _ = tf.unstack(act, axis=-1) # [B, T]
        prob_b = prob_db + prob_xb # [B, T]

    with tf.name_scope('evaluation'):
        # Downbeat
        prob_db = prob_db * seq_mask_float
        # pred_db = tf.cast(tf.greater(prob_db, hp.threshold), tf.int32)
        prob_db_pool = output_probability_pooling(prob_db, pool_size=7) # pooling by local maximums
        pred_db_pool = tf.cast(tf.greater(prob_db_pool, hp.threshold), tf.int32)
        # Beat
        prob_b = prob_b * seq_mask_float
        # pred_b = tf.cast(tf.greater(prob_b, hp.threshold), tf.int32)
        prob_b_pool = output_probability_pooling(prob_b, pool_size=7) # pooling by local maximums
        pred_b_pool = tf.cast(tf.greater(prob_b_pool, hp.threshold), tf.int32)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path=graph_location)

        # Validation
        valid_pred_db_pool_list, valid_pred_b_pool_list = [], []
        valid_mask_list = []
        valid_y_db_list, valid_y_b_list = [], []
        valid_batch_size = hp.batch_size
        for i in range(0, n_valid_samples, valid_batch_size):
            valid_run_list = [pred_db_pool, pred_b_pool, seq_mask_bool, y_db, y_b]
            valid_feed_fict = {x_spec: valid_dict['spec'][i:i + valid_batch_size],
                               x_len: valid_dict['len'][i:i + valid_batch_size],
                               y_db: valid_dict['downbeat'][i:i + valid_batch_size],
                               y_b: valid_dict['beat'][i:i + valid_batch_size],
                               y_hop: valid_dict['hop'][i:i + valid_batch_size],
                               dropout: 0.0,
                               is_training: False}
            valid_pred_db_pool, valid_pred_b_pool, valid_mask, valid_y_db, valid_y_b = sess.run(valid_run_list, feed_dict=valid_feed_fict)
            # Store batch-wise results
            valid_pred_db_pool_list.append(valid_pred_db_pool)
            valid_pred_b_pool_list.append(valid_pred_b_pool)
            valid_mask_list.append(valid_mask)
            valid_y_db_list.append(valid_y_db)
            valid_y_b_list.append(valid_y_b)
        valid_pred_db_pool = np.concatenate(valid_pred_db_pool_list, axis=0)
        valid_pred_b_pool = np.concatenate(valid_pred_b_pool_list, axis=0)
        valid_mask = np.concatenate(valid_mask_list, axis=0)
        valid_y_db = np.concatenate(valid_y_db_list, axis=0)
        valid_y_b = np.concatenate(valid_y_b_list, axis=0)

        # Evaluation: Downbeat
        score_dict_db_pool, label_op_dict_db_pool, pred_op_dict_db_pool = beat_eval(valid_pred_db_pool, valid_dict,
                                                                                    f_measure_threshold=0.07,
                                                                                    target='downbeat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
        score_sorted_db_pool = sorted(score_dict_db_pool.items(), key=lambda item: item[1][2]) # sorted by F1
        # Evaluation: Beat
        score_dict_b_pool, label_op_dict_b_pool, pred_op_dict_b_pool = beat_eval(valid_pred_b_pool, valid_dict,
                                                                                 f_measure_threshold=0.07,
                                                                                 target='beat') # {key: (P, R, F1, n_pred_beats, n_true_beats)...}
        score_sorted_b_pool = sorted(score_dict_b_pool.items(), key=lambda item: item[1][2]) # sorted by F1


        # Print evaluation results
        db_mean_F1 = np.mean([s[1][2] for s in score_sorted_db_pool])
        b_mean_F1 = np.mean([s[1][2] for s in score_sorted_b_pool])
        print('averaged F1 for downbeat %.4f' % db_mean_F1)
        print('averaged F1 for beat %.4f' % b_mean_F1)
        print()
        # Print best and worst cases
        np.set_printoptions(linewidth=150)
        best_case = score_sorted_db_pool[-1][0]
        print('best case: %s, Downbeat F1 %.4f, Beat F1 %.4f' %
              (best_case[:-23], score_dict_db_pool[best_case][2], score_dict_b_pool[best_case][2]))
        print('best_true_db =', np.round(label_op_dict_db_pool[best_case], 2))
        print('best_pred_db =', np.round(pred_op_dict_db_pool[best_case], 2))
        print('best_true_b =', np.round(label_op_dict_b_pool[best_case], 2))
        print('best_pred_b =', np.round(pred_op_dict_b_pool[best_case], 2))
        print()
        worst_case = score_sorted_db_pool[0][0]
        print('worst case: %s, Downbeat F1 %.4f, Beat F1 %.4f' %
              (worst_case[:-23], score_dict_db_pool[worst_case][2], score_dict_b_pool[worst_case][2]))
        print('worst_true_db =', np.round(label_op_dict_db_pool[worst_case], 2))
        print('worst_pred_db =', np.round(pred_op_dict_db_pool[worst_case], 2))
        print('worst_true_b =', np.round(label_op_dict_b_pool[worst_case], 2))
        print('worst_pred_b =', np.round(pred_op_dict_b_pool[worst_case], 2))


if __name__ == '__main__':

    hyperparameters = namedtuple('hyperparameters',
                                 ['sequence_length',
                                  'n_filters',
                                  'hidden_size',
                                  'batch_size',
                                  'drop',
                                  'threshold',
                                  'n_in_succession',
                                  'n_training_epochs',
                                  'initial_learning_rate',
                                  'evaluation_tolerance_window',
                                  'activation'])

    hp = hyperparameters(sequence_length=2048,
                         n_filters=20,
                         hidden_size=20,
                         batch_size=64,
                         drop=0.1,
                         threshold=0.5,
                         n_in_succession=10,
                         n_training_epochs=100,
                         initial_learning_rate=1e-3,
                         evaluation_tolerance_window=70,
                         activation=tf.nn.elu)

    # # Train the baseline model
    # train_audio_beat_downbeat_estimation_baseline()

    # # Train the proposed model in audio domain
    # train_audio_beat_downbeat_estimation()

    # # Train the proposed model in symbolic domain
    # train_symbolic_beat_downbeat_estimation()


    # Inference with the pretrained model
    inference()



