import os
import warnings

import pickle
import random

import tensorflow as tf
import numpy as np
import pandas as pd

from globals import *
from functions import *



def generate_TFRecords(save_dir):

    np.random.seed(SEED)

    X = []
    y = []
    y_idx = []

    # generate some data
    idx = pd.date_range(start = '2019-01-01 00:00:00',
                        end = '2019-03-30 00:00:00',
                        freq = '1min')
    data = np.random.rand(len(idx),9)
    df = pd.DataFrame(data=data,
                      index=idx,
                      columns=['Feature1', 'Feature2', 'Feature3', 'Feature4',
                               'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Target'])


    # dataframe -> numpy array
    x_ts = df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8']].values
    y_ts = df['Target'].values
    idx = idx.tolist()



    # tf.keras implementation of RNNs requires input features of shape (SEQUENCE_LENGTH, N_FEATURES)
    # the target will be the t+1 value of the 'Target' column.
    for sample_n in range(len(df) - SEQUENCE_LENGTH - 1):

        print('Generating sample %i/%i' % (sample_n, len(df) - SEQUENCE_LENGTH - 1))

        X.append(x_ts[sample_n:sample_n + SEQUENCE_LENGTH, :])   # append the sample's feature to X list
        y.append(y_ts[sample_n + SEQUENCE_LENGTH])           # append the sample's target to y list

        y_idx.append(idx[sample_n + SEQUENCE_LENGTH])        # append the sample's index to y_idx list



    # tensorflow documentation for TFRecord suggests to divide the whole dataset in shards with size between 100MB-200MB
    byte_per_sample = X[0].nbytes
    byte_per_shard_max = 100000000 # 100MB
    n_samples_per_shard_max = int(np.floor(byte_per_shard_max / byte_per_sample))

    # each shard will contain an integer number of batches
    n_batch_per_shard = int(np.floor(n_samples_per_shard_max / BATCH_SIZE))
    n_samples_per_shard = n_batch_per_shard * BATCH_SIZE

    n_samples_tot_raw = len(X)
    n_shards_tot = int(np.floor(n_samples_tot_raw / n_samples_per_shard)) # a few samples will be discarded...
    n_samples_tot = n_shards_tot * n_samples_per_shard

    # isolate the test set (10% of n_shards_tot) before shuffling. In post processing, plotting a non shuffled time
    # series prediction fosters interpretability
    n_shards_test = np.max((int(np.floor(n_shards_tot * 0.1)),1))
    n_samples_test = n_shards_test * n_samples_per_shard



    # save tfrecords on disk. The last shard will have less samples than the others so it'll be discarded
    print('Saving test set...')

    start_sample_shard = n_samples_tot - n_samples_test

    for shard_n in range(n_shards_tot - n_shards_test, n_shards_tot):

        # first, save the target's date
        tempYidx = y_idx[start_sample_shard:start_sample_shard + n_samples_per_shard]

        filename = 'test_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot - 1).zfill(7) + 'Yidx.pkl'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(tempYidx, f)

        # select the appropriate number of samples to be saved in the shard file
        tempX = np.array(X[start_sample_shard:start_sample_shard + n_samples_per_shard])
        tempY = np.array(y[start_sample_shard:start_sample_shard + n_samples_per_shard])

        if DATASET_FORMAT == 'numpy':
            filename = 'test_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot-1).zfill(7) + '_X_rnn.npy'
            # (why n_shards_tot-1? because we discard the last shard)
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            np.save(filepath, tempX)

            filename = 'test_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot-1).zfill(7) + 'Y.npy'
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            np.save(filepath, tempY)

        elif DATASET_FORMAT == 'tfrecord':

            filename = 'test_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot-1).zfill(7) + '.tfrecord'
            filepath = os.path.join(save_dir, filename)
            print(filepath)

            # define the writer
            with tf.io.TFRecordWriter(filepath) as writer:

                for i, (X_sample, Y_sample) in enumerate(zip(tempX, tempY)):
                    # each sample is first serialized in Google's Protocol Buffer...
                    example = serialize_example(X_sample[:, 0],
                                                X_sample[:, 1],
                                                X_sample[:, 2],
                                                X_sample[:, 3],
                                                X_sample[:, 4],
                                                X_sample[:, 5],
                                                X_sample[:, 6],
                                                X_sample[:, 7],
                                                Y_sample)
                    # ... and then added to the shard file
                    writer.write(example)

        start_sample_shard += n_samples_per_shard



    print('Shuffling train/vali set')
    # shuffle
    test_split = n_samples_tot - n_samples_test
    X = X[:test_split] # test data are already saved to disk, we can do this without problem
    y = y[:test_split]
    y_idx = y_idx[:test_split]

    temp = list(zip(X,y,y_idx))
    random.shuffle(temp)
    X, y, y_idx = zip(*temp)
    del temp



    # the procedure of saving training and validation datasets to .tfrecord files is the same followed for the test set
    print('Saving train set...')

    n_shards_vali = n_shards_test
    start_sample_shard = 0

    for shard_n in range(n_shards_tot - n_shards_test - n_shards_vali):

        tempYidx = y_idx[start_sample_shard:start_sample_shard + n_samples_per_shard]
        filename = 'train_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot - 1).zfill(7) + 'Yidx.pkl'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(tempYidx, f)

        tempX = np.array(X[start_sample_shard:start_sample_shard + n_samples_per_shard])
        tempY = np.array(y[start_sample_shard:start_sample_shard + n_samples_per_shard])

        if DATASET_FORMAT == 'numpy':
            filename = 'train_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot-1).zfill(7) + '_X_rnn.npy'
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            np.save(filepath, tempX)

            filename = 'train_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot-1).zfill(7) + 'Y.npy'
            filepath = os.path.join(save_dir,filename)
            print(filepath)
            np.save(filepath, tempY)

        elif DATASET_FORMAT == 'tfrecord':

            filename = 'train_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot-1).zfill(7) + '.tfrecord'
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            with tf.io.TFRecordWriter(filepath) as writer:
                for i, (X_sample, Y_sample) in enumerate(zip(tempX, tempY)):

                    example = serialize_example(X_sample[:, 0],
                                                X_sample[:, 1],
                                                X_sample[:, 2],
                                                X_sample[:, 3],
                                                X_sample[:, 4],
                                                X_sample[:, 5],
                                                X_sample[:, 6],
                                                X_sample[:, 7],
                                                Y_sample)
                    writer.write(example)

        start_sample_shard += n_samples_per_shard



    # The validation set will have the same size of the test set
    print('Saving vali set...')
    start_sample_shard = n_shards_tot - n_shards_test - n_shards_vali

    for shard_n in range(n_shards_tot - n_shards_test - n_shards_vali, n_shards_tot - n_shards_test):

        tempYidx = y_idx[start_sample_shard:start_sample_shard + n_samples_per_shard]
        filename = 'vali_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot - 1).zfill(7) + 'Yidx.pkl'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(tempYidx, f)

        tempX = np.array(X[start_sample_shard:start_sample_shard + n_samples_per_shard])
        tempY = np.array(y[start_sample_shard:start_sample_shard + n_samples_per_shard])

        if DATASET_FORMAT == 'numpy':

            filename = 'vali_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot - 1).zfill(7) + '_X_rnn.npy'
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            np.save(filepath, tempX)

            filename = 'vali_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot - 1).zfill(7) + 'Y.npy'
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            np.save(filepath, tempY)

        elif DATASET_FORMAT == 'tfrecord':
            filename = 'vali_' + str(shard_n).zfill(7) + '-of-' + str(n_shards_tot - 1).zfill(7) + '.tfrecord'
            filepath = os.path.join(save_dir, filename)
            print(filepath)
            with tf.io.TFRecordWriter(filepath) as writer:
                for i, (X_sample, Y_sample) in enumerate(zip(tempX, tempY)):

                    example = serialize_example(X_sample[:, 0],
                                                X_sample[:, 1],
                                                X_sample[:, 2],
                                                X_sample[:, 3],
                                                X_sample[:, 4],
                                                X_sample[:, 5],
                                                X_sample[:, 6],
                                                X_sample[:, 7],
                                                Y_sample)
                    writer.write(example)

        start_sample_shard += n_samples_per_shard


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    generate_TFRecords(save_dir)

    if DEBUG_RNN:

        # build a simple model just to check that everything works
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.GRU(10, return_sequences = False, activation = 'sigmoid'))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer='adam', loss='mse')

        if DATASET_FORMAT == 'tfrecord':

            files = tf.io.matching_files(os.path.join(save_dir, '*.tfrecord'))
            dataset = tf.data.TFRecordDataset(files)
            dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
            dataset = dataset.map(parse_fn)
            dataset = dataset.batch(BATCH_SIZE)

            a = []
            b = []

            # to actually "see" a batch of saved data, load check_features.npy and check_targets.npy
            for batch in dataset.take(BATCH_SIZE):

                a.append(batch[0].numpy())
                b.append(batch[1].numpy())

            np.save(os.path.join(save_dir, 'check_features.npy'), np.array(a))
            np.save(os.path.join(save_dir, 'check_targets.npy'), np.array(b))

            model.fit(x = dataset,
                      epochs = 2)

        else:
            x = np.load(os.path.join(save_dir, '0000000-of-0000000_X_rnn.npy'))
            y = np.load(os.path.join(save_dir, '0000000-of-0000000_Y.npy'))

            model.fit(x = x, y = y,
                      epochs = 2)




