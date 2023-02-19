import numpy as np
import tensorflow as tf
## function for using mixup data augmentation
def data_mixup_3_labels(dataset, alpha=0.2, batch_size = 32):
    '''
    #provide_a tf dataset (x,y) with batch size.
    '''
    ds_one = dataset.shuffle(batch_size * 100)
    ds_two = dataset.shuffle(batch_size * 100)
    
    train_ds = tf.data.Dataset.zip((ds_one, ds_two))
    
    def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)
    
    def mix_up(ds_one, ds_two, alpha):
        # Unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = tf.shape(images_one)[0]
#         print(ds_one)

        # Sample lambda and reshape it to do the mixup
        l = sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels_0 = labels_one[0] * y_l + labels_two[0] * (1 - y_l)
        labels_1 = labels_one[1] * y_l + labels_two[1] * (1 - y_l)
        labels_2 = labels_one[2] * y_l + labels_two[2] * (1 - y_l)
        return (images, (labels_0,labels_1,labels_2))

    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha), num_parallel_calls=tf.data.AUTOTUNE
    )
    
    
    return train_ds_mu
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
        
class MixupGenerator_2level():
    def __init__(self, X_train, y_train_c, y_train_f, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train_c = y_train_c
        self.y_train_f = y_train_f
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y_c, y_f = self.__data_generation(batch_ids)

                yield (X, [y_c, y_f])

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train_c, list):
            y_c = []

            for y_train_c_ in self.y_train_c:
                y1c = y_train_c_[batch_ids[:self.batch_size]]
                y2c = y_train_c_[batch_ids[self.batch_size:]]
                y_c.append(y1c * y_l + y2c * (1 - y_l))

        if isinstance(self.y_train_f, list):
            y_f = []

            for y_train_f_ in self.y_train_f:
                y1f = y_train_f_[batch_ids[:self.batch_size]]
                y2f = y_train_f_[batch_ids[self.batch_size:]]
                y_f.append(y1f * y_l + y2f * (1 - y_l))
        else:
            y1c = self.y_train_c[batch_ids[:self.batch_size]]
            y1f = self.y_train_f[batch_ids[:self.batch_size]]
            
            y2c = self.y_train_c[batch_ids[self.batch_size:]]
            y2f = self.y_train_f[batch_ids[self.batch_size:]]
            
            y_c = y1c * y_l + y2c * (1 - y_l)
            y_f = y1f * y_l + y2f * (1 - y_l)

        return X, y_c, y_f
        
class MixupGenerator_3level():
    def __init__(self, X_train, y_train_c1, y_train_c2, y_train_f, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train_c1 = y_train_c1
        self.y_train_c2 = y_train_c2
        self.y_train_f = y_train_f
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y_c1, y_c2, y_f = self.__data_generation(batch_ids)

                yield (X, [y_c1, y_c2, y_f])

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train_c1, list):
            y_c1 = []

            for y_train_c1_ in self.y_train_c1:
                y1c = y_train_c1_[batch_ids[:self.batch_size]]
                y2c = y_train_c1_[batch_ids[self.batch_size:]]
                y_c1.append(y1c * y_l + y2c * (1 - y_l))
                
        if isinstance(self.y_train_c2, list):
            y_c2 = []

            for y_train_c2_ in self.y_train_c2:
                y1c2 = y_train_c2_[batch_ids[:self.batch_size]]
                y2c2 = y_train_c2_[batch_ids[self.batch_size:]]
                y_c2.append(y1c2 * y_l + y2c2 * (1 - y_l))

        if isinstance(self.y_train_f, list):
            y_f = []

            for y_train_f_ in self.y_train_f:
                y1f = y_train_f_[batch_ids[:self.batch_size]]
                y2f = y_train_f_[batch_ids[self.batch_size:]]
                y_f.append(y1f * y_l + y2f * (1 - y_l))
        else:
            y1c = self.y_train_c1[batch_ids[:self.batch_size]]
            y1c2 = self.y_train_c2[batch_ids[:self.batch_size]]
            y1f = self.y_train_f[batch_ids[:self.batch_size]]
            
            y2c = self.y_train_c1[batch_ids[self.batch_size:]]
            y2c2 = self.y_train_c2[batch_ids[self.batch_size:]]
            y2f = self.y_train_f[batch_ids[self.batch_size:]]
            
            y_c1 = y1c * y_l + y2c * (1 - y_l)
            y_c2 = y1c2 * y_l + y2c2 * (1 - y_l)
            y_f = y1f * y_l + y2f * (1 - y_l)

        return X, y_c1, y_c2, y_f