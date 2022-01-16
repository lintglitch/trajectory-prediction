# modified from https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

import os
import keras
import tensorflow as tf
import numpy as np
import time

class InceptionTime():
    def __init__(self, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.bottleneck_size = 32

        # settings init
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        self.regularizer = None


    def build(self, input_shape, number_output_classes):
        model = self._build_model(input_shape, number_output_classes)
        model.summary()
        return model


    # def fit(self, x_train, y_train, x_val, y_val, name='model'):
    #     if not tf.config.list_physical_devices('GPU'):
    #         print('Warning: No GPU found. Training without GPU acceleration.')

    #     start_time = time.time()

    #     hist = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epochs, shuffle=True,
    #                             verbose=True, validation_data=(x_val, y_val), callbacks=self.callbacks)

    #     duration = time.time() - start_time
    #     print(f"Training took {duration}s")

    #     # save the model
    #     self.model.save( os.path.join(self.output_directory, name) )
    #     keras.backend.clear_session()
    #     return hist


    # def predict_once(self, data):
    #     """
    #     Gives N predictions using the trained model. Input should be numpy array of (N, length, obj_info)
    #     """
    #     if not self.model:
    #         return None

    #     return self.model(data)


    # def evaluate(self, x, y):
    #     """
    #     Gives evaluation of the model for the given x and y.

    #     Returns:
    #         results - list of scalar losses over all classes
    #         confustion_matrix - 
    #     """
    #     results = self.model.evaluate(x, y, self.batch_size)
    #     #get predictions
    #     preds = self.model.predict(x)
    #     #convert from one hot encoding to lists with class number
    #     y = [np.argmax(x) for x in y]
    #     preds = [np.argmax(x) for x in preds]
    #     #get confusion matrix
    #     confusion_matrix = tf.math.confusion_matrix(y, preds)

    #     return results, confusion_matrix


    # def plot_structure(self, filename, dpi=96):
    #     """
    #     Plots the model structure and saves it to the given filename
    #     """
    #     self.model.summary()
    #     keras.utils.plot_model(self.model, dpi=96, to_file=filename)


    def _inception_module(self, input_tensor, stride=1, activation='linear'):



        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False, activity_regularizer=self.regularizer)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False, activity_regularizer=self.regularizer)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same', activity_regularizer=self.regularizer)(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False, activity_regularizer=self.regularizer)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x


    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False, activity_regularizer=self.regularizer)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x


    def _build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
        #               metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])

        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
        #                                               min_lr=0.0001)
        # file_path = self.output_directory + 'best_model.hdf5'
        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                    save_best_only=True)
        # self.callbacks = [reduce_lr, model_checkpoint]

        return model