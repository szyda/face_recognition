import os
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
import tensorflow as tf


class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator):
        super(F1ScoreCallback, self).__init__()
        self.val_generator = val_generator

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.val_generator)
        y_pred = (val_pred.flatten() > 0.5).astype(int)

        y_true = []
        for i in range(len(self.val_generator)):
            _, label = self.val_generator[i]
            y_true.extend(label.flatten())

        f1 = f1_score(y_true, y_pred)
        print(f"\nEpoch {epoch + 1}: Validation F1 Score: {f1:.4f}")

class FaceRecognition:
    def __init__(self, input_shape=(224, 224, 3), learning_rate=0.0001, dropout_rate=0.35):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # eksperyment
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        pooled_output = base_model.output
        pooled_output = GlobalAveragePooling2D()(pooled_output)
        pooled_output = Dense(256, activation='relu')(pooled_output)
        pooled_output = Dropout(self.dropout_rate)(pooled_output)

        feature_extractor = Model(inputs=base_model.input, outputs=pooled_output)

        input_image1 = Input(shape=self.input_shape)
        input_image2 = Input(shape=self.input_shape)

        features_image1 = feature_extractor(input_image1)
        features_image2 = feature_extractor(input_image2)

        features_image1 = Lambda(lambda x: K.l2_normalize(x, axis=1))(features_image1)
        features_image2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(features_image2)

        distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([features_image1, features_image2])
        similarity_score = Dense(1, activation='sigmoid')(distance)

        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model = Model(inputs=[input_image1, input_image2], outputs=similarity_score)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

        model.summary()

        return model

    def get_callbacks(self, val_generator, log_dir='logs/fit'):
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'model.weights.h5',
            monitor='val_loss',
            save_weights_only=True,
            verbose=1,
            save_best_only=True,
            mode='min'
        )

        f1_callback = F1ScoreCallback(val_generator=val_generator)

        return [tensorboard, checkpoint, f1_callback]

    def train(self, model, train_generator, val_generator, epochs=30):
        callbacks = self.get_callbacks(val_generator=val_generator)

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def save_model(self, filepath='celebs-500.weights.h5'):
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")
