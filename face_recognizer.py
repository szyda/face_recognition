import os
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from data_processor import DataProcessor


class FaceRecognition:
    def __init__(self, input_shape=(224, 224, 3), learning_rate=0.0001, dropout_rate=0.3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.feature_extractor = self.build_feature_extractor()
        self.model = self.build_model()

    def build_feature_extractor(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)

        for layer in base_model.layers[:-3]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Dropout(self.dropout_rate)(x)
        feature_extractor = Model(inputs=base_model.input, outputs=x)

        return feature_extractor

    def build_model(self):
        input_image1 = Input(shape=self.input_shape)
        input_image2 = Input(shape=self.input_shape)

        features_image1 = self.feature_extractor(input_image1)
        features_image2 = self.feature_extractor(input_image2)

        l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([features_image1, features_image2])
        similarity_score = Dense(1, activation='sigmoid')(l1_distance)

        model = Model(inputs=[input_image1, input_image2], outputs=similarity_score)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')])

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

        return [tensorboard, checkpoint]

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

    def save_features(self, known_images_dir='dataset', features_path='known_features.npy',
                      labels_filepath='known_labels.npy'):
        known_features = []
        known_labels = []

        for person in os.listdir(known_images_dir):
            person_dir = os.path.join(known_images_dir, person)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    if os.path.isfile(img_path):
                        try:
                            img = DataProcessor.preprocess_image(img_path)
                            img = np.expand_dims(img, axis=0)

                            feature = self.feature_extractor.predict(img)
                            known_features.append(feature.flatten())
                            known_labels.append(person)

                            print(f"Processed image {img_path}")

                        except Exception as e:
                            print(f"Failed to process image {img_path}: {e}")

        known_features = np.array(known_features)

        np.save(features_path, known_features)
        np.save(labels_filepath, known_labels)
        print(f"Features saved to {features_path} and labels saved to {labels_filepath}")

    def save_model(self, filepath='celebs-500.weights.h5'):
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")
