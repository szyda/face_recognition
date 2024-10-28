import os
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class FaceRecognition:
    def __init__(self, input_shape=(224, 224, 3), learning_rate=1e-4, dropout_rate=0.3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # pretrained weights are not updated during training,
        # preserving the learned features and reducing computational overhead.
        # base_model.trainable = False

        pooled_output = base_model.output
        pooled_output = GlobalAveragePooling2D()(pooled_output)

        feature_extractor = Model(inputs=base_model.input, outputs=pooled_output, name='Feature_Extractor')

        input_image1 = Input(shape=self.input_shape, name='Image_Input_1')
        input_image2 = Input(shape=self.input_shape, name='Image_Input_2')

        features_image1 = feature_extractor(input_image1)
        features_image2 = feature_extractor(input_image2)

        # difference = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='Feature_Difference')([features_image1, features_image2])

        # euclidean distance
        difference = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1,keepdims=True)), name='Euclidean_Distance')([features_image1, features_image2])

        similarity_score = Dense(1, activation='sigmoid', name='Similarity_Output')(difference)

        model = Model(inputs=[input_image1, input_image2], outputs=similarity_score, name='Simplified_Siamese_Network')
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()

        return model

    def get_callbacks(self, log_dir='logs/fit'):
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

        checkpoint = ModelCheckpoint(
            'model.weights.h5',
            monitor='val_accuracy',
            save_weights_only=True,
            verbose=1
        )

        return [tensorboard, checkpoint]

    def train(self, model, train_generator, val_generator, epochs=50):
        callbacks = self.get_callbacks()

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def save_model(self, filepath='celebs-1000.weights.h5'):
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")

    # def load_model(self, filepath='models/best_model.h5'):
    #     from tensorflow.keras.models import load_model
    #     self.model = load_model(filepath, compile=False)
    #
    #     optimizer = Adam(learning_rate=self.learning_rate)
    #     self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    #     print(f"Model loaded from {filepath}")
