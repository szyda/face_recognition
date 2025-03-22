import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.keras.backend as K
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import boto3


class FaceRecognition:
    def __init__(self, input_shape=(224, 224, 3), learning_rate=0.0001, dropout_rate=0.3, file_path='model.weights.h5'):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.feature_extractor = self.build_feature_extractor()
        self.model = self.build_model()
        self.file_path = file_path

        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.s3_file_path = os.getenv("FILE_PATH")
        self.local_file_path = " "

        self.load_weights()


    def build_feature_extractor(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)

        for layer in base_model.layers[:-4]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
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
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def get_callbacks(self, log_dir='logs/fit'):
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

        checkpoint = ModelCheckpoint(
            self.file_path,
            monitor='val_loss',
            save_weights_only=True,
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        return [tensorboard, checkpoint]

    def train(self, train_generator, val_generator, epochs=30):
        callbacks = self.get_callbacks()
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def save_model(self):
        self.model.save_weights(self.file_path)
        print(f"Weights saved to {self.file_path}")

    def extract_embedding(self, image):
        embedding = self.feature_extractor.predict(image)
        return embedding.flatten()

    def load_weights(self):
        if not os.path.exists(self.local_file_path):
            print("Weights not available locally.")

        else:
            self.model.load_weights(self.local_file_path)
            print(f"Weights loaded from: {self.local_file_path}")

    def evaluate(self, validation_generator):
        y_true = []
        y_pred_scores = []

        for batch in validation_generator:
            (img1_batch, img2_batch), labels = batch

            if len(labels) == 0 or img1_batch.size == 0 or img2_batch.size == 0:
                continue

            predictions = self.model.predict([img1_batch, img2_batch]).flatten()
            y_true.extend(labels)
            y_pred_scores.extend(predictions)

        y_true = np.array(y_true)
        y_pred_scores = np.array(y_pred_scores)

        thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
        for threshold in thresholds:
            y_pred = (y_pred_scores >= threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred)
            print(f"\nThreshold: {threshold}")
            print("Confusion Matrix:")
            print(cm)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dissimilar", "Similar"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix at Threshold {threshold}")
            plt.savefig(f'confusion_matrix_threshold_{threshold}.png')
            plt.show()
            plt.close()

            print("\nClassification Report:")
            print(
                classification_report(
                    y_true, y_pred, target_names=["Dissimilar", "Similar"]
                )
            )
