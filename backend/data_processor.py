import os
import cv2
import numpy as np
import random
from itertools import combinations
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

class DataProcessor(Sequence):
    def __init__(self,
                 identity_to_images,
                 identities,
                 image_size=(224, 224),
                 batch_size=32,
                 num_pairs_per_identity=60,
                 augment=False,
                 shuffle=True,
                 mode='train',
                 **kwargs):

        super().__init__(**kwargs)
        self.mode = mode
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_pairs_per_identity = num_pairs_per_identity
        self.augment = augment if mode == 'train' else False
        self.shuffle = shuffle

        self.identity_to_images = {identity: identity_to_images[identity] for identity in identities}
        self.pairs, self.labels = self._generate_pairs(identities)
        self.indices = np.arange(len(self.pairs))

        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            self.datagen = None

    @staticmethod
    def preprocess_image(image_input, image_size=(224, 224)):
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to read image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Input must be a file path or an image array.")

        image = cv2.resize(image, image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)

        return image

    @staticmethod
    def load_data(data_directory, num_identities_to_use=None, num_images_per_identity=None):
        identity_to_images = {}
        identities = sorted(os.listdir(data_directory))

        # choose the number of identites (debug and development mode)
        if num_identities_to_use:
            identities = identities[:num_identities_to_use]

        for identity in identities:
            identity_dir = os.path.join(data_directory, identity)
            if os.path.isdir(identity_dir):
                img_files = [os.path.join(identity_dir, f)
                             for f in os.listdir(identity_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(img_files) >= 2:
                    img_files = sorted(img_files)
                    if num_images_per_identity:
                        img_files = img_files[:num_images_per_identity]
                    identity_to_images[identity] = img_files

        return identity_to_images

    @staticmethod
    def split_identities(identities, validation_split):
        identities = list(identities)
        identities.sort()
        num_train = int(len(identities) * (1 - validation_split))
        train_identities = identities[:num_train]
        val_identities = identities[num_train:]

        return train_identities, val_identities

    def _generate_pairs(self, identities):
        positive_pairs = []
        negative_pairs = []
        identity_to_images = {identity: self.identity_to_images[identity] for identity in identities}
        all_identities = list(identity_to_images.keys())

        for identity, images in identity_to_images.items():
            if len(images) < 2:
                continue
            possible_pairs = list(combinations(images, 2))
            possible_pairs = sorted(possible_pairs)
            random.shuffle(possible_pairs)
            selected_pairs = possible_pairs[:min(self.num_pairs_per_identity, len(possible_pairs))]
            positive_pairs.extend(selected_pairs)

        num_positive = len(positive_pairs)
        if num_positive == 0:
            print("Warning: No positive pairs generated.")
            return [], []

        while len(negative_pairs) < num_positive:
            id1, id2 = random.sample(all_identities, 2)
            if id1 == id2:
                continue
            img1 = random.choice(identity_to_images[id1])
            img2 = random.choice(identity_to_images[id2])
            negative_pairs.append((img1, img2))

        pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        combined = list(zip(pairs, labels))
        combined = sorted(combined)
        if self.shuffle:
            random.shuffle(combined)

        pairs[:], labels[:] = zip(*combined)

        set_type = "training" if self.mode == 'train' else "validation"
        print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs for {set_type}.")

        return list(pairs), list(labels)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # debug
        if index >= self.__len__():
            print(f"Index {index} is out of range. Total batches: {self.__len__()}")
            raise IndexError("Index out of range")

        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_pairs = [self.pairs[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        img1_batch = []
        img2_batch = []
        valid_labels = []

        for (img1_path, img2_path), label in zip(batch_pairs, batch_labels):
            try:
                img1 = self.preprocess_image(img1_path, self.image_size)
                img2 = self.preprocess_image(img2_path, self.image_size)
            except ValueError as e:
                print(e)
                continue

            if self.augment and self.datagen:
                img1 = self.datagen.random_transform(img1)
                img2 = self.datagen.random_transform(img2)

            img1_batch.append(img1)
            img2_batch.append(img2)
            valid_labels.append(label)

        img1_batch = np.array(img1_batch)
        img2_batch = np.array(img2_batch)
        batch_labels_array = np.array(valid_labels)

        return (img1_batch, img2_batch), batch_labels_array

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    @staticmethod
    def crop_face(image, image_path=None):
        if image is None:
            raise ValueError(f"No image at {image}")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            print("No faces detected")
            return None

        x, y, w, h = faces[0]
        face = image[y:y + h, x:x + w]

        return face
