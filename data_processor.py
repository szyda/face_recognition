import os
import cv2
import numpy as np
import random
from itertools import combinations
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


class DataProcessor(Sequence):
    _identity_to_images = None
    _train_identities = None
    _val_identities = None

    def __init__(self,
                 data_directory,
                 image_size=(224, 224),
                 batch_size=32,
                 num_pairs_per_identity=60,
                 validation_split=0.3,
                 augment=False,
                 shuffle=True,
                 mode='train',
                 num_identities_to_use=None,
                 num_images_per_identity=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.mode = mode
        self.data_directory = data_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_pairs_per_identity = num_pairs_per_identity
        self.validation_split = validation_split
        self.augment = augment if mode == 'train' else False
        self.shuffle = shuffle
        self.num_identities_to_use = num_identities_to_use
        self.num_images_per_identity = num_images_per_identity

        if DataProcessor._identity_to_images is None:
            DataProcessor._identity_to_images = self._load_data()
            DataProcessor._train_identities, DataProcessor._val_identities = self._split_identities()

        self.identity_to_images = DataProcessor._identity_to_images

        if self.mode == 'train':
            identities = DataProcessor._train_identities
        else:
            identities = DataProcessor._val_identities

        self.identity_to_images = {identity: self.identity_to_images[identity] for identity in identities}
        self.pairs, self.labels = self._generate_pairs(identities, training=self.mode == 'train')

        self.indices = np.arange(len(self.pairs))

        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            self.datagen = None

        self.on_epoch_end()

    @staticmethod
    def preprocess_image(image_path, image_size=(224, 224)):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = preprocess_input(image)
        return image

    def _load_data(self):
        identity_to_images = {}
        identities = os.listdir(self.data_directory)

        if self.num_identities_to_use:
            identities = identities[:self.num_identities_to_use]

        for identity in identities:
            identity_dir = os.path.join(self.data_directory, identity)
            if os.path.isdir(identity_dir):
                img_files = [os.path.join(identity_dir, f)
                             for f in os.listdir(identity_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(img_files) >= 2:
                    identity_to_images[identity] = img_files[:self.num_images_per_identity]
        print(f"Loaded {len(identity_to_images)} identities from {self.data_directory}.")

        return identity_to_images

    def _split_identities(self):
        # Use the class-level _identity_to_images instead of the instance-level one
        identities = list(DataProcessor._identity_to_images.keys())
        random.shuffle(identities)
        num_train = int(len(identities) * (1 - self.validation_split))
        train_identities = identities[:num_train]
        val_identities = identities[num_train:]
        print(f"Split into {len(train_identities)} training and {len(val_identities)} validation identities.")

        return train_identities, val_identities

    def _generate_pairs(self, identities, training=True):
        positive_pairs = []
        negative_pairs = []
        identity_to_images = {identity: self.identity_to_images[identity] for identity in identities}
        all_identities = list(identity_to_images.keys())

        for identity, images in identity_to_images.items():
            possible_pairs = list(combinations(images, 2))
            random.shuffle(possible_pairs)
            selected_pairs = possible_pairs[:min(self.num_pairs_per_identity, len(possible_pairs))]
            positive_pairs.extend(selected_pairs)

        num_positive = len(positive_pairs)
        while len(negative_pairs) < num_positive:
            id1, id2 = random.sample(all_identities, 2)
            img1 = random.choice(identity_to_images[id1])
            img2 = random.choice(identity_to_images[id2])
            negative_pairs.append((img1, img2))

        pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        set_type = "training" if training else "validation"
        print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs for {set_type}.")

        return list(pairs), list(labels)

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_pairs = [self.pairs[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        img1_batch = []
        img2_batch = []
        valid_labels = []

        for (img1_path, img2_path), label in zip(batch_pairs, batch_labels):
            try:
                img1 = DataProcessor.preprocess_image(img1_path, self.image_size)
                img2 = DataProcessor.preprocess_image(img2_path, self.image_size)
            except ValueError as e:
                print(e)
                continue

            if self.augment and self.datagen:
                seed = random.randint(0, 100000)
                img1 = self.datagen.random_transform(img1, seed=seed)
                img2 = self.datagen.random_transform(img2, seed=seed)

            img1_batch.append(img1)
            img2_batch.append(img2)
            valid_labels.append(label)

        img1_batch = np.array(img1_batch)
        img2_batch = np.array(img2_batch)
        batch_labels_array = np.array(valid_labels)

        return (img1_batch, img2_batch), batch_labels_array

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
