import os
import cv2
import numpy as np
import random

class DataProcessor:
    def __init__(self, data_directory="dataset", image_size=(224, 224), batch_size=32, validation_split=0.2):
        self.data_directory = data_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.load_and_split_data()
        self.train_pairs = []
        self.train_labels_pair = []
        self.val_pairs = []
        self.val_labels_pair = []

        self.generate_pairs()

    def load_data(self):
        images = []
        labels = []
        label_to_images = {}

        for identity in os.listdir(self.data_directory):
            identity_dir = os.path.join(self.data_directory, identity)
            if os.path.isdir(identity_dir):
                img_files = [os.path.join(identity_dir, f) for f in os.listdir(identity_dir) if f.endswith('.png')]
                labels.extend([identity] * len(img_files))
                images.extend(img_files)
                label_to_images[identity] = img_files

        return images, labels, label_to_images

    def load_and_split_data(self):
        images, labels, label_to_images = self.load_data()

        # split into train and validation
        for label, img_list in label_to_images.items():
            num_images = len(img_list)
            random.shuffle(img_list)
            num_val = int(num_images * self.validation_split)
            num_train = num_images - num_val

            self.train_images.extend(img_list[:num_train])
            self.train_labels.extend([label] * num_train)
            self.val_images.extend(img_list[num_train:])
            self.val_labels.extend([label] * num_val)

        print(f"Training set: {len(self.train_images)} images")
        print(f"Validation set: {len(self.val_images)} images")

    def build_label_to_images(self, images, labels):
        label_to_images = {}
        for img, label in zip(images, labels):
            label_to_images.setdefault(label, []).append(img)
        return label_to_images

    def generate_pairs(self, num_pairs_per_identity=20):
        label_to_images_train = self.build_label_to_images(self.train_images, self.train_labels)
        self.train_pairs, self.train_labels_pair = self.create_pairs(label_to_images_train, num_pairs_per_identity)

        label_to_images_val = self.build_label_to_images(self.val_images, self.val_labels)
        self.val_pairs, self.val_labels_pair = self.create_pairs(label_to_images_val, num_pairs_per_identity)

    def create_pairs(self, label_to_images, num_pairs_per_identity):
        positive_pairs = []
        negative_pairs = []

        # positive pairs
        for label, img_list in label_to_images.items():
            if len(img_list) >= 2:
                pairs = list(zip(img_list[:-1], img_list[1:]))
                random.shuffle(pairs)
                positive_pairs.extend(pairs[:num_pairs_per_identity])

        # negative pairs
        labels_list = list(label_to_images.keys())
        num_negative_pairs = len(positive_pairs)
        while len(negative_pairs) < num_negative_pairs:
            label1, label2 = random.sample(labels_list, 2)
            img1 = random.choice(label_to_images[label1])
            img2 = random.choice(label_to_images[label2])
            negative_pairs.append((img1, img2))

        pairs = positive_pairs + negative_pairs
        labels_pair = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        combined = list(zip(pairs, labels_pair))
        random.shuffle(combined)
        pairs[:], labels_pair[:] = zip(*combined)

        print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")
        return pairs, labels_pair

    def data_generator(self, pairs, labels_pair):
        num_samples = len(pairs)
        while True:
            for offset in range(0, num_samples, self.batch_size):
                batch_pairs = pairs[offset:offset+self.batch_size]
                batch_labels = labels_pair[offset:offset+self.batch_size]

                img1_batch = []
                img2_batch = []
                for img1_path, img2_path in batch_pairs:
                    img1 = cv2.imread(img1_path)
                    img2 = cv2.imread(img2_path)
                    if img1 is None or img2 is None:
                        continue
                    img1 = cv2.resize(img1, self.image_size)
                    img2 = cv2.resize(img2, self.image_size)
                    img1 = img1 / 255.0
                    img2 = img2 / 255.0
                    img1_batch.append(img1)
                    img2_batch.append(img2)

                img1_batch = np.array(img1_batch)
                img2_batch = np.array(img2_batch)
                batch_labels_array = np.array(batch_labels[:len(img1_batch)])

                yield (img1_batch, img2_batch), batch_labels_array
