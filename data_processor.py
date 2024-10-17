import os
import cv2
import numpy as np
import random
import shutil
from collections import Counter
from tensorflow.keras.utils import Sequence

class DataPreprocessor(Sequence):
    def __init__(self, images=None, labels=None, data_directory="/Users/sszyda/face_recognition/dataset", training_directory="/Users/sszyda/face_recognition/train",
                 val_directory="/Users/sszyda/face_recognition/val", image_size=(224, 224), batch_size=32, validation_split=0.3, shuffle=True):
        self.data_directory = data_directory
        self.training_directory = training_directory
        self.val_directory = val_directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.images = images
        self.labels = labels
        self.pairs = []
        self.labels_pair = []
        self.indexes = []

        if images is not None and labels is not None:
            self.on_epoch_end()  

    def load_images(self, directory):
        images = []
        labels = []

        for celebrity in os.listdir(directory):
            celebrity_directory = os.path.join(directory, celebrity)

            if os.path.isdir(celebrity_directory):
                for name in os.listdir(celebrity_directory):
                    image_path = os.path.join(celebrity_directory, name)
                    if cv2.imread(image_path) is not None:
                        images.append(image_path)
                        labels.append(celebrity)
                    else:
                        print(f"Warning: Could not read image {image_path}")

        return images, labels

    def divide_dataset(self):
        images, labels = self.load_images(self.data_directory)

        if os.path.exists(self.training_directory):
            shutil.rmtree(self.training_directory)
        if os.path.exists(self.val_directory):
            shutil.rmtree(self.val_directory)

        os.makedirs(self.training_directory, exist_ok=True)
        os.makedirs(self.val_directory, exist_ok=True)

        label_to_images = {}
        for img, label in zip(images, labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img)

        train_images = []
        train_labels = []
        val_images = []
        val_labels = []

        for label, img_list in label_to_images.items():
            if len(img_list) < 2:
                continue

            random.shuffle(img_list)  

            num_images = len(img_list)

            if num_images >= 4:
                num_val = max(2, int(num_images * self.validation_split))
                num_train = num_images - num_val

                if num_train < 2:
                    num_train = 2
                    num_val = num_images - num_train

                if num_val < 2:
                    num_val = 2
                    num_train = num_images - num_val

                train_imgs = img_list[:num_train]
                val_imgs = img_list[num_train:num_train + num_val]

                train_images.extend(train_imgs)
                train_labels.extend([label] * len(train_imgs))

                val_images.extend(val_imgs)
                val_labels.extend([label] * len(val_imgs))
            else:
                pass

        for img, label in zip(train_images, train_labels):
            label_dir = os.path.join(self.training_directory, label)
            os.makedirs(label_dir, exist_ok=True)
            dest_path = os.path.join(label_dir, os.path.basename(img))
            if not os.path.exists(dest_path):
                shutil.copy2(img, dest_path)

        for img, label in zip(val_images, val_labels):
            label_dir = os.path.join(self.val_directory, label)
            os.makedirs(label_dir, exist_ok=True)
            dest_path = os.path.join(label_dir, os.path.basename(img))
            if not os.path.exists(dest_path):
                shutil.copy2(img, dest_path)

        print(f"Training set: {len(train_images)} images")
        print(f"Validation set: {len(val_images)} images")

        train_distribution = Counter(train_labels)
        val_distribution = Counter(val_labels)
        print(f"Training distribution: {train_distribution}")
        print(f"Validation distribution: {val_distribution}")

    def load_training_data(self):
        images, labels = self.load_images(self.training_directory)
        return images, labels

    def load_validation_data(self):
        images, labels = self.load_images(self.val_directory)
        return images, labels

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        positive_pairs, negative_pairs = self.generate_pairs(self.images, self.labels)
        self.pairs = positive_pairs + negative_pairs
        self.labels_pair = [1]*len(positive_pairs) + [0]*len(negative_pairs)
        self.indexes = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.pairs))
        batch_indexes = self.indexes[start:end]
        batch_pairs = [self.pairs[i] for i in batch_indexes]
        batch_labels = [self.labels_pair[i] for i in batch_indexes]

        img1_batch = []
        img2_batch = []
        valid_labels = []
        for (img1_path, img2_path), label in zip(batch_pairs, batch_labels):
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
            valid_labels.append(label)

        if not img1_batch:
            return self.__getitem__((index + 1) % self.__len__())  

        img1_batch = np.array(img1_batch)
        img2_batch = np.array(img2_batch)
        batch_labels = np.array(valid_labels)

        return (img1_batch, img2_batch), batch_labels

    def generate_pairs(self, images, labels, num_pairs_per_class=100):
        positive_pairs = []
        negative_pairs = []

        label_to_images = {}
        for img, label in zip(images, labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img)

        # positive pairs
        total_positive_pairs = 0
        for label, img_list in label_to_images.items():
            if len(img_list) >= 2:
                pairs = [(img_list[i], img_list[j]) for i in range(len(img_list)) for j in range(i + 1, len(img_list))]
                random.shuffle(pairs)
                pairs = pairs[:num_pairs_per_class]
                positive_pairs.extend(pairs)
                num_pairs_generated = len(pairs)
                total_positive_pairs += num_pairs_generated
                print(f"Generated {num_pairs_generated} positive pairs for class '{label}'")

        print(f"Total positive pairs generated: {total_positive_pairs}")

        # negative pairs
        all_images = images.copy()
        img_to_label = {img: label for img, label in zip(images, labels)}
        negative_pairs_set = set()
        attempts = 0
        max_attempts = len(positive_pairs) * 10  p
        while len(negative_pairs) < len(positive_pairs) and attempts < max_attempts:
            img1, img2 = random.sample(all_images, 2)
            label1 = img_to_label[img1]
            label2 = img_to_label[img2]
            if label1 != label2 and (img1, img2) not in negative_pairs_set:
                negative_pairs.append((img1, img2))
                negative_pairs_set.add((img1, img2))
            attempts += 1

        if len(negative_pairs) < len(positive_pairs):
            print("Warning: Could not generate enough negative pairs to balance the dataset.")

        print(f"Total negative pairs generated: {len(negative_pairs)}")

        self.pairs = positive_pairs + negative_pairs
        self.labels_pair = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        combined = list(zip(self.pairs, self.labels_pair))
        random.shuffle(combined)
        self.pairs[:], self.labels_pair[:] = zip(*combined)

        return positive_pairs, negative_pairs

    def get_data_generator(self, images, labels):
        self.images = images
        self.labels = labels
        self.on_epoch_end()
        return self  
