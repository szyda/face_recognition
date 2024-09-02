import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

class DataProcessor:
    def __init__(self, lfw_dataset, image_size=(224, 224), batch_size=32):
        self.lfw_dataset = lfw_dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.data = self.load_all_data()

    def preprocess_image(self, example):
        x, y = example['image'], example['label']
        x = tf.image.resize(x, self.image_size)
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    def augment_image(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label

    # on m2 32gb macbook i can cache it (i think)
    def load_all_data(self):
        lfw_data = self.lfw_dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        lfw_data = lfw_data.cache()
        return lfw_data.batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    def split_data(self, data):
        images = []
        labels = []

        for batch in data:
            if 'image' in batch and 'label' in batch:
                images.append(batch['image'])
                labels.append(batch['label'])
            else:
                print("Warning: Batch missing 'image' or 'label' key. Skipping this batch.")

        if images and labels:
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
        else:
            raise ValueError("No valid images or labels to concatenate.")

        return images, labels

    def get_train_data(self):
        train_data = self.lfw_dataset.map(self.preprocess).cache().shuffle(buffer_size=10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return train_data

    def get_validation_data(self):
        val_data = self.lfw_dataset.map(self.preprocess).cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return val_data

    def preprocess(self, sample):
        image = sample['image']
        label = sample['label']

        image = tf.image.resize(image, self.image_size)
        image = image / 255.0

        return image, label

    def generate_image_pairs(self, data, max_pairs_per_class=50):
        pairs = []
        labels = []

        class_images = {}
        for images, labels_batch in data:
            for img, lbl in zip(images, labels_batch):
                lbl_str = lbl.numpy().decode('utf-8') if isinstance(lbl.numpy(), bytes) else str(lbl.numpy())
                if lbl_str not in class_images:
                    class_images[lbl_str] = []
                class_images[lbl_str].append(img)

        unique_labels = list(class_images.keys())

        # for testing, will delete later
        for class_name, images in class_images.items():
            print(f"Class {class_name}: {len(images)} images")

        for class_name, images in class_images.items():
            if len(images) < 2:
                continue

            random.shuffle(images)
            num_images = len(images)
            num_positive_pairs = min(max_pairs_per_class, num_images * (num_images - 1) // 2)

            pairs += [[images[i], images[j]] for i in range(num_images) for j in range(i + 1, num_images)]
            labels += [1] * len(pairs[-num_positive_pairs:])

            pairs = pairs[:num_positive_pairs]
            labels = labels[:num_positive_pairs]

            while len(pairs) < 2 * max_pairs_per_class:
                different_class = random.choice([c for c in unique_labels if c != class_name])
                different_image = random.choice(class_images[different_class])
                pairs.append([random.choice(images), different_image])
                labels.append(0)

        print(f"Generated {len(pairs)} pairs.")

        return np.array(pairs), np.array(labels)

