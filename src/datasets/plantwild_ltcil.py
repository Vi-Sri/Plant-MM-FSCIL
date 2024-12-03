import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

random.seed(1)
template = ['a photo of a {}, a type of plant disease.']


class PlantWild_LTCIL(DatasetBase):
    """
    PlantWild dataset class modified for Long-Tailed Class Incremental Learning (LT-CIL).
    This class splits the dataset into incremental tasks with long-tailed distributions.
    """

    dataset_dir = 'plantwild'

    def __init__(self, root, num_shots, num_tasks, imbalance_type='exp', imbalance_factor=0.01, shuffle_classes=True):
        """
        Initializes the PlantWild_LTCIL dataset.

        Parameters:
        - root (str): Root directory of the dataset.
        - num_shots (int): Number of few-shot samples per class.
        - num_tasks (int): Number of incremental tasks.
        - imbalance_type (str): Type of imbalance ('exp', 'step', 'fewshot', or 'none').
        - imbalance_factor (float): Factor to control the degree of imbalance.
        - shuffle_classes (bool): Whether to shuffle class order.
        """
        root = os.path.abspath(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, 'split_ltc.json')

        self.template = template
        self.num_tasks = num_tasks
        self.imbalance_type = imbalance_type
        self.imbalance_factor = imbalance_factor
        self.shuffle_classes = shuffle_classes

        self.image_names_list = []
        self.image_labels_list = []
        self.classes = sorted(os.listdir(self.image_dir))

        # Get all the image names and corresponding labels
        for i, cls in enumerate(self.classes):
            c_root = os.path.join(self.image_dir, cls)
            image_names = os.listdir(c_root)
            for j, img in enumerate(image_names):
                self.image_names_list.append(os.path.join(cls, img))
                self.image_labels_list.append(i)

        # If split file does not exist, create it
        if not os.path.exists(self.split_path):
            self.create_ltc_split(num_tasks, imbalance_type, imbalance_factor)
        else:
            self.load_split()

        # Generate few-shot dataset for each task
        self.generate_fewshot_datasets(num_shots)

        super().__init__(train_x=self.train_tasks, val=self.val_tasks, test=self.test_tasks)

    def create_ltc_split(self, num_tasks, imbalance_type, imbalance_factor):
        """
        Creates and saves the LT-CIL split.

        Parameters:
        - num_tasks (int): Number of incremental tasks.
        - imbalance_type (str): Type of imbalance.
        - imbalance_factor (float): Factor to control imbalance.
        """
        print("Creating LT-CIL split...")
        # Organize data per class
        class_data = defaultdict(list)
        for img, label in zip(self.image_names_list, self.image_labels_list):
            class_data[label].append(img)

        # Sort classes based on the number of samples (descending)
        sorted_classes = sorted(class_data.keys(), key=lambda x: len(class_data[x]), reverse=True)

        # Apply imbalance
        if imbalance_type == 'exp':
            # Exponential decay for class frequencies
            max_samples = len(class_data[sorted_classes[0]])
            for idx, cls in enumerate(sorted_classes):
                factor = imbalance_factor ** (idx / (len(sorted_classes) - 1))
                desired_num = int(max_samples * factor)
                class_data[cls] = class_data[cls][:desired_num]
        elif imbalance_type == 'step':
            # Step imbalance
            step = len(sorted_classes) // 2
            for idx, cls in enumerate(sorted_classes):
                if idx < step:
                    continue  # Head classes remain unchanged
                else:
                    class_data[cls] = class_data[cls][:int(len(class_data[cls]) * imbalance_factor)]
        elif imbalance_type == 'fewshot':
            # Few-shot imbalance: first 50 classes have many samples, the rest have few
            for idx, cls in enumerate(sorted_classes):
                if idx < 50:
                    continue
                else:
                    class_data[cls] = class_data[cls][:max(1, int(len(class_data[cls]) * imbalance_factor))]
        elif imbalance_type == 'none':
            # No imbalance
            pass
        else:
            raise ValueError("Unsupported imbalance type. Choose from 'exp', 'step', 'fewshot', 'none'.")

        # Shuffle classes if required
        if self.shuffle_classes:
            random.shuffle(sorted_classes)

        # Split classes into tasks
        classes_per_task = len(sorted_classes) // self.num_tasks
        remaining = len(sorted_classes) % self.num_tasks
        train_tasks = []
        val_tasks = []
        test_tasks = []

        start = 0
        for t in range(self.num_tasks):
            end = start + classes_per_task + (1 if t < remaining else 0)
            task_classes = sorted_classes[start:end]
            start = end

            # Collect training data for the task
            task_train = []
            for cls in task_classes:
                imgs = class_data[cls]
                task_train.extend(imgs)

            # Split into train and validation
            split = int(len(task_train) * 0.8)
            random.shuffle(task_train)
            train = task_train[:split]
            val = task_train[split:]

            train_tasks.append({'x': train, 'y': [self.image_labels_list[self.image_names_list.index(img)] for img in train]})
            val_tasks.append({'x': val, 'y': [self.image_labels_list[self.image_names_list.index(img)] for img in val]})
            # For simplicity, use the same test set as validation
            test_tasks.append({'x': val, 'y': [self.image_labels_list[self.image_names_list.index(img)] for img in val]})

        # Save the split
        split = {
            'train': train_tasks,
            'val': val_tasks,
            'test': test_tasks,
            'class_order': sorted_classes
        }
        write_json(split, self.split_path)
        print(f"Saved LT-CIL split to {self.split_path}")

    def load_split(self):
        """
        Loads the LT-CIL split from the split file.
        """
        print("Loading LT-CIL split...")
        split = read_json(self.split_path)
        self.train_tasks = split['train']
        self.val_tasks = split['val']
        self.test_tasks = split['test']
        self.class_order = split['class_order']
        print(f"Loaded LT-CIL split from {self.split_path}")

    def generate_fewshot_datasets(self, num_shots):
        """
        Generates few-shot datasets for each task.

        Parameters:
        - num_shots (int): Number of few-shot samples per class.
        """
        print(f"Generating few-shot datasets with {num_shots} shots per class...")
        fewshot_train = []
        for task in self.train_tasks:
            selected = {}
            for img, label in zip(task['x'], task['y']):
                if label not in selected and len(selected) < num_shots:
                    selected[label] = img
            fewshot_train.extend(selected.values())

        # Update train_tasks with few-shot samples
        self.train_tasks = []
        for task in self.train_tasks:
            task_fewshot = []
            for img, label in zip(task['x'], task['y']):
                if img in fewshot_train:
                    task_fewshot.append(img)
            self.train_tasks.append({'x': task_fewshot, 'y': [self.image_labels_list[self.image_names_list.index(img)] for img in task_fewshot]})

        print("Few-shot datasets generated.")

    def read_data(self, split_file):
        """
        Override read_data to match LT-CIL split format.
        """
        # Not used in LT-CIL as we handle splits differently
        pass

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        """
        Override split_trainval to match LT-CIL split format.
        """
        # Not used in LT-CIL as we handle splits differently
        pass

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        """
        Override save_split to match LT-CIL split format.
        """
        # Not used in LT-CIL as we handle splits differently
        pass

    @staticmethod
    def read_split(filepath, path_prefix):
        """
        Override read_split to match LT-CIL split format.
        """
        # Not used in LT-CIL as we handle splits differently
        pass

    @staticmethod
    def read_split_base(split_base, path_prefix, name_list, label_list, classes):
        """
        Override read_split_base to match LT-CIL split format.
        """
        # Not used in LT-CIL as we handle splits differently
        pass

    @staticmethod
    def read_split_txt(path, classes, prefix):
        """
        Override read_split_txt to match LT-CIL split format.
        """
        # Not used in LT-CIL as we handle splits differently
        pass
