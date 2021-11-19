import abc
import csv
import os

import numpy as np
from collections import defaultdict
from data_prep.config import *
from data_prep.data_preprocess_utils import save_json_file, sanitize_text, print_label_distribution, split_data
from data_prep.graph_io import GraphIO


class DataPreprocessor(GraphIO):

    def preprocess(self, min_len=6):
        """
        Applies some preprocessing to the data, e.g. replacing special characters, filters non-required articles out.
        :param min_len: Minimum required length for articles.
        :return: Numpy arrays for article tests (x_data), article labels (y_data), and article names (doc_names).
        """

        x_data, y_data, doc_names, x_lengths, invalid = [], [], [], [], []

        data_file = os.path.join(self.data_tsv_dir, self.dataset, CONTENT_INFO_FILE_NAME)

        with open(data_file, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                if isinstance(row['text'], str) and len(row['text']) >= min_len:
                    text = sanitize_text(row['text'])
                    x_data.append(text)
                    x_lengths.append(len(text))
                    y_data.append(int(row['label']))
                    doc_names.append(str(row['id']))
                else:
                    invalid.append(row['id'])

        print(f"Average length = {sum(x_lengths) / len(x_lengths)}")
        print(f"Maximum length = {max(x_lengths)}")
        print(f"Minimum Length = {min(x_lengths)}")
        print(f"Total data points invalid and therefore removed (length < {min_len}) = {len(invalid)}")

        return np.array(x_data), np.array(y_data), np.array(doc_names)

    def create_data_splits(self, test_size=0.2, val_size=0.1, splits=1, duplicate_stats=False):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution. Currently only supports splitting data in 1 split for each set.

        :param test_size: Size of the test split compared to the whole data.
        :param val_size: Size of the val split compared to the whole data.
        :param splits: Number of splits.
        :param duplicate_stats: If the statistics about duplicate texts should be collected or not.
        """

        self.print_step("Creating Data Splits")

        data = self.preprocess()

        if duplicate_stats:
            # counting duplicates in test set
            texts = []
            duplicates = defaultdict(lambda: {'counts': 1, 'd_names': {'real': [], 'fake': []}, 'classes': set()})

            for i in range(len(data[0])):
                d_text = data[0][i]
                if d_text in texts:
                    duplicates[d_text]['counts'] += 1
                    duplicates[d_text]['d_names'][self.labels()[data[1][i]]].append(data[2][i])
                else:
                    texts.append(d_text)

            duplicates_file = self.data_tsv_path('duplicates_info.json')
            save_json_file(duplicates, duplicates_file, converter=self.np_converter)

        # Creating train-val-test split with same/similar label distribution in each split

        # one tuple is one split and contains: (x, y, doc_names)
        rest_split, test_split = split_data(splits, test_size, data)

        assert len(set(test_split[2])) == len(test_split[2]), "Test split contains duplicate doc names!"

        # split rest data into validation and train splits
        train_split, val_split = split_data(splits, val_size, rest_split)

        assert len(set(val_split[2])) == len(val_split[2]), "Validation split contains duplicate doc names!"
        assert len(set(train_split[2])) == len(train_split[2]), "Train split contains duplicate doc names!"

        print_label_distribution(train_split[1])
        print_label_distribution(val_split[1])
        print_label_distribution(test_split[1])

        print("\nWriting train-val-test files..")

        splits = {'train': train_split, 'val': val_split, 'test': test_split}
        for split, data in splits.items():
            x, y, name_list = data

            split_path = self.data_tsv_path('splits')
            if not os.path.exists(split_path):
                os.makedirs(split_path)

            split_file_path = os.path.join(split_path, f'{split}.tsv')
            print(f"{split} file in : {split_file_path}")

            with open(split_file_path, 'w', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                csv_writer.writerow(['id', 'text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([name_list[i], x[i], y[i]])

        doc_splits_file = self.data_tsv_path(DOC_SPLITS_FILE_NAME)
        print("Writing doc_splits in : ", doc_splits_file)

        doc_names_train, doc_names_test, doc_names_val = train_split[2], test_split[2], val_split[2]
        print("\nTotal train = ", len(doc_names_train))
        print("Total test = ", len(doc_names_test))
        print("Total val = ", len(doc_names_val))

        split_dict = {'test_docs': doc_names_test, 'train_docs': doc_names_train, 'val_docs': doc_names_val}
        save_json_file(split_dict, doc_splits_file, converter=self.np_converter)

    def store_doc2labels(self, doc2labels):
        """
        Stores the doc2label dictionary as JSON file in the respective directory.

        :param doc2labels: Dictionary containing document names mapped to the label for that document.
        """

        print(f"Total docs : {len(doc2labels)}")
        doc2labels_file = self.data_complete_path(DOC_2_LABELS_FILE_NAME)

        print(f"Writing doc2labels JSON in :  {doc2labels_file}")
        save_json_file(doc2labels, doc2labels_file, converter=self.np_converter)

    def store_doc_contents(self, contents):
        """
        Stores document contents (doc name/id, title, text and label) as a TSV file.

        :param contents: List of entries (doc name/id, title, text and label) for every article.
        """
        print("\nCreating the data corpus file for: ", self.dataset)

        content_dest_file = self.data_tsv_path(CONTENT_INFO_FILE_NAME)
        with open(content_dest_file, 'w', encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['id', 'title', 'text', 'label'])
            for file_content in contents:
                csv_writer.writerow(file_content)

        print("Final file written in :  ", content_dest_file)

    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError