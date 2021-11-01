import torch
from transformers import AutoTokenizer
from transformers.data.processors.squad import *
from tqdm import tqdm
import json, os
import pandas as pd

import pickle5 as pickle

from torch.utils.data import Dataset


class CorpusQA(Dataset):
    def __init__(
        self, path, evaluate, model_name="xlm-roberta-base", local_files_only=False
    ):
        self.doc_stride = 128
        self.max_query_len = 64
        self.max_seq_len = 384

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            use_fast=False,
            local_files_only=local_files_only,
        )

        self.data, self.examples, self.features = self.preprocess(path, evaluate)

    def preprocess(self, file, evaluate=False):
        file = file.split("/")
        filename = file[-1]
        data_dir = "/".join(file[:-1])

        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}".format(type(self.tokenizer).__name__, filename)
        )

        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            features_and_dataset = torch.load(cached_features_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            processor = SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(data_dir, filename)
            else:
                examples = processor.get_train_examples(data_dir, filename)

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_len,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_len,
                is_training=not evaluate,
                return_dataset="pt",
                threads=1,
            )

            dataset[:][1] = [item.bool() for item in dataset[:][1]]
            dataset[:][2] = [item.bool() for item in dataset[:][2]]

            torch.save(
                {"features": features, "dataset": dataset, "examples": examples},
                cached_features_file,
            )

        return dataset, examples, features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return {
            "input_ids": self.data[id][0],
            "attention_mask": self.data[id][1],
            "token_type_ids": self.data[id][2],
            "answer_start": self.data[id][3],
            "answer_end": self.data[id][4],
        }


class CorpusSC(Dataset):
    def __init__(
        self, path, file, model_name="xlm-roberta-base", local_files_only=False
    ):
        self.max_sequence_length = 128
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = 0
        self.sequence_a_segment_id = 0
        self.sequence_b_segment_id = 1
        self.pad_token_segment_id = 0
        self.cls_token_segment_id = 0
        self.mask_padding_with_zero = True
        self.doc_stride = 128
        self.max_query_length = 64

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=False, local_files_only=local_files_only
        )

        self.label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

        cached_data_file = path + f"_{type(self.tokenizer).__name__}.pickle"

        if os.path.exists(cached_data_file):
            self.data = pickle.load(open(cached_data_file, "rb"))
        else:
            self.data = self.preprocess(path, file)
            with open(cached_data_file, "wb") as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def preprocess(self, path, file):

        labels = []
        input_ids = []
        token_type_ids = []
        attention_mask = []

        if file == "csv":
            header = ["premise", "hypothesis", "label"]
            df = pd.read_csv(path, sep="\t", header=None, names=header)

            premise_list = df["premise"].to_list()
            hypothesis_list = df["hypothesis"].to_list()
            label_list = df["label"].to_list()

            # Tokenize input pair sentences
            ids = self.tokenizer(
                premise_list,
                hypothesis_list,
                add_special_tokens=True,
                max_length=self.max_sequence_length,
                truncation=True,
                padding=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )
            input_ids = ids["input_ids"]
            attention_mask = ids["attention_mask"].bool()
            token_type_ids = ids["token_type_ids"].bool()

            labels = torch.tensor(
                [self.label_dict[label] for label in label_list], dtype=torch.uint8
            )
        else:
            with open(path, encoding="utf-8") as f:
                data = [json.loads(jline) for jline in f.readlines()]
                for row in tqdm(data):
                    label = row["gold_label"]

                    if label not in ["neutral", "contradiction", "entailment"]:
                        continue

                    sentence1_tokenized = self.tokenizer.tokenize(row["sentence1"])
                    sentence2_tokenized = self.tokenizer.tokenize(row["sentence2"])
                    if (
                        len(sentence1_tokenized) + len(sentence2_tokenized) + 3
                        > self.max_sequence_length
                    ):
                        continue

                    input_ids, token_type_ids, attention_mask = self.encode(
                        sentence1_tokenized, sentence2_tokenized
                    )
                    labels.append(self.label_dict[label])
                    input_ids.append(torch.unsqueeze(input_ids, dim=0))
                    token_type_ids.append(torch.unsqueeze(token_type_ids, dim=0))
                    attention_mask.append(torch.unsqueeze(attention_mask, dim=0))

            labels = torch.tensor(labels)
            input_ids = torch.cat(input_ids, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

        dataset = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

        return dataset

    def encode(self, sentence1, sentence2):

        tokens = []
        segment_mask = []
        input_mask = []

        tokens.append(self.cls_token)
        segment_mask.append(self.cls_token_segment_id)
        input_mask.append(1 if self.mask_padding_with_zero else 0)

        for tok in sentence1:
            tokens.append(tok)
            segment_mask.append(self.sequence_a_segment_id)
            input_mask.append(1 if self.mask_padding_with_zero else 0)

        tokens.append(self.sep_token)
        segment_mask.append(self.sequence_a_segment_id)
        input_mask.append(1 if self.mask_padding_with_zero else 0)

        for tok in sentence2:
            tokens.append(tok)
            segment_mask.append(self.sequence_b_segment_id)
            input_mask.append(1 if self.mask_padding_with_zero else 0)

        tokens.append(self.sep_token)
        segment_mask.append(self.sequence_b_segment_id)
        input_mask.append(1 if self.mask_padding_with_zero else 0)

        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        while len(tokens) < self.max_sequence_length:
            tokens.append(self.pad_token)
            segment_mask.append(self.pad_token_segment_id)
            input_mask.append(0 if self.mask_padding_with_zero else 1)

        tokens = torch.tensor(tokens)
        segment_mask = torch.tensor(segment_mask)
        input_mask = torch.tensor(input_mask)

        return tokens, segment_mask, input_mask

    def __len__(self):
        return self.data["input_ids"].shape[0]

    def __getitem__(self, id):
        return {
            "input_ids": self.data["input_ids"][id],
            "token_type_ids": self.data["token_type_ids"][id],
            "attention_mask": self.data["attention_mask"][id],
            "label": self.data["label"][id],
        }
