
import os
import json
import operator
import numpy as np
import time
import random
import pickle

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from transformers import RobertaModel, BertModel, RobertaForMaskedLM


def data_file_gen(data_path):
    for label in os.listdir(f'{data_path}/'):
        for file_name in os.listdir(f'{data_path}/{label}'):
            data_file = f'{data_path}/{label}/{file_name}'
            yield data_file


class FakeNewsNetWithEntity(Dataset):
    def __init__(self, data_path, input_embed_path) -> None:
        super().__init__()
        self.texts, self.texts_embed, self.entities, self.labels, self.seq_lens = self.read_data(data_path, input_embed_path)

    def read_data(self, data_path, input_embed_path):
        # get input text embedding dictionary
        with open(input_embed_path) as f:
            input_emb = pickle.load(f)

        label_id = {"real": 0, "fake": 1}
        max_len = 512
        texts, texts_emb, entities, labels, seq_lens = [], [], [], [], []
        for data_file in data_file_gen(data_path):
            label = label_id[data_file.split('/')[-2]]
            with open(data_file, 'r') as f:
                text = f.readline().strip()
                if text is None or len(text.split()) == 0:
                    continue
                text_emb = input_emb[text]
            entity_seq = []
            for line in f.readlines():
                try:
                    item = eval(line.strip())
                except SyntaxError:
                    continue
                entity_seq.append(item['entity_name'])
            
            text_split = text.split()
            if len(text_split) > max_len:
                text = " ".join(text_split[:max_len])

            labels.append(label)
            texts.append(text)
            seq_lens.append(len(text))
            entities.append(entity_seq)
            texts_emb.append(text_emb)

        return texts, texts_emb, entities, labels, seq_lens
    
    def __getitem__(self, idx):
        return self.texts[idx], self.entities[idx], self.labels[idx], self.seq_lens[idx], self.texts_embed[idx]

    def __len__(self):
        return len(self.labels)


class PromptTokenzierWithEntityCollator():
    def __init__(self, tokenizer, token_idx, entity_idx, label_idx, texts_emb_idx, sort_key):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 
        self.entity_idx = entity_idx
        self.texts_emb_idx = texts_emb_idx

        self.sort_key = sort_key

        self.tokenizer = tokenizer
        self.mask_ids = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # pre-trained language model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        for param in self.roberta.base_model.parameters():  # type: ignore
            param.requires_grad = False
        self.roberta.embeddings.word_embeddings.requires_grad = False

        self.unused_ids = torch.tensor([-1])
        self.prefix_prompt = "The veracity of the following news is <mask> . "
        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids'][0][:-1]
        self.prefix_ids = torch.cat([self.cls_id, self.unused_ids, self.unused_ids, self.prefix_ids[1:]], dim=0)
        self.postfix_ids = self.eos_id

        self.cls_id = torch.tensor([self.prefix_ids[0]])
        self.eos_id = torch.tensor([self.postfix_ids[-1]])

        self.add_len = int(len(self.prefix_ids) + len(self.postfix_ids))
        self.add_attention_mask = torch.ones(self.add_len)

        self.max_len = 512 - self.add_len
        self.max_en_num = 200
        self.max_neigh_num = 10

        self.pad_emd = self.roberta.embeddings.word_embeddings(torch.tensor([self.pad_token_id]))

        desc_path_politifact = "desc_emb/politifact_desc_emb.pkl"
        desc_path_snope = "desc_emb/snope_desc_emb.pkl"
        desc_path_coaid = "desc_emb/coaid_desc_emb.pkl"
        desc_path_coaid_refine = "desc_emb/coaid_refine_desc_emb.pkl"
        self.desc_to_emb_politifact = self.read_desc_emb(desc_path_politifact)
        self.desc_to_emb_snope = self.read_desc_emb(desc_path_snope)
        self.desc_to_emb_coaid = self.read_desc_emb(desc_path_coaid)
        self.desc_to_emb_coaid_refine = self.read_desc_emb(desc_path_coaid_refine)
        self.desc_to_emb = self.desc_to_emb_politifact | self.desc_to_emb_snope | self.desc_to_emb_coaid | self.desc_to_emb_coaid_refine

        self.ent_to_embed = self.read_desc_emb("ent_to_neighs/ent_to_embed.pkl")
        self.ent_to_neighs_embed = self.read_desc_emb("ent_to_neighs/ent_to_neighs_embed.pkl")

    def read_desc_emb(self, path):
        with open(path, 'rb') as f:
            desc_emb = pickle.load(f)
        return desc_emb

    def _collate_fn(self, batch):  # modify for constructing input_ids
        ret = []
        # batch: texts, entities, labels, seq_lens, texts_emb
        batch.sort(key=self.sort_key, reverse=True)
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                input_ids_lst, attention_mask_lst = [], []
                for sample in samples:
                    inputs = self.tokenizer(sample,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    if len(inputs) == 2:  # roberta
                        input_ids, attention_mask = inputs
                    elif len(inputs) == 3:  # bert
                        input_ids, _, attention_mask = inputs
                    else:
                        raise RuntimeError
                    input_ids = input_ids[0][1:-1]
                    attention_mask = attention_mask[0][1:-1]
                    if len(input_ids) > self.max_len:
                        input_ids = input_ids[:self.max_len]
                        attention_mask = attention_mask[:self.max_len]
                    input_ids = torch.cat([self.prefix_ids, input_ids, self.postfix_ids], dim=0)
                    attention_mask = torch.cat([attention_mask, self.add_attention_mask], dim=0)

                    input_ids_lst.append(input_ids)
                    attention_mask_lst.append(attention_mask)

                input_ids = rnn_utils.pad_sequence(input_ids_lst, batch_first=True)
                attention_mask = rnn_utils.pad_sequence(attention_mask_lst, batch_first=True)
                ret.append(input_ids)
                ret.append(attention_mask)
            elif i == self.entity_idx:
                ############ entity embedding
                # entities : [batch_size, entity_num, entity_seq]
                entity_embs = []
                entity_lens = []
                max_entities = max(len(entities) for entities in samples)
                for entity in samples:
                    entity_emb_lst = []
                    # print("entity: ", entity)
                    if entity == []:
                        # entity_emb_lst.append(torch.zeros(768))
                        entity_emb_lst.append(self.pad_emd)
                    else:

                        for en in entity:
                            if en in self.ent_to_embed:
                                entity_emb_lst.append(self.ent_to_embed[en])
                            else:
                                inputs = self.tokenizer(en,
                                                        padding=False,
                                                        truncation=False,
                                                        return_tensors="pt").values()
                                if len(inputs) == 2:  # roberta
                                    en_ids, _ = inputs
                                elif len(inputs) == 3:  # bert
                                    en_ids, _, _ = inputs
                                else:
                                    raise RuntimeError
                                # print(en_ids)
                                en_ids = en_ids[0][1:-1]
                                en_emb = torch.mean(self.roberta.embeddings.word_embeddings(en_ids), dim=0)
                                entity_emb_lst.append(en_emb)

                    padded_entity_emb = torch.stack([self.pad_emd] * max_entities).squeeze()
                    num_entities = len(entity_emb_lst)

                    entity_lens.append(min(max_entities, num_entities))
                    for j in range(min(max_entities, num_entities)):
                        padded_entity_emb[j, :] = entity_emb_lst[j]

                    entity_embs.append(padded_entity_emb)

                entity_embs = torch.stack(entity_embs)  # [batch_size, entity_num, emb_len] e.g. [16, 8, 768]
                entity_lens = torch.tensor(entity_lens)
                ret.append(entity_embs)
                ret.append(entity_lens)

                ########## Description embedding
                # entities : [batch_size, entity_num, entity_seq]
                entity_embs = []
                max_entities = max(len(entities) for entities in samples)
                for entity in samples:
                    # eg: ['Chris Christie', 'Medicaid', 'Patient Protection and Affordable Care Act']
                    entity_emb_lst = []
                    if entity == []:
                        # entity_emb_lst.append(torch.zeros(768))
                        entity_emb_lst.append(self.pad_emd)
                    else:
                        for ent in entity:
                            try:
                                entity_emb_lst.append(self.desc_to_emb[ent])
                            except KeyError:
                                inputs = self.tokenizer(ent,
                                                        padding=False,
                                                        truncation=False,
                                                        return_tensors="pt").values()
                                if len(inputs) == 2:  # roberta
                                    en_ids, _ = inputs
                                elif len(inputs) == 3:  # bert
                                    en_ids, _, _ = inputs
                                else:
                                    raise RuntimeError
                                # print(en_ids)
                                en_ids = en_ids[0][1:-1]
                                en_emb = torch.mean(self.roberta.embeddings.word_embeddings(en_ids), dim=0).reshape(1, -1)
                                entity_emb_lst.append(en_emb)

                    padded_entity_emb = torch.stack([self.pad_emd] * max_entities).squeeze()
                    num_entities = len(entity_emb_lst)

                    for j in range(min(max_entities, num_entities)):
                        padded_entity_emb[j, :] = entity_emb_lst[j]

                    entity_embs.append(padded_entity_emb)

                entity_embs = torch.stack(entity_embs)  # [batch_size, entity_num, emb_len] e.g. [16, 8, 768]
                ret.append(entity_embs)

                ########### Neigbors embedding
                entity_neigh_embs = []
                # samples: [batch_size, entity_num, neighbor_num, neighbor_seq]
                max_entities = max(len(entities) for entities in samples)
                for en_sample in samples:
                    en_neigh_embs = []
                    for neighs in en_sample:
                        if neighs in self.ent_to_neighs_embed:
                            neighs_embs = self.ent_to_neighs_embed[neighs]
                        else:
                            neighs_embs = []
                            inputs = self.tokenizer(neighs,
                                                    padding=False,
                                                    truncation=False,
                                                    return_tensors="pt").values()
                            if len(inputs) == 2:  # roberta
                                en_ids, _ = inputs
                            elif len(inputs) == 3:  # bert
                                en_ids, _, _ = inputs
                            else:
                                raise RuntimeError
                            # print(en_ids)
                            en_ids = en_ids[0][1:-1]
                            en_emb = torch.mean(self.roberta.embeddings.word_embeddings(en_ids), dim=0).reshape(1, -1)
                            neighs_embs.append(en_emb)

                        padded_neighs_embs = torch.stack([self.pad_emd] * self.max_neigh_num).squeeze()
                        # padded_neighs_embs = torch.zeros(self.max_neigh_num, neighs_embs[0].size(0))
                        num_neigh = len(neighs_embs)
                        for j in range(min(self.max_neigh_num, num_neigh)):
                            padded_neighs_embs[j, :] = neighs_embs[j]

                        en_neigh_embs.append(padded_neighs_embs)

                    padded_en_neigh_embs = [torch.stack([self.pad_emd] * self.max_neigh_num).squeeze()] * max_entities
                    padded_en_neigh_embs = torch.stack(padded_en_neigh_embs)
                    entity_num = len(en_neigh_embs)
                    for j in range(min(self.max_en_num, entity_num)):
                        padded_en_neigh_embs[j, :, :] = en_neigh_embs[j]

                    entity_neigh_embs.append(padded_en_neigh_embs)

                entity_neigh_embs = torch.stack(entity_neigh_embs)
                if torch.isnan(entity_neigh_embs).any().item():
                    print(entity_neigh_embs)
                # print(entity_neigh_embs.size())
                ret.append(entity_neigh_embs)
            elif i == self.texts_emb_idx:
                texts_emb = []
                for sample in samples:
                    texts_emb.append([item.numpy() for item in sample])
                ret.append(torch.tensor(np.array(texts_emb)))
            else:
                ret.append(torch.tensor(samples))
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)
