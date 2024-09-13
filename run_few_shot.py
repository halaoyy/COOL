import os
import random
import itertools

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.dataset import T
from transformers import RobertaTokenizer, RobertaConfig, BertTokenizer, BertConfig

from data import FakeNewsNet, TokenizedCollator, TokenizedWithPromptCollator, FakeNewsNetWithEntity, PromptTokenzierWithEntityCollator, PromptTokenzierWithEntityCollator_KPL
from trainer import Trainer
from utils import load_config, set_seed, train_val_split, print_measures, get_label_blance
from model import COOL
# from transformers import RobertaConfig

if __name__ == "__main__":
    fine_tune_all = True
    use_learnable_token = True
    with_answer_weights = True
    only_mask = False
    using_prefix = True
    using_postfix = False
    use_knowledgeable_token = True
    is_claim_mask = True  # if claim is fore-positioned
    num_prefix_soft = 2
    shot = 8
    
    config = load_config("config/few_shot.ini")

    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    bert_config = RobertaConfig()

    s_dataset = "snope_complete"
    t_dataset = "politifact_complete"

    seeds = trainer_config["seed"]

    positive_words = ['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'factual', 'correct', 'fact', 'truth']
    negative_words = ['false', 'fake', 'unreal', 'misleading', 'artificial', 'bogus', 'virtual', 'incorrect', 'wrong', 'fault']
    prompt_words = positive_words + negative_words

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    mask_token = tokenizer(tokenizer.mask_token)['input_ids'][1]  # type: ignore

    tokenized_collator = PromptTokenzierWithEntityCollator(tokenizer,
                                                                token_idx=0,
                                                                entity_idx=1,
                                                                label_idx=2,
                                                                texts_emb_idx=4,
                                                                sort_key=lambda x: x[3])  # type: ignore

    pos_tokens = tokenizer(" ".join(positive_words))['input_ids'][1:-1]  # type: ignore
    neg_tokens = tokenizer(" ".join(negative_words))['input_ids'][1:-1]  # type: ignore

    res = []
    for seed in seeds:
        set_seed(seed)
        s_data_path = data_config['data_dir'] + "/" + s_dataset
        t_data_path = data_config['data_dir'] + "/" + t_dataset
        s_input_embed_path = ""
        t_input_embed_path = ""
        s_data = FakeNewsNetWithEntity(s_data_path, s_input_embed_path)
        t_data = FakeNewsNetWithEntity(t_data_path, t_input_embed_path)

        if trainer_config['cuda'] and torch.cuda.is_available:
            torch.cuda.empty_cache()

        t_ids = [i for i in range(len(t_data))]  # type: ignore
        s_ids = [i for i in range(len(s_data))]
        random.shuffle(t_ids)
        random.shuffle(s_ids)

        t_train_ids_pool, t_val_ids_pool = get_label_blance(t_data, t_ids, shot)
        t_train_ids = t_train_ids_pool
        t_val_ids = t_val_ids_pool
        t_test_ids = t_ids.copy()
        for i in itertools.chain(t_train_ids_pool, t_val_ids_pool):
            t_test_ids.remove(i)

        t_train_data = Subset(t_data, t_train_ids)
        val_data = Subset(t_data, t_val_ids)
        test_data = Subset(t_data, t_test_ids)

        t_train_iter = DataLoader(dataset=t_train_data,
                                    batch_size=data_config["batch_size"],
                                    collate_fn=tokenized_collator)
        s_train_iter = DataLoader(dataset=s_data,
                                    batch_size=data_config["batch_size"],
                                    collate_fn=tokenized_collator)
        val_iter = DataLoader(dataset=val_data,
                                batch_size=data_config["batch_size"],
                                collate_fn=tokenized_collator)
        test_iter = DataLoader(dataset=test_data,
                                batch_size=data_config["batch_size"],
                                collate_fn=tokenized_collator)

        model = COOL(model_config,   # type: ignore
                        bert_config=bert_config,
                        trainer_config=trainer_config,
                        mask_token_id=mask_token,
                        positive_token_ids = pos_tokens,
                        negative_token_ids = neg_tokens,
                        with_learnable_emb = use_learnable_token,
                        with_answer_weights = with_answer_weights,
                        with_position_weights = False,
                        num_learnable_token = num_learnable_token,
                        zero_shot=(shot == 0),
                        fine_tune_all=fine_tune_all)


        trainer = Trainer(trainer_config)

        best_res, best_model = trainer.train(s_train_iter=s_train_iter,
                                                t_train_iter=t_train_iter,
                                                val_iter=val_iter,
                                                model=model,
                                                trainset_size=len(t_train_data),
                                                batch_size=data_config['batch_size'],
                                                class_balance=(s_dataset))
        model.load_state_dict(best_model)  # type: ignore
        test_loss, test_metrics = trainer.evaluate(model, test_iter)

        print("------------------------------------------")
        print("-Test: ")
        print_measures(test_loss, test_metrics)

        r = [test_loss.item()]
        r.extend([x for x in test_metrics.values()])
        res.append(r)

        with open("result.csv", 'a+') as f:
            save_str = ",".join([str(x) for x in test_metrics.values()])
            f.write(f"{note},{shot},roberta-{mode},{s_dataset},{test_loss}," + save_str +"\n")

    res = np.array(res).T.tolist()  # type: ignore
    for r in res:
        r.remove(max(r))
        r.remove(min(r))

    merge_res = [np.mean(x) for x in res]
    with open("merge_res.csv", 'a+') as f:
        save_str = ",".join([str(x) for x in merge_res])
        f.write(f"{note},{shot},roberta-{mode},{s_dataset},{t_dataset}" + save_str +"\n")