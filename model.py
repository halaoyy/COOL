
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaModel, BertModel, RobertaForMaskedLM
from torch.nn.functional import cross_entropy
import os
import random
import layer


# reference: https://huggingface.co/transformers/model_doc/roberta.html
class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size: int = 400):  # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], hidden_size)
        self.dropout = nn.Dropout(config["dropout"])
        self.out_proj = nn.Linear(hidden_size, config["num_labels"])

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SCL(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None):
        inrep_1.cuda()
        inrep_2.cuda()
        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 is None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = torch.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = torch.diag(cosine_similarity)
            cos_diag = torch.diag_embed(diag)  # bs,bs

            label = torch.unsqueeze(label_1, -1)
            if label.shape[0] == 1:
                cos_loss = torch.zeros(1)
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = torch.cat((label, label), -1)
                    else:
                        label_mat = torch.cat((label_mat, label), -1)  # bs, bs

                mid_mat_ = (label_mat.eq(label_mat.t()))
                mid_mat = mid_mat_.float()

                cosine_similarity = (cosine_similarity - cos_diag) / self.temperature  # the diag is 0
                mid_diag = torch.diag_embed(torch.diag(mid_mat))
                mid_mat = mid_mat - mid_diag

                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.bool(), -float('inf'))  # mask the diag

                cos_loss = torch.log(
                    torch.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1

                cos_loss = cos_loss * mid_mat

                cos_loss = torch.sum(cos_loss, dim=1) / (torch.sum(mid_mat, dim=1) + 1e-10)  # bs
        else:
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = torch.cat((inrep_2, pad), 0)
                    label_2 = torch.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = torch.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = torch.unsqueeze(label_1, -1)
            label_1_mat = torch.cat((label_1, label_1), -1)
            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = label_1_mat
                else:
                    label_1_mat = torch.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = torch.unsqueeze(label_2, -1)
            label_2_mat = torch.cat((label_2, label_2), -1)
            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = label_2_mat
                else:
                    label_2_mat = torch.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = torch.log(torch.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat  # find the sample with the same label
            cos_loss = torch.sum(cos_loss, dim=1) / (torch.sum(mid_mat, dim=1) + 1e-10)

        cos_loss = -torch.mean(cos_loss, dim=0)

        return cos_loss

class COOL(nn.Module):
    def __init__(self,
                 config,
                 bert_config,
                 trainer_config,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 with_learnable_emb=True,
                 with_answer_weights=True,
                 with_position_weights=False,
                 num_learnable_token=2,
                 zero_shot=False,
                 fine_tune_all=True):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('../cross_disinformation_detection/roberta-base')
        self.masklm = RobertaLMHead(bert_config)
        self.trainer_config = trainer_config
        self.loss_fn = cross_entropy
        self.scl = SCL(temperature=0.1)

        if not fine_tune_all:  # freeze the pretrained encoder  # default: True
            for param in self.roberta.base_model.parameters():  # type: ignore
                param.requires_grad = False
            self.roberta.embeddings.word_embeddings.requires_grad = True

        self.vocab_size = bert_config.vocab_size
        self.mask_token_id = mask_token_id

        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids

        # when in zero shot condition, simply sum over all ids
        self.zero_shot = zero_shot
        if zero_shot:
            # with_learnable_emb = False
            with_answer_weights = False

        if with_answer_weights:
            # assume weights follow a uniform distribution
            self.positive_weights = nn.Parameter(torch.rand(
                len(positive_token_ids)), requires_grad=True)
            self.negative_weights = nn.Parameter(torch.rand(
                len(negative_token_ids)), requires_grad=True)
        else:
            self.positive_weights = nn.Parameter(torch.ones(
                len(positive_token_ids)), requires_grad=False)
            self.negative_weights = nn.Parameter(torch.ones(
                len(negative_token_ids)), requires_grad=False)

        if with_position_weights:  # randomly select a [x y]-liked weight vector, x y uniformed sampled from [0, 1]
            self.position_weights = nn.Parameter(
                torch.rand(2), requires_grad=True)
        else:
            self.position_weights = nn.Parameter(
                torch.ones(2), requires_grad=False)

        self.learnable_tokens = - 1
        self.knowledgeable_tokens = - 2  # set tokens for knowledgeable soft prompt
        self.num_learnable_token = num_learnable_token

        hidden_size = config['hidden_size']
        dropout = config['dropout']
        if with_learnable_emb:
            self.learnable_token_emb = nn.Embedding(
                num_embeddings=self.num_learnable_token, embedding_dim=hidden_size)
        else:
            self.learnable_token_emb = None
        # self.learnable_token_ffn = nn.Linear(in_features=300, out_features=768)

        # self.entity_project = nn.Linear(in_features=768, out_features=300)
        self.entity_conv1 = nn.Conv2d(1, hidden_size, (3, hidden_size))

        self.entity_gru = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size//2, bidirectional=True, batch_first=True, dropout=dropout)

        self.norm = nn.LayerNorm(normalized_shape=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Linear(3*hidden_size, hidden_size)
        self.Multihead_SelfAttention = torch.nn.MultiheadAttention(hidden_size, num_heads=1, dropout=0.1)
        self.Multihead_SelfAttention2 = torch.nn.MultiheadAttention(hidden_size, num_heads=1, dropout=0.1)
        self.pos_masked_attn = layer.Attention(hidden_size, "dot-linear")
        self.neg_masked_attn = layer.Attention(hidden_size, "dot-linear")
        self.coaid_weights = torch.tensor([1.0, 1.0]).to(self.trainer_config['device'])
        self.ffn = nn.Linear(3*hidden_size, 2)
        w = torch.empty((2, 50265))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)

    def forward(self, t_input_ids, t_attention_mask, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_labels, t_texts_emb,
                s_input_ids=None, s_attention_mask=None, s_entity_embs=None, s_entity_lens=None, s_entity_desc_embs=None, s_entity_neigh_embs=None, s_labels=None, s_texts_emb=None):
        
        t_batch_size, t_seq_len = t_input_ids.size()
        t_mask_ids = (t_input_ids == self.mask_token_id).nonzero(as_tuple=True)
        if torch.isnan(t_entity_embs).any().item() or torch.isnan(t_entity_neigh_embs).any().item():
            print("input contains nan")

        if s_input_ids is not None:
            s_batch_size, s_seq_len = s_input_ids.size()
            s_mask_ids = (s_input_ids == self.mask_token_id).nonzero(as_tuple=True)

        if self.learnable_token_emb is None:
            # roberta
            roberta_outputs = self.roberta(
                input_ids, attention_mask)  # type: ignore
        else:  # if self.learnable_token_emb is not None and the entity list is not empty
            t_input_emb = self.encode_entities(t_input_ids, t_entity_embs, t_entity_lens, t_entity_desc_embs, t_entity_neigh_embs, t_texts_emb)
            t_roberta_outputs = self.roberta(
                inputs_embeds=t_input_emb, attention_mask=t_attention_mask)  # type: ignore

            if s_input_ids is not None:
                s_input_emb = self.encode_entities(s_input_ids, s_entity_embs, s_entity_lens, s_entity_desc_embs,
                                                   s_entity_neigh_embs, s_texts_emb)
                s_roberta_outputs = self.roberta(
                    inputs_embeds=s_input_emb, attention_mask=s_attention_mask)  # type: ignore

        t_sequence_output = t_roberta_outputs[0]
        t_logits = self.masklm(t_sequence_output)
        _, _, t_vocab_size = t_logits.size()

        t_mask_logits_scl = t_logits[t_mask_ids]  # batch_size, vocab_size
        t_mask_logits = F.log_softmax(t_mask_logits_scl, dim=1)
        # batch_size, mask_num, vocab_size
        t_mask_logits = t_mask_logits.view(t_batch_size, -1, t_vocab_size)
        _, t_mask_num, _ = t_mask_logits.size()

        # batch_size, mask_num, vocab_size
        t_mask_logits = (t_mask_logits.transpose(1, 2) *
                         self.position_weights[:t_mask_num]).transpose(1, 2)

        t_mask_logits = t_mask_logits.sum(dim=1).squeeze(
            1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        # batch_size, len(positive_token_ids)
        t_positive_logits = t_mask_logits[:,
                            self.positive_token_ids] * positive_weight
        # batch_size, len(negative_token_ids)
        t_negative_logits = t_mask_logits[:,
                            self.negative_token_ids] * negative_weight

        t_positive_logits = t_positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        t_negative_logits = t_negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        t_cls_logits = torch.cat([t_positive_logits, t_negative_logits], dim=1)

        if s_input_ids is not None:
            # s_sequence_output, s_other_outputs = s_roberta_outputs
            s_sequence_output = s_roberta_outputs[0]
            s_logits = self.masklm(s_sequence_output)
            _, _, s_vocab_size = s_logits.size()

            s_mask_logits_scl = s_logits[s_mask_ids]
            s_mask_logits = F.log_softmax(s_mask_logits_scl, dim=1)
            s_mask_logits = s_mask_logits.view(s_batch_size, -1, s_vocab_size)
            _, s_mask_num, _ = s_mask_logits.size()

            s_mask_logits = (s_mask_logits.transpose(1, 2) *
                             self.position_weights[:s_mask_num]).transpose(1, 2)

            s_mask_logits = s_mask_logits.sum(dim=1).squeeze(
                1)  # batch_size, vocab_size

            positive_weight = F.softmax(self.positive_weights, dim=0)
            negative_weight = F.softmax(self.negative_weights, dim=0)

            s_positive_logits = s_mask_logits[:,
                                self.positive_token_ids] * positive_weight
            s_negative_logits = s_mask_logits[:,
                                self.negative_token_ids] * negative_weight

            s_positive_logits = s_positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
            s_negative_logits = s_negative_logits.sum(1).unsqueeze(1)  # batch_size, 1
            s_cls_logits = torch.cat([s_positive_logits, s_negative_logits], dim=1)

        if s_input_ids is not None:
            t_ce_loss = self.loss_fn(input=t_cls_logits, target=t_labels, weight=self.coaid_weights)
            s_ce_loss = self.loss_fn(input=s_cls_logits, target=s_labels, weight=self.coaid_weights)
            s_scl_loss = self.scl(s_mask_logits_scl, s_mask_logits_scl, s_labels)
            t_scl_loss = self.scl(s_mask_logits_scl, t_mask_logits_scl, s_labels, t_labels)

            alp = 0.5
            # data augmentation and adversarial learning
            t_mask_logits_scl.retain_grad()
            t_ce_loss.backward(retain_graph=True)
            unnormalized_noise = t_mask_logits_scl.grad.detach_()
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            norm = unnormalized_noise.norm(p=2, dim=-1)
            normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)

            noise_norm = 1.5

            target_noise = noise_norm * normalized_noise
            noise_t_mask_logits_scl = target_noise + t_mask_logits_scl
            noise_scl_loss = self.scl(s_mask_logits_scl, noise_t_mask_logits_scl, s_labels, t_labels)

            # noise ce loss
            noise_t_mask_logits = F.log_softmax(noise_t_mask_logits_scl, dim=1)
            noise_t_mask_logits = noise_t_mask_logits.view(t_batch_size, -1, t_vocab_size)
            _, noise_t_mask_num, _ = noise_t_mask_logits.size()

            noise_t_mask_logits = (noise_t_mask_logits.transpose(1, 2) *
                                   self.position_weights[:noise_t_mask_num]).transpose(1, 2)

            noise_t_mask_logits = noise_t_mask_logits.sum(dim=1).squeeze(
                1)  # batch_size, vocab_size

            positive_weight = F.softmax(self.positive_weights, dim=0)
            negative_weight = F.softmax(self.negative_weights, dim=0)

            noise_t_positive_logits = noise_t_mask_logits[:,
                                      self.positive_token_ids] * positive_weight
            noise_t_negative_logits = noise_t_mask_logits[:,
                                      self.negative_token_ids] * negative_weight

            noise_t_positive_logits = noise_t_positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
            noise_t_negative_logits = noise_t_negative_logits.sum(1).unsqueeze(1)  # batch_size, 1
            noise_t_cls_logits = torch.cat([noise_t_positive_logits, noise_t_negative_logits], dim=1)
            noise_ce_loss = self.loss_fn(input=noise_t_cls_logits, target=t_labels, weight=self.coaid_weights)

            noise_loss = (1 - alp) * noise_ce_loss + alp * noise_scl_loss
            
            total_loss = (((1 - alp) * s_ce_loss + alp * s_scl_loss) + (
                    (1 - alp) * t_ce_loss + alp * t_scl_loss) + noise_loss) / 3

            return t_cls_logits, total_loss
        else:
            return t_cls_logits

    def encode_entities(self, input_ids, entity_embs, entity_lens, entity_desc_embs, entity_neigh_embs, texts_emb):
        """
        input_ids: [batch_size, seq_len]
        entity_embs: [batch_size, entity_num, embed_dim]
        entity_neigh_embs: [batch_size, entity_num, neigh_num, embed_dim]
        """
        batch_size, seq_len = input_ids.size()

        input_emb_mean = texts_emb
        input_emb_mean = input_emb_mean.view(batch_size, 1, 1, -1)
        entity_neigh_embs = torch.max(entity_neigh_embs * input_emb_mean.repeat(1, entity_neigh_embs.shape[1], entity_neigh_embs.shape[2], 1), dim=2)[0]
        entity_embs = torch.cat((entity_neigh_embs, entity_embs, entity_desc_embs), -1)
        entity_embs = self.feed_forward(entity_embs)  # [B, L, H]
        
        #################### entity embedding learning #####################
        ###### Postive & Negative masked attention
        input_emb_mean = input_emb_mean.squeeze()  # [B, H]
        pos_a = self.pos_masked_attn(input_emb_mean, entity_embs, entity_lens)
        pos_a = torch.reshape(pos_a, [batch_size, 1, -1])
        e1 = torch.bmm(pos_a, entity_embs).squeeze()  # [B, H]
        neg_a = self.neg_masked_attn(input_emb_mean, entity_embs, entity_lens, "neg")
        neg_a = torch.reshape(neg_a, [batch_size, 1, -1])
        e2 = torch.bmm(neg_a, entity_embs).squeeze()  # [B, H]      
        
        ######## extract entity-enhanced learnable prompts
        replace_embeds = self.learnable_token_emb(torch.arange(
            self.num_learnable_token).to(self.trainer_config['device']))
        replace_embeds = replace_embeds.unsqueeze(0).repeat(
            batch_size, 1, 1)
        eo1, eo2 = replace_embeds[:, 0], replace_embeds[:, 1]
        e1, e2 = e1 + eo1, e2 + eo2
        
        replace_embeds = torch.cat(
            [e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)
        replace_embeds = self.norm(replace_embeds)

        add_ids = (input_ids == self.learnable_tokens).nonzero(as_tuple=True)
        input_ids[add_ids] = self.mask_token_id
        input_emb = self.roberta.embeddings.word_embeddings(input_ids)  # type: ignore
        input_emb[add_ids] = replace_embeds.view(-1, 768)
        input_emb = input_emb.view(batch_size, seq_len, -1)
        return input_emb

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, hidden_size: int = 200):   # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)  # type: ignore
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x

    def _tie_weights(self):
        self.bias = self.decoder.bias
