import GCL.losses as L
from GCL.losses.infonce import choose_lossf
from torch.optim import AdamW
from GCL.models import DualBranchContrast
from ..utils import *
from ..model import *
from ..evaluate import *
from copy import deepcopy
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from functools import partial
from torch.utils.data.dataloader import default_collate
from transformers import (RobertaModel, RobertaTokenizer)
from torch import nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime

def collate_fn(batch, pad_token_id):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return input_ids_padded, attention_masks_padded


def preprocess_text_dataset(text_dataset, tokenizer):
    input_ids = []
    attention_masks = []
    
    for document in text_dataset:
        sentence = document['sentence']
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
        input_ids.append(inputs['input_ids'].squeeze(0))
        attention_masks.append(inputs['attention_mask'].squeeze(0))
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids_padded, attention_masks_padded


class CustomBertModel(nn.Module):
    def __init__(self, model_name):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output


class CustomRoBERTaModel(nn.Module):
    def __init__(self, model_name):
        super(CustomRoBERTaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config = self.roberta.config

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output


def RGSN(ts, args, path, logger, amr_dataset1, text_dataset, model_name):
    epo = args.epoch
    batch_size = args.batch_size
    shuffle = args.shuffle
    device = args.device
    hid_units = args.hid_units
    num_layers = args.num_layer
    interval = args.interval
    factor = args.factor
    pn = args.pn
    sample = args.sample
    lr = args.lr
    mode = args.mode
    loss_method_name = args.lossf
    tau = args.tau

    loss_method = choose_lossf(loss_method_name, tau)
    amr_dataset_train, input_dim, id_dim = amr_dataset1


    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name, cache_dir=None, do_lower_case=False)
    pad_token_id = tokenizer.pad_token_id
    custom_collate_fn = partial(collate_fn, pad_token_id=pad_token_id)
    bert_model = CustomRoBERTaModel(model_name)


    input_ids, attention_masks = preprocess_text_dataset(text_dataset, tokenizer)  
    text_dataset = TensorDataset(input_ids, attention_masks)
    text_feature_transform = nn.Linear(1024, num_layers*hid_units)



    for t in range(1, ts + 1):

        if sample is not None:
            np.random.seed(t)
            sample = min(sample, len(amr_dataset_train))
            amr_train_list = np.random.permutation(range(len(amr_dataset_train)))[: sample].tolist()
            amr_dataset_train = [amr_dataset_train[i] for i in amr_train_list]

        amr_dataloader = GeometricDataLoader(amr_dataset_train, batch_size=batch_size, shuffle=shuffle)
        text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
        
        contrast_model = DualBranchContrast(loss=loss_method, mode=mode).to(device)

        gconv1 = Graph_RGSN(id_dim=id_dim, input_dim=input_dim, hidden_dim=hid_units, num_layers=num_layers, num_relations=12).to(device)

        bert_params = list(bert_model.named_parameters())
        gconv_params = list(gconv1.named_parameters())

        optimizer1 = AdamW([
            {'params': [p for n, p in bert_params if p.requires_grad], 'lr': 1e-5}, 
            {'params': [p for n, p in gconv_params if p.requires_grad], 'lr': 1e-3} ])

        getcore1 = GetCore(amr_dataset_train, 'kcore', pn, factor, device)
        logger.info('Preprocessing is ok.')

        losses = []
        for epoch in tqdm(range(1, epo + 1)):
            loss = train_RGSN(bert_model, tokenizer, gconv1, contrast_model, amr_dataloader, text_dataloader, optimizer1, device, getcore1, text_feature_transform)
            losses.append(loss)
            logger.info("T{}, epoch:{}, loss:{:.4f}".format(t, epoch, loss))

            if epoch % interval == 0:
                save_directory = path + '_' + str(t) + '_' + str(epoch) + '_Pretrained_Model'
                os.makedirs(save_directory, exist_ok=True)
                model_weights_path = os.path.join(save_directory, 'pytorch_model.bin')
                torch.save(bert_model.state_dict(), model_weights_path)
                config_path = os.path.join(save_directory, 'config.json')
                with open(config_path, 'w') as f:
                    f.write(bert_model.config.to_json_string())
                tokenizer.save_pretrained(save_directory)
        epochs = range(1, epo + 1)
        

def train_RGSN(model, tokenizer, gconv, contrast_model, graph_dataloader, text_dataloader, optimizer, device, getcore, 
          text_feature_transform):
    model.train()
    contrast_model.train()
    model.to(device)
    contrast_model.to(device)
    text_feature_transform.to(device)

    epoch_loss = 0
    batch_num = 0
    for graph_data, text_data in zip(graph_dataloader, text_dataloader):
        batch_num += 1
        optimizer.zero_grad() 
        graph_data = graph_data.to(device)      
        text_features = model(text_data[0].to(device), attention_mask=text_data[1].to(device))             
        compressed_text_features = text_feature_transform(text_features)
        edge_type = getcore.get_edge_type(graph_data.edge_index, graph_data.edge_attr)
        _, g1 = gconv(graph_data.x, graph_data.edge_index, edge_type, graph_data.batch)
        if compressed_text_features.shape[0] < graph_data.batch_size:
            continue  
        h1, g1 = [gconv.project(feature) for feature in [compressed_text_features, g1]]
        loss = contrast_model(h1=h1, g1=g1, batch= graph_data.batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(graph_dataloader)