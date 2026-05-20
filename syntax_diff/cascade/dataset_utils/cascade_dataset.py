import os
import spacy
import logging
import pathlib
from collections import Counter

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM
from dataset_utils.flan_collator import DataCollatorForFlanLM


def get_data_root():
    return os.environ.get(
        "SYNTAX_DIFFUSION_DATA_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets")),
    )


def get_dataset(dataset_name, cascade=None):
    data_root = get_data_root()
    if dataset_name == 'amazon':
        data_path = os.path.join(data_root, 'amazon')
        dataset = load_dataset('json', data_files={f'{split}': os.path.join(data_path, f'{split}_data.json') for split in ['train', 'valid']})
        dataset = process_amazon(dataset, cascade)
    elif dataset_name == 'roc':
        data_path = os.path.join(data_root, 'ROCstory')
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(data_path, f'roc_{split}.json') for split in ['train', 'valid']})
        dataset = process_roc(dataset, cascade)
    elif dataset_name == 'yelp':
        data_path = os.path.join(data_root, 'yelp')
        dataset = load_dataset('json', data_files={f'{split}': os.path.join(data_path, f'{split}_data.json') for split in ['train', 'valid']})
        dataset = process_yelp(dataset, cascade)
    elif dataset_name == 'emotion':
        data_path = os.path.join(data_root, 'emotion')
        dataset = load_dataset('json', data_files={f'{split}': os.path.join(data_path, f'{split}_data.json') for split in ['train', 'valid']})
        dataset = process_emotion(dataset, cascade)
    else:
        raise NotImplementedError
    return dataset

def process_amazon(dataset, cascade):
    def extract_amazon_text(example):
        sentence = example['text']
        doc = nlp(sentence)
        syntax_words = [token.pos_ for token in doc]
        if cascade:
            syntax_sentence = " ".join(syntax_words)
            return {'text': sentence, 'syntax': syntax_sentence}
        else:
            sentence = " ".join(syntax_words)
            return {'text': sentence}

    nlp = spacy.load("en_core_web_sm")
    dataset = dataset.map(extract_amazon_text,)
    dataset = dataset.shuffle(seed=77)
    dataset = dataset.remove_columns(['class'])
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=10000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_yelp(dataset, cascade):
    def extract_yelp_text(example):
        sentence = example['text']
        doc = nlp(sentence)
        syntax_words = [token.pos_ for token in doc]
        if cascade:
            syntax_sentence = " ".join(syntax_words)
            return {'text': sentence, 'syntax': syntax_sentence}
        else:
            sentence = " ".join(syntax_words)
            return {'text': sentence}

    nlp = spacy.load("en_core_web_sm")
    dataset = dataset.map(extract_yelp_text,)
    dataset = dataset.shuffle(seed=77)
    dataset = dataset.remove_columns(['class'])
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_emotion(dataset, cascade):
    def extract_emotion_text(example):
        sentence = example['text']
        doc = nlp(sentence)
        syntax_words = [token.pos_ for token in doc]
        if cascade:
            syntax_sentence = " ".join(syntax_words)
            return {'text': sentence, 'syntax': syntax_sentence}
        else:
            sentence = " ".join(syntax_words)
            return {'text': sentence}

    nlp = spacy.load("en_core_web_sm")
    dataset = dataset.map(extract_emotion_text,)
    dataset = dataset.shuffle(seed=77)
    dataset = dataset.remove_columns(['class'])
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_roc(dataset, cascade):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentence = text[2:-2]

        doc = nlp(sentence)
        syntax_words = [token.pos_ for token in doc]
        if cascade:
            syntax_sentence = " ".join(syntax_words)
            return {'text': sentence, 'syntax': syntax_sentence}
        else:
            sentence = " ".join(syntax_words)
            return {'text': sentence}
    
    nlp = spacy.load("en_core_web_sm")
    dataset = dataset.map(extract_roc_text,)
    dataset = dataset.shuffle(seed=77)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset


def get_dataloader(args, dataset, model_config, text_tokenizer, syntax_tokenizer, max_seq_len,
                   shuffle=False, cascade=None):
    def tokenization(example):
        if cascade:
            syntax = example['syntax']
            target = example['text']
            syntax_inputs = syntax_tokenizer(syntax, padding="max_length", truncation=True, max_length=max_seq_len)
            model_inputs = text_tokenizer(target, padding="max_length", truncation=True, max_length=max_seq_len)
            # Add model target to model inputs
            for k in syntax_inputs.keys():
                model_inputs[f'syntax_{k}'] = syntax_inputs[k]
            return model_inputs
        else:
            target = example["text"]
            model_inputs = syntax_tokenizer(target, padding="max_length", truncation=True, max_length=max_seq_len)
            return model_inputs
    
    if cascade:
        tokenizer = text_tokenizer
    else:
        tokenizer = syntax_tokenizer

    if 'mbart' in args.enc_dec_model:
        collate_fn=default_data_collator
    elif 'bart' in args.enc_dec_model:
        collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
    elif 't5' in args.enc_dec_model:
        collate_fn=DataCollatorForFlanLM(tokenizer)
    else:
        raise NotImplementedError
    
    if cascade:
        dataset = dataset.map(tokenization, remove_columns=['text', 'syntax'], batched=True, num_proc=None)
    else:
        dataset = dataset.map(tokenization, remove_columns=['text'], batched=True, num_proc=None)
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 0
        )
    return dl

def get_class_dataloader(args, dataset, model_config, text_tokenizer, syntax_tokenizer, max_seq_len,
                   batch_size, shuffle=False):
    def tokenization(example):
        syntax = example['syntax']
        target = example['text']
        syntax_inputs = syntax_tokenizer(syntax, padding="max_length", truncation=True, max_length=max_seq_len)
        model_inputs = text_tokenizer(target, padding="max_length", truncation=True, max_length=max_seq_len)
        # Add model target to model inputs
        for k in syntax_inputs.keys():
            model_inputs[f'syntax_{k}'] = syntax_inputs[k]
        return model_inputs

    tokenizer = text_tokenizer

    if 'mbart' in args.enc_dec_model:
        collate_fn=default_data_collator
    elif 'bart' in args.enc_dec_model:
        collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
    elif 't5' in args.enc_dec_model:
        collate_fn=DataCollatorForFlanLM(tokenizer)
    else:
        raise NotImplementedError
    
    dataset = dataset.map(tokenization, remove_columns=['text', 'syntax'], batched=True, num_proc=None)

    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 0
        )
    return dl

def create_tokenizer(path):
    from transformers import PreTrainedTokenizerFast
    
    logging.info(f"Loading tokenizer from {path}/syntax-pos.json")
    file_path = f"{str(pathlib.Path(path))}/syntax-pos.json"

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=file_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        padding_side="right",
    )

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    return tokenizer
