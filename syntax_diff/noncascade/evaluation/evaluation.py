import os
import torch
from evaluate import load
from nltk.util import ngrams
from collections import defaultdict
import spacy
import numpy as np
from transformers import pipeline

def compute_perplexity(all_texts_list, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').type

    valid_texts = [
        x for x in all_texts_list
        if isinstance(x, str) and len(x.strip()) > 0
    ]

    results = perplexity.compute(predictions=valid_texts, model_id=model_id, device=device)
    return results['mean_perplexity']

def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount['unique_words']

def compute_diversity(all_texts_list):
    ngram_range = [2,3,4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1-len(ngram_sets[n])/ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity
    return metrics


def compute_bleu(all_texts_list, human_references):
    bleu = load("bleu")

    human_references = [[ref] for ref in human_references]
    results = bleu.compute(predictions=all_texts_list, references=human_references)
    
    return results['bleu']

def compute_bertscore(all_texts_list, human_references):
    bert = load("bertscore")

    human_references = [[ref] for ref in human_references]
    results = bert.compute(predictions=all_texts_list, references=human_references, lang="en", rescale_with_baseline=True)

    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()
    
    return results

def compute_rouge(all_texts_list, human_references):
    rouge = load("rouge")

    human_references = [[ref] for ref in human_references]
    results = rouge.compute(predictions=all_texts_list, references=human_references)
    
    return results

def compute_mauve(all_texts_list, human_references, model_id):
    torch.cuda.empty_cache() 
    assert model_id == 'gpt2-large'
    assert len(all_texts_list) == len(human_references)
    mauve = load("mauve")

    results = mauve.compute(predictions=all_texts_list, references=human_references, featurize_model_name=model_id, max_text_length=256, device_id=0)
    
    return results.mauve, results.divergence_curve

def compute_classifier(all_texts_list, labels, dataset_name):
    classifier_root = os.environ.get("SYNTAX_DIFFUSION_CLASSIFIER_DIR", "./saved_text_classifier")
    classifier = pipeline('sentiment-analysis', model=os.path.join(classifier_root, dataset_name, "checkpoint"))
    pred_label_list = classifier(all_texts_list)
    pred_list = [label['label'] for label in pred_label_list]
    
    if dataset_name in {"amazon", "amazon_svo"}:
        label2id = {"neg": 0, "pos": 1}
    elif dataset_name in {"yelp", "yelp_svo"}:
        label2id = {"pos": 0, "neg": 1, "neutral": 2}
    elif dataset_name in {"emotion", "emotion_svo"}:
        label2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
    pred_labels = [label2id[label] for label in pred_list]

    base = len(labels)
    correct = 0
    for i in range(base):
        if labels[i] == pred_labels[i]:
            correct += 1
    results = correct/base

    return results


from collections import Counter

def compute_corpus_ngram_overlap(all_texts_list, human_references, ngram_range=[2,3,4], epsilon=1e-8):
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    def tokenize(texts):
        return [[str(token) for token in tokenizer(sentence)] for sentence in texts]
    
    real_tokens = tokenize(human_references)
    gen_tokens = tokenize(all_texts_list)
    
    metrics = {}
    overlap_values = []
    for n in ngram_range:
        real_ngram_counter = Counter()
        gen_ngram_counter = Counter()
        
        for tokens in real_tokens:
            real_ngram_counter.update(ngrams(tokens, n))
        for tokens in gen_tokens:
            gen_ngram_counter.update(ngrams(tokens, n))
        
        common_ngrams = set(real_ngram_counter.keys()).intersection(set(gen_ngram_counter.keys()))
        matched_count = sum(min(real_ngram_counter[ng], gen_ngram_counter[ng]) for ng in common_ngrams)
        
        total_real_ngrams = sum(real_ngram_counter.values())
        overlap_ratio = matched_count / total_real_ngrams if total_real_ngrams > 0 else 0
        metrics[f'{n}gram_overlap'] = overlap_ratio
        overlap_values.append(overlap_ratio)

    weights = [1 / len(overlap_values)] * len(overlap_values)
    log_sum = sum(w * np.log(o + epsilon) for w, o in zip(weights, overlap_values))
    ngram_score = float(np.exp(log_sum))
    metrics["ngram_score"] = ngram_score

    return metrics

def compute_corpus_ngram_syntax_overlap(all_texts_list, human_references, ngram_range=[2,3,4], epsilon=1e-8):
    # Dummy function for extracting syntax features (e.g., POS tags or dependency relations)
    def extract_syntax_features(sentence):
        doc = nlp(sentence)
        syntax_words = [token.pos_ for token in doc]
        syntax_sentence = " ".join(syntax_words)
        return syntax_sentence

    # Extract syntax features
    nlp = spacy.load("en_core_web_sm")
    real_syntax = [extract_syntax_features(sentence) for sentence in human_references]
    gen_syntax = [extract_syntax_features(sentence) for sentence in all_texts_list]

    metrics = {}
    overlap_values = []    
    for n in ngram_range:
        real_syntax_counter = Counter()
        gen_syntax_counter = Counter()

        for features in real_syntax:
            real_syntax_counter.update(ngrams(features, n))
        for features in gen_syntax:
            gen_syntax_counter.update(ngrams(features, n))

        common_syntax_ngrams = set(real_syntax_counter.keys()).intersection(set(gen_syntax_counter.keys()))
        matched_count = sum(min(real_syntax_counter[ng], gen_syntax_counter[ng]) for ng in common_syntax_ngrams)

        total_real_syntax_ngrams = sum(real_syntax_counter.values())
        overlap_ratio = matched_count / total_real_syntax_ngrams if total_real_syntax_ngrams > 0 else 0
        metrics[f'{n}gram_syntax_overlap'] = overlap_ratio
        overlap_values.append(overlap_ratio)

    weights = [1 / len(overlap_values)] * len(overlap_values)
    log_sum = sum(w * np.log(o + epsilon) for w, o in zip(weights, overlap_values))
    syntax_score = float(np.exp(log_sum))
    metrics["syntax_ngram_score"] = syntax_score

    return metrics

def compute_sentence_ngram_syntax_overlap(all_texts_list, real_syntax, ngram_range=[2,3,4], epsilon=1e-8):
    def extract_syntax_features(sentence):
        doc = nlp(sentence)
        syntax_words = [token.pos_ for token in doc]
        syntax_sentence = " ".join(syntax_words)
        return syntax_sentence

    nlp = spacy.load("en_core_web_sm")
    gen_syntax = [extract_syntax_features(sentence) for sentence in all_texts_list]

    metrics = {}
    n_to_scores = {n: [] for n in ngram_range}
    sentence_scores = []

    for pred_tokens, ref_tokens in zip(gen_syntax, real_syntax):
        per_sentence_overlap = []
        for n in ngram_range:
            ref_ngrams = list(ngrams(ref_tokens, n))
            pred_ngrams = list(ngrams(pred_tokens, n))

            if len(ref_ngrams) == 0:
                overlap_ratio = 0.0
            else:
                ref_counter = Counter(ref_ngrams)
                pred_counter = Counter(pred_ngrams)
                common_ngrams = set(ref_counter.keys()).intersection(set(pred_counter.keys()))
                matched_count = sum(min(ref_counter[ng], pred_counter[ng]) for ng in common_ngrams)
                overlap_ratio = matched_count / len(ref_ngrams)

            n_to_scores[n].append(overlap_ratio)
            per_sentence_overlap.append(overlap_ratio)

        weights = [1 / len(per_sentence_overlap)] * len(per_sentence_overlap)
        log_sum = sum(w * np.log(o + epsilon) for w, o in zip(weights, per_sentence_overlap))
        sentence_scores.append(float(np.exp(log_sum)))

    for n in ngram_range:
        metrics[f'{n}gram_syntax_overlap_sent_mean'] = float(np.mean(n_to_scores[n])) if n_to_scores[n] else 0.0
    metrics["syntax_ngram_sent_score"] = float(np.mean(sentence_scores)) if sentence_scores else 0.0

    return metrics
