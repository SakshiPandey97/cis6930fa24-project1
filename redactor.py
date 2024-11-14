import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from warnings import filterwarnings
filterwarnings('ignore')

import logging as python_logging
python_logging.getLogger("torch").setLevel(python_logging.CRITICAL)

import os
import sys
import argparse
import glob
import re
import warnings
from functools import lru_cache
import nltk
from nltk.corpus import wordnet
import spacy
from spacy.matcher import Matcher
from transformers import pipeline
import numpy as np

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def output_stats(stats, stats_type, filename):
    stats_str = f"File: {filename}\n"
    for key, values in stats.items():
        stats_str += f"{key}: {len(values)}\n"
        for item in values:
            stats_str += f"  Text: {item['text']} Start: {item['start']} End: {item['end']}\n"
    if stats_type == "stderr":
        print(stats_str, file=sys.stderr)
    elif stats_type == "stdout":
        print(stats_str)
    else:
        with open(stats_type, "a", encoding='utf-8') as f:
            f.write(stats_str + "\n")

def get_plural(word):
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    elif word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    else:
        return word + 's'

@lru_cache(maxsize=None)
def get_extended_synonyms(concept):
    synonyms = set()
    synsets = wordnet.synsets(concept)
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            synonyms.add(synonym)
            synonyms.add(get_plural(synonym))
            if lemma.derivationally_related_forms():
                for related_lemma in lemma.derivationally_related_forms():
                    related_word = related_lemma.name().replace('_', ' ').lower()
                    synonyms.add(related_word)
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                hypernym_word = lemma.name().replace('_', ' ').lower()
                synonyms.add(hypernym_word)
        for hyponym in syn.hyponyms():
            for lemma in hyponym.lemmas():
                hyponym_word = lemma.name().replace('_', ' ').lower()
                synonyms.add(hyponym_word)
    synonyms.add(concept.lower())
    synonyms.add(get_plural(concept.lower()))
    return synonyms

def redact_names(text, nlp, stats):
    doc = nlp(text)
    redacted_chars = list(text)
    stats.setdefault('NAMES', [])
    
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            stats['NAMES'].append({'text': ent.text, 'start': ent.start_char, 'end': ent.end_char})
            for i in range(ent.start_char, ent.end_char):
                redacted_chars[i] = "█"
    
    def redact_email_names(match):
        username, domain = match.groups()
        stats['NAMES'].append({'text': username, 'start': match.start(1), 'end': match.end(1)})
        return "█" * len(username) + domain
    
    text = re.sub(r'([a-zA-Z0-9._%+-]+)(@[\w.-]+)', redact_email_names, ''.join(redacted_chars))
    
    email_headers = re.findall(r'(From|To):\s*([A-Za-z\s]+)', text)
    for _, name in email_headers:
        for part in name.split():
            text = re.sub(rf'\b{re.escape(part)}\b', lambda m: redact_name_part(m, stats), text)
    
    def redact_name_part(m, stats):
        stats['NAMES'].append({'text': m.group(0), 'start': m.start(), 'end': m.end()})
        return "█" * len(m.group(0))
    
    patterns = {
        r'X-Folder: (.*?)(_Jan|_Feb|_Mar|_Apr|_May|_Jun|_Jul|_Aug|_Sep|_Oct|_Nov|_Dec)\d{4}': "X-Folder: ",
        r'X-Origin: (.*)': "X-Origin: ",
        r'X-FileName: (.*?)(\.nsf)': "X-FileName: ",
        r'/CN=([^/>]+)(?=>)': "/CN=",
        r'</O=ENRON/OU=[^/]+/CN=RECIPIENTS/CN=[^>]+>,\s*([A-Za-z ,]+)(?=(,|$))': ""
    }
    
    for pattern, prefix in patterns.items():
        text = re.sub(pattern, lambda m: redact_generic(m, prefix, stats), text, flags=re.IGNORECASE)
    
    def redact_generic(match, prefix, stats):
        text_to_redact = match.group(1)
        stats.setdefault('NAMES', []).append({'text': text_to_redact, 'start': match.start(1), 'end': match.end(1)})
        redacted = "█" * len(text_to_redact)
        return prefix + redacted
    
    return text

def redact_phone_numbers(text, nlp_sm, matcher, stats):
    doc = nlp_sm(text)
    spans = []
    
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        spans.append((span.start_char, span.end_char, span.text))
    
    additional_pattern = re.compile(
        r'\b\d{3}\.\d{3}\.\d{4}\b|\b\d{3}\s\d{3}\s\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}\b|\b\d{10}\b|\b\d{3}-\d{3}-\d{4}\b'
    )
    
    for match in additional_pattern.finditer(text):
        spans.append((match.start(), match.end(), match.group()))
    
    spans = sorted(spans, key=lambda x: x[0])
    merged_spans = []
    
    for span in spans:
        if not merged_spans or span[0] > merged_spans[-1][1]:
            merged_spans.append(span)
        else:
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span[1]), merged_spans[-1][2])
    
    redacted_chars = list(text)
    stats.setdefault('PHONE_NUMBERS', [])
    
    for start, end, span_text in merged_spans:
        stats['PHONE_NUMBERS'].append({'text': span_text, 'start': start, 'end': end})
        for i in range(start, end):
            redacted_chars[i] = "█"
    
    return ''.join(redacted_chars)

def redact_addresses(text, nlp, stats):
    doc = nlp(text)
    redacted_chars = list(text)
    stats.setdefault('ADDRESSES', [])

    # Redact addresses identified by spaCy
    for ent in doc.ents:
        if ent.label_ in ['LOC', 'GPE', 'FAC', 'ADDRESS']:
            stats['ADDRESSES'].append({'text': ent.text, 'start': ent.start_char, 'end': ent.end_char})
            for i in range(ent.start_char, ent.end_char):
                redacted_chars[i] = "█"

    address_patterns = [
        r'\b\d{1,5}\s\w+(\s\w+){0,4}\b',  
        r'\b[A-Za-z]+\sRoad\b',  
        r'\b[A-Za-z]+\sStreet\b',  
        r'\b[A-Za-z]+\sAvenue\b',  
        r'\b[A-Za-z]+\sBoulevard\b', 
        r'\b[A-Za-z]+\sLane\b',  
        r'\b[A-Za-z]+\sDrive\b',  
        r'\b\d{5}(?:-\d{4})?\b' 
    ]

    for pattern in address_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            stats['ADDRESSES'].append({'text': match.group(), 'start': start, 'end': end})
            for i in range(start, end):
                redacted_chars[i] = "█"

    return ''.join(redacted_chars)

def redact_dates(text, nlp, stats):
    doc = nlp(text)
    redacted_chars = list(text)
    stats.setdefault('DATES', [])
    
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            stats['DATES'].append({'text': ent.text, 'start': ent.start_char, 'end': ent.end_char})
            for i in range(ent.start_char, ent.end_char):
                redacted_chars[i] = "█"
    
    return ''.join(redacted_chars)

def redact_with_huggingface(text, entity_types, ner_pipeline, stats):
    entities = ner_pipeline(text)
    redacted_chars = list(text)
    stats.setdefault('NAMES', [])
    stats.setdefault('ADDRESSES', [])
    stats.setdefault('DATES', [])
    
    for entity in entities:
        if entity['entity_group'] in entity_types:
            start, end = entity['start'], entity['end']
            category = {'PER': 'NAMES', 'LOC': 'ADDRESSES', 'GPE': 'ADDRESSES', 'DATE': 'DATES'}.get(entity['entity_group'])
            if category:
                stats[category].append({'text': entity['word'], 'start': start, 'end': end})
                for i in range(start, end):
                    redacted_chars[i] = "█"
    
    return ''.join(redacted_chars)

def most_similar(word, nlp, topn=10):
    with nlp.select_pipes(enable=["transformer"]):
        word_doc = nlp(word)
    word_vector = word_doc.vector
    if word_vector is None or word_vector.size == 0:
        return []
    similarities = []
    wordnet_words = list(wordnet.words())
    for w in wordnet_words[:10000]:
        if w.isalpha():
            with nlp.select_pipes(enable=["transformer"]):
                w_doc = nlp(w)
            w_vector = w_doc.vector
            if w_vector is not None and w_vector.size > 0:
                sim = np.dot(word_vector, w_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(w_vector))
                similarities.append((w, sim))
    similarities = sorted(similarities, key=lambda item: -item[1])
    return [w for w, sim in similarities[:topn]]

def redact_concepts(text, concepts, nlp, stats):
    doc = nlp(text)
    redacted_chars = list(text)
    stats.setdefault('CONCEPTS', [])
    stats.setdefault('CONCEPT_WORDS', [])
    similar_words_set = set()
    for concept in concepts:
        synonyms = get_extended_synonyms(concept)
        similar_words_set.update(synonyms)
        for syn in synonyms:
            similar_words_set.update(most_similar(syn, nlp))
    similar_words_lower = {word.lower() for word in similar_words_set}
    similar_words_lower.update({concept.lower() for concept in concepts})
    for sentence in doc.sents:
        sentence_contains_concept = False
        for token in sentence:
            token_text_lower = token.text.lower()
            if token_text_lower in similar_words_lower:
                sentence_contains_concept = True
                start, end = token.idx, token.idx + len(token)
                stats['CONCEPT_WORDS'].append({'text': token.text, 'start': start, 'end': end})
                for i in range(start, end):
                    redacted_chars[i] = "█"
        if sentence_contains_concept:
            sentence_start = sentence.start_char
            sentence_end = sentence.end_char
            stats['CONCEPTS'].append({'text': sentence.text, 'start': sentence_start, 'end': sentence_end})
            for i in range(sentence_start, sentence_end):
                if redacted_chars[i] != '\n':
                    redacted_chars[i] = "█"

    return ''.join(redacted_chars)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")
    parser = argparse.ArgumentParser(description='Redact sensitive information from text documents.')
    parser.add_argument('--input', type=str, required=True, nargs='+', help='Glob pattern for input files to redact')
    parser.add_argument('--output', type=str, required=True, help='Directory to store output files')
    parser.add_argument('--names', action='store_true', help='Censor names in the text')
    parser.add_argument('--dates', action='store_true', help='Censor dates in the text')
    parser.add_argument('--phones', action='store_true', help='Censor phone numbers in the text')
    parser.add_argument('--address', action='store_true', help='Censor addresses in the text')
    parser.add_argument('--concept', type=str, nargs='+', help='Redact text related to specific concepts')
    parser.add_argument('--stats', type=str, help='Output redaction statistics to stderr, stdout, or a file')
    args = parser.parse_args()

    stats_type = args.stats.lower() if args.stats and args.stats.lower() in ['stderr', 'stdout'] else args.stats
    nlp_trf = spacy.load('en_core_web_trf')
    nlp_sm = spacy.load('en_core_web_sm')

    if args.phones:
        matcher = Matcher(nlp_sm.vocab)
        phone_patterns = [
            [{"ORTH": "(", "OP": "?"}, {"SHAPE": "ddd"}, {"ORTH": ")", "OP": "?"},
             {"IS_SPACE": True, "OP": "?"}, {"SHAPE": "ddd"}, {"IS_SPACE": True, "OP": "?"},
             {"ORTH": "-", "OP": "?"}, {"IS_SPACE": True, "OP": "?"}, {"SHAPE": "dddd"}],
            [{"TEXT": {"REGEX": r"^\d{3}$"}}, {"ORTH": "-"},
             {"TEXT": {"REGEX": r"^\d{3}$"}}, {"ORTH": "-"},
             {"TEXT": {"REGEX": r"^\d{4}$"}}],
            [{"TEXT": {"REGEX": r"\d{5}"}}, {"IS_SPACE": True, "OP": "?"},
             {"TEXT": {"REGEX": r"\d{6}"}}]
        ]
        matcher.add("PHONE_NUMBER", phone_patterns)

    concepts = args.concept if args.concept else []
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    os.makedirs(args.output, exist_ok=True)

    for file_pattern in args.input:
        for file in glob.glob(file_pattern):
            if os.path.isfile(file):
                stats = {}
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    if args.names:
                        text = redact_names(text, nlp_trf, stats)
                        text = redact_with_huggingface(text, ['PER'], ner_pipeline, stats)
                    
                    if args.phones:
                        text = redact_phone_numbers(text, nlp_sm, matcher, stats)
                    
                    if args.address:
                        text = redact_addresses(text, nlp_trf, stats)
                        text = redact_with_huggingface(text, ['LOC', 'GPE'], ner_pipeline, stats)
                    
                    if args.dates:
                        text = redact_dates(text, nlp_trf, stats)
                        text = redact_with_huggingface(text, ['DATE'], ner_pipeline, stats)
                    
                    if concepts:
                        text = redact_concepts(text, concepts, nlp_trf, stats)
                    
                    output_path = os.path.join(args.output, os.path.basename(file) + ".censored")
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(text)
                    
                    if stats_type:
                        output_stats(stats, stats_type, file)
                
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
