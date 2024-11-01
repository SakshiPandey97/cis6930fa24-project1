import os
import argparse
import spacy
import glob
import re
import warnings
from transformers import pipeline
from spacy.tokens import Span
from spacy.matcher import Matcher
import numpy as np
import nltk
from nltk.corpus import wordnet
from spacy.matcher import PhraseMatcher
import torch

def redact_names(text, nlp):
    doc = nlp(text)
    redacted_chars = list(text)
    redacted_positions = set()
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            for i in range(ent.start_char, ent.end_char):
                if i not in redacted_positions:
                    redacted_chars[i] = "█"
                    redacted_positions.add(i)
    redacted_text = ''.join(redacted_chars)
    redacted_text = re.sub(r'([a-zA-Z0-9._%+-]+)(@enron\.com)', lambda m: "█" * len(m.group(1)) + m.group(2), redacted_text)
    email_names = re.findall(r'From: ([a-zA-Z.]+)@enron\.com|To: ([a-zA-Z.]+)@enron\.com', redacted_text)
    for from_name, to_name in email_names:
        for name in filter(None, [from_name, to_name]):
            name_parts = name.split('.')
            for part in name_parts:
                if part:
                    redacted_text = re.sub(rf'\b{part}\b', lambda m: "█" * len(m.group(0)), redacted_text, flags=re.IGNORECASE)
    redacted_text = re.sub(r'X-Folder: (.*?)(_Jan|_Feb|_Mar|_Apr|_May|_Jun|_Jul|_Aug|_Sep|_Oct|_Nov|_Dec)\d{4}', lambda m: "X-Folder: " + "█" * len(m.group(1)) + m.group(2), redacted_text)
    redacted_text = re.sub(r'X-Origin: (.*)', lambda m: "X-Origin: " + "█" * len(m.group(1)), redacted_text)
    redacted_text = re.sub(r'X-FileName: (.*?)(\.nsf)', lambda m: "X-FileName: " + "█" * len(m.group(1)) + m.group(2), redacted_text)
    redacted_text = re.sub(r'([a-zA-Z0-9._%+-]+)(@[a-zA-Z0-9.-]+)', lambda m: "█" * len(m.group(1)) + m.group(2), redacted_text)
    redacted_text = re.sub(r'/CN=([^/>]+)(?=>)', lambda m: "/CN=" + "█" * len(m.group(1)), redacted_text, flags=re.IGNORECASE)
    redacted_text = re.sub(r'</O=ENRON/OU=[^/]+/CN=RECIPIENTS/CN=[^>]+>, ([A-Za-z ,]+)(?=(,|$))', lambda m: "█" * len(m.group(1)), redacted_text, flags=re.IGNORECASE)
    return redacted_text

def redact_phone_numbers(text, nlp_sm, matcher):
    doc = nlp_sm(text)
    matches = matcher(doc)
    redacted_chars = list(text)
    redacted_positions = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        start_char = span.start_char
        end_char = span.end_char
        for i in range(start_char, end_char):
            if i not in redacted_positions:
                redacted_chars[i] = "█"
                redacted_positions.add(i)
    redacted_text = ''.join(redacted_chars)
    additional_phone_pattern = re.compile(
        r"""
        \b\d{3}\.\d{3}\.\d{4}\b
        |
        \b\d{3}\s\d{3}\s\d{4}\b
        """,
        re.VERBOSE
    )
    def replace_with_blocks(match):
        return "█" * len(match.group())
    redacted_text = additional_phone_pattern.sub(replace_with_blocks, redacted_text)
    return redacted_text

def redact_addresses(text, nlp):
    doc = nlp(text)
    redacted_chars = list(text)
    redacted_positions = set()
    for ent in doc.ents:
        if ent.label_ in ['LOC', 'GPE', 'FAC']:
            for i in range(ent.start_char, ent.end_char):
                if i not in redacted_positions:
                    redacted_chars[i] = "█"
                    redacted_positions.add(i)
    redacted_text = ''.join(redacted_chars)
    return redacted_text

def redact_dates(text, nlp):
    doc = nlp(text)
    redacted_chars = list(text)
    redacted_positions = set()
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            for i in range(ent.start_char, ent.end_char):
                if i not in redacted_positions:
                    redacted_chars[i] = "█"
                    redacted_positions.add(i)
    redacted_text = ''.join(redacted_chars)
    return redacted_text

def redact_with_huggingface(text, entity_types, ner_pipeline):
    entities = ner_pipeline(text)
    redacted_chars = list(text)
    redacted_positions = set()
    for entity in entities:
        if entity['entity_group'] in entity_types:
            start = entity['start']
            end = entity['end']
            for i in range(start, end):
                if i not in redacted_positions:
                    redacted_chars[i] = "█"
                    redacted_positions.add(i)
    redacted_text = ''.join(redacted_chars)
    return redacted_text

def get_synonyms(concepts):
    synonyms = set()
    for concept in concepts:
        for syn in wordnet.synsets(concept):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
                for der in lemma.derivationally_related_forms():
                    synonyms.add(der.name())
        morphs = [wordnet.morphy(concept, pos) for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]]
        synonyms.update(filter(None, morphs))
    synonyms = set([syn.replace('_', ' ').lower() for syn in synonyms])
    return list(synonyms)

def redact_concepts(text, concepts, nlp):
    doc = nlp(text)
    redacted_chars = list(text)
    redacted_positions = set()
    concept_docs = [nlp(concept) for concept in concepts]
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(concept) for concept in concepts]
    matcher.add("CONCEPTS", patterns)
    for sent in doc.sents:
        sent_contains_concept = False
        matches = matcher(sent)
        if matches:
            sent_contains_concept = True
        else:
            for concept_doc in concept_docs:
                if sent.vector_norm and concept_doc.vector_norm:
                    similarity = sent.similarity(concept_doc)
                    if similarity >= 0.65:
                        sent_contains_concept = True
                        break
        if sent_contains_concept:
            start_char = sent.start_char
            end_char = sent.end_char
            for i in range(start_char, end_char):
                if i not in redacted_positions:
                    if redacted_chars[i] != '\n':
                        redacted_chars[i] = "█"
                    redacted_positions.add(i)
    redacted_text = ''.join(redacted_chars)
    return redacted_text

def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")

    parser = argparse.ArgumentParser(description='Redact sensitive information from text documents.')
    parser.add_argument('--input', type=str, required=True, nargs='+', help='Require glob pattern for input files to redact')
    parser.add_argument('--output', type=str, required=True, help='Directory to store output files')
    parser.add_argument('--names', action='store_true', help='Censor names in the text')
    parser.add_argument('--dates', action='store_true', help='Censor dates in the text')
    parser.add_argument('--phones', action='store_true', help='Censor phone numbers in the text')
    parser.add_argument('--address', action='store_true', help='Censor addresses in the text')
    parser.add_argument('--concept', action='append', help='Redact text related to specific concepts')
    parser.add_argument('--stats', type=str, choices=['stderr', 'stdout', 'file'], help='Output redaction statistics to stderr, stdout, or a file')
    args = parser.parse_args()

    nlp_trf = spacy.load('en_core_web_trf')

    # Load SpaCy small model and setup Matcher only if phone numbers need to be redacted
    if args.phones:
        nlp_sm = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp_sm.vocab)
        patterns = []
        pattern1 = [
            {"ORTH": "(", "OP": "?"},
            {"SHAPE": "ddd"},
            {"ORTH": ")", "OP": "?"},
            {"IS_SPACE": True, "OP": "?"},
            {"SHAPE": {"REGEX": "ddd|dddd|ddddd"}},
            {"IS_SPACE": True, "OP": "?"},
            {"ORTH": "-", "OP": "?"},
            {"IS_SPACE": True, "OP": "?"},
            {"SHAPE": {"REGEX": "ddd|dddd|ddddd"}},
        ]
        patterns.append(pattern1)
        us_pattern2 = [
            {"TEXT": {"REGEX": r"^\d{3}$"}},
            {"ORTH": "-"},
            {"TEXT": {"REGEX": r"^\d{3}$"}},
            {"ORTH": "-"},
            {"TEXT": {"REGEX": r"^\d{4}$"}},
        ]
        patterns.append(us_pattern2)
        pattern6 = [
            {"TEXT": {"REGEX": r"\d{5}"}},
            {"IS_SPACE": True, "OP": "?"},
            {"TEXT": {"REGEX": r"\d{6}"}},
        ]
        patterns.append(pattern6)
        matcher.add("PHONE_NUMBER", patterns)

    # Prepare concepts and synonyms outside the loop
    if args.concept:
        concepts = args.concept
        synonyms = get_synonyms(concepts)
        all_concepts = list(set(concepts + synonyms))
    else:
        all_concepts = []

    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    # Ensure the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Iterate over input files
    for file_pattern in args.input:
        for file in glob.glob(file_pattern):
            if file.endswith(".txt"):
                try:
                    with open(file, 'r') as f:
                        text = f.read()
                    if args.names:
                        text = redact_names(text, nlp_trf)
                        text = redact_with_huggingface(text, ['PER'], ner_pipeline)
                    if args.phones:
                        text = redact_phone_numbers(text, nlp_sm, matcher)
                    if args.address:
                        text = redact_addresses(text, nlp_trf)
                        text = redact_with_huggingface(text, ['LOC', 'GPE'], ner_pipeline)
                    if args.dates:
                        text = redact_dates(text, nlp_trf)
                        text = redact_with_huggingface(text, ['DATE'], ner_pipeline)
                    if args.concept:
                        text = redact_concepts(text, all_concepts, nlp_trf)
                    output_path = os.path.join(args.output, os.path.basename(file) + ".censored")
                    with open(output_path, 'w') as out_f:
                        out_f.write(text)
                    print(f"File processed and saved to: {output_path}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    main()
