# The Redactor

**Author**: Sakshi Pandey
**Email**: sakshi.pandey@ufl.edu

## Introduction

The Redactor is a text redaction tool designed to censor sensitive information from plain text documents. It can redact names, dates, phone numbers, addresses, and any text related to specific concepts. The tool utilizes Natural Language Processing (NLP) models and techniques, including SpaCy, NLTK, and Hugging Face's BERT-based models, to ensure thorough and accurate redaction.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Flags and Parameters](#flags-and-parameters)
- [Design Decisions](#design-decisions)
- [Models and Libraries Used](#models-and-libraries-used)
- [Tests](#tests)
- [Bugs and Assumptions](#bugs-and-assumptions)
- [Acknowledgements](#acknowledgements)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SakshiPandey97/cis6930fa24-project1.git
   cd cis6930fa24-project1
   ```

2. **Create a virtual environment using Pipenv**:

   ```bash
   pipenv install
   pipenv shell
   ```

3. **Download necessary NLTK data**:

   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

4. **Ensure SpaCy models are installed**:

   The `Pipfile` includes the SpaCy models as dependencies, and they should be installed automatically.

## Usage

Run the script using the command line:

```bash
pipenv run python redactor.py --input [INPUT_FILES] --output [OUTPUT_DIRECTORY] [FLAGS]
```

**Example**:

```bash
pipenv run python redactor.py --input '*.txt' \
    --names --dates --phones --address \
    --concept 'bank' \
    --output 'redacted_output' \
    --stats stderr
```

This command will read all `.txt` files in the current directory, redact specified sensitive information, save the redacted files to the `redacted_output` directory, and output statistics to `stderr`.

## Flags and Parameters

- **`--input`**: Glob pattern for input files to redact.
- **`--output`**: Directory to store output files.
- **`--names`**: Censor personal names using SpaCy's NER and custom regex.
- **`--dates`**: Censor written dates using SpaCy's NER.
- **`--phones`**: Censor phone numbers using SpaCy `Matcher` and regex.
- **`--address`**: Censor addresses using SpaCy's NER.
- **`--concept`**: Redact text related to specific concepts, expanding synonyms with NLTK WordNet.
- **`--stats`**: Output redaction statistics to `stderr`, `stdout`, or a file.

## Design Decisions

Machine learning models do not always capture every instance of sensitive information, leading to potential gaps in redaction. To ensure thorough coverage, The Redactor combines multiple techniques:

- **SpaCy Named Entity Recognition (NER)**: Used for standard named entities like names, dates, and locations.
- **Hugging Face BERT-based NER**: Provides additional accuracy for certain entities.
- **Regex**: Handles patterns that are hard to capture with machine learning models, such as emails and specific metadata fields.

This hybrid approach ensures comprehensive redaction by covering the limitations of each individual method.

## Models and Libraries Used

- **NLTK WordNet**: Extracts synonyms and related forms of concepts.
- **Regex**: Matches specific patterns like email addresses and phone numbers.
- **SpaCy**: Uses `en_core_web_trf` for advanced NLP tasks and `en_core_web_sm` for efficient pattern matching.
- **Hugging Face Transformers (BERT)**: `dslim/bert-base-NER` for additional NER capabilities.

## Tests

The `tests/` directory contains unit tests for each redaction function. To run the tests:

```bash
pipenv run python -m pytest
```

**Test Descriptions**:
- **`test_names.py`**: Tests name redaction.
- **`test_phones.py`**: Tests phone number redaction.
- **`test_address.py`**: Tests address redaction.
- **`test_concepts.py`**: Tests concept-based redaction.
- **`test_dates.py`**: Tests date redaction.

## Bugs and Assumptions

- **Bugs**:
  - **Address Redaction**: The address redaction functionality is weak and may miss certain addresses, especially those not conforming to standard formats. To improve accuracy, a labeled dataset with diverse address examples would be needed to train a custom model. Since it relies on machine learning, there is always the possibility of missing entities that are not well represented in the training data.
  - **Phone Number Coverage**: The tool primarily redacts US and UK phone numbers. Other formats may not be recognized and therefore may not be redacted effectively.
  - **Performance on Large Files**: Redacting very large files can be slow due to the computational overhead of NLP models.
  - **Incomplete Redaction**: Some complex patterns may not be fully redacted if they don't match the predefined patterns or entities.
  - **Encoding Issues**: Files with non-UTF-8 encoding may cause errors.

- **Assumptions**: Input files are plain text in English. Note that the tool does not preserve whitespace when censoring text to neatly remove all details of sensitive information, which may affect the text structure.

## Acknowledgements

- **SpaCy**: For NLP models and tools.
- **NLTK**: For WordNet lexical database.
- **Hugging Face**: For the BERT-based NER model.
