# The Redactor

**Author**: Sakshi Pandey  
**Email**: [sakshi.pandey@ufl.edu](mailto:sakshi.pandey@ufl.edu)

## Introduction

The Redactor is a sophisticated text redaction tool designed to censor sensitive information from plain text documents. It efficiently redacts names, dates, phone numbers, addresses, and text related to specific concepts. Leveraging advanced Natural Language Processing (NLP) models and techniques, including SpaCy, NLTK, and Hugging Face's BERT-based models, The Redactor ensures thorough and accurate redaction while maintaining the integrity of the original document structure.

## Installation

Follow these steps to set up The Redactor on your local machine:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SakshiPandey97/cis6930fa24-project1.git
   cd cis6930fa24-project1
   ```

2. **Create a Virtual Environment Using Pipenv**:

   ```bash
   pipenv install
   pipenv shell
   ```

3. **Download Necessary NLTK Data if not installed automatically.**:

   Launch a Python interpreter and run:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

4. **Ensure SpaCy Models Are Installed**:

   The `Pipfile` includes SpaCy models as dependencies. They should install automatically with `pipenv install`. If not, manually install them:

   ```bash
   python -m spacy download en_core_web_trf
   python -m spacy download en_core_web_sm
   ```

## Usage

Run The Redactor script via the command line with appropriate flags:

```bash
pipenv run python redactor.py --input [INPUT_FILES] --output [OUTPUT_DIRECTORY] [FLAGS]
```

**Example**:

```bash
pipenv run python redactor.py --input 'docs/*.txt' \
    --names --dates --phones --address \
    --concept 'bank' \
    --output 'redacted_output' \
    --stats stderr
```

This command processes all `.txt` files in the current directory, redacts specified sensitive information, saves the redacted files to the `redacted_output` directory, and outputs redaction statistics to `stderr`.

## Flags and Parameters

- **`--input`**: Glob pattern(s) for input files to redact. Supports multiple patterns.
- **`--output`**: Directory to store output files. Created if it doesn't exist.
- **`--names`**: Censor personal names using SpaCy's Named Entity Recognition (NER) and custom regex patterns.
- **`--dates`**: Censor dates using SpaCy's NER.
- **`--phones`**: Censor phone numbers using SpaCy's `Matcher` and regex patterns.
- **`--address`**: Censor addresses using SpaCy's NER.
- **`--concept`**: Redact text related to specific concepts. Accepts one or more concepts, expanding synonyms using NLTK WordNet and similarity using SpaCy.
- **`--stats`**: Output redaction statistics. Options include `stderr`, `stdout`, or a file path.

## Design Decisions

The Redactor employs a hybrid approach combining multiple techniques to ensure comprehensive redaction:

- **SpaCy Named Entity Recognition (NER)**: Identifies standard named entities like names, dates, and locations.
- **Hugging Face BERT-based NER**: Enhances accuracy for certain entities not captured by SpaCy alone.
- **Regular Expressions (Regex)**: Targets specific patterns such as emails and phone numbers that are challenging for machine learning models.
- **NLTK WordNet**: Expands concept-based redaction by extracting synonyms and related terms.
- **SpaCy Matcher**: Detects complex patterns, particularly for phone numbers and other formatted data.

This combination mitigates the limitations of individual methods, ensuring thorough and accurate redaction across diverse document structures.

## Models and Libraries Used

- **NLTK WordNet**: Utilized for extracting synonyms and related forms of specified concepts, enhancing the tool's ability to redact concept-related text comprehensively.
- **Regex**: Employed to match and redact specific patterns like email addresses and phone numbers.
- **SpaCy**:
  - `en_core_web_trf`: A transformer-based model used for advanced NLP tasks, including accurate entity recognition.
  - `en_core_web_sm`: A smaller, efficient model used for pattern matching and quicker processing where high accuracy is less critical.
- **Hugging Face Transformers (BERT)**: Specifically, the `dslim/bert-base-NER` model is used to supplement SpaCy's NER capabilities, capturing entities that SpaCy might miss.
- **Transformers Pipeline**: Facilitates seamless integration of Hugging Face models into the redaction workflow.
- **Pipenv**: Manages project dependencies and virtual environments.

## Redaction Process

The Redactor processes each input file by sequentially applying redaction functions based on the specified flags. The process ensures that overlapping or nested sensitive information is handled correctly, and redaction statistics are meticulously recorded.

### Selective Redaction

For specific entities like names, addresses, dates, and phone numbers, The Redactor replaces the entire sensitive text with block characters (`█`). 

**Examples**:

#### Names

- **Original**:

  ```
  Hello, my name is Jack O'Lantern. Welcome to the haunted house.
  ```

- **Redacted**:

  ```
  Hello, my name is ██████████████. Welcome to the haunted house.
  ```

#### Addresses

- **Original**:

  ```
  The meeting is at 123 Elm Street, Springfield.
  ```

- **Redacted**:

  ```
  The meeting is at ████████████, ██████████.
  ```

#### Dates

- **Original**:

  ```
  The event is scheduled for October 31, 2023.
  ```

- **Redacted**:

  ```
  The event is scheduled for ████████████████.
  ```

#### Phone Numbers

- **Original**:

  ```
  Contact us at 666-123-4567 or (666) 765-4321.
  ```

- **Redacted**:

  ```
  Contact us at ████████████ or ██████████████.
  ```

### Comprehensive Redaction (Concepts)

When redacting based on specific concepts, The Redactor removes entire sentences containing the specified concepts, including all characters, whitespace, and punctuation. This ensures complete removal of contextual information related to the concept.

**Example**:

- **Concept**: `bank`

- **Original**:

  ```
  The bank is located downtown. Customers rely on financial institutions.
  ```

- **Redacted**:

  ```
  ███████████████████████████████████████████. ████████████████████████████████████.
  ```

## Benefits of this Redaction Method

- **Maintains Document Layout**: Matching block characters to the original text length and spacing preserves the document's structure and alignment.

## Statistics Tallying

The Redactor meticulously tracks and tallies redaction statistics to provide insights into the redaction process. Statistics include the number and details of each type of redacted information per file.

### How Statistics Are Tallied

1. **Data Structure**: A dictionary named `stats` is used to store redaction details categorized by entity types (ex. `NAMES`, `PHONE_NUMBERS`, `ADDRESSES`, `DATES`, `CONCEPTS`, `CONCEPT_WORDS`).

2. **Recording Redactions**:
   - For each redacted entity, an entry is added to the corresponding category in `stats`.
   - Each entry includes:
     - **`text`**: The exact text that was redacted.
     - **`start`**: The starting character index of the redacted text in the original document.
     - **`end`**: The ending character index of the redacted text.

3. **Outputting Statistics**:
   - After processing each file, the `output_stats` function formats the `stats` dictionary into a readable string.
   - Depending on the `--stats` flag, statistics are outputted to:
     - `stderr`: Standard error stream.
     - `stdout`: Standard output stream.
     - A specified file: Appended to the provided file path.

4. **Example of Statistics Output**:

   ```
   File: example.txt
   NAMES: 3
     Text: Jack Rodriguez Start: 15 End: 23
     Text: Mason Bertolli Start: 45 End: 55
   PHONE_NUMBERS: 2
     Text: (123) 456-7890 Start: 200 End: 214
     Text: 987-654-3210 Start: 300 End: 313
   ADDRESSES: 1
     Text: Elm Street Start: 400 End: 414
   DATES: 1
     Text: January 1, 2024 Start: 500 End: 515
   CONCEPTS: 2
     Text: The bank is located downtown. Start: 600 End: 625
     Text: Many customers have trust funds there. Start: 700 End: 740
   CONCEPT_WORDS: 3
     Text: bank Start: 605 End: 609
     Text: trust Start: 725 End: 736
   ```

This detailed reporting allows users to understand what was redacted and where, facilitating transparency and accountability in the redaction process.

## Tests

The `tests/` directory contains comprehensive unit tests for each redaction function, ensuring reliability and accuracy.

### Running the Tests

Execute the following command to run all tests:

```bash
pipenv run python -m pytest
```

### Test Descriptions

- **`test_names.py`**: Validates the redaction of personal names, ensuring both SpaCy and regex-based detections are effective.
- **`test_phones.py`**: Ensures phone numbers in various formats are correctly identified and redacted.
- **`test_address.py`**: Tests the redaction of addresses, verifying the coverage of different address formats.
- **`test_concepts.py`**: Checks the accurate redaction of concept-related text, including synonym and similar word handling.
- **`test_dates.py`**: Confirms that dates in multiple formats are properly detected and redacted.

## Bugs and Assumptions

### Bugs

- **Address Redaction**: 
  - **Issue**: The address redaction functionality may miss certain addresses, especially those not adhering to standard formats.

- **Phone Number Coverage**:
  - **Issue**: Currently, the focus of this project was for US and UK phone number formats to be effectively redacted. Phone numbers in other international formats may remain unredacted.

- **Incomplete Redaction**:
  - **Issue**: Complex patterns or unconventional formats may not be fully redacted if they don't match predefined patterns or recognized entities. For example an email address like this: susan@bestbank.com containing the word bank may not be redacted given the concept bank.

### Assumptions

- **Language**: Input files are plain text in English.
- **File Format**: Input files are text-based.
- **Entity Representation**: Relies on the accuracy of underlying NLP models and regex patterns; assumes that most sensitive information adheres to standard representations.

