## ADDED Requirements

### Requirement: NLP Pipeline with spaCy
The system SHALL implement a complete NLP pipeline using spaCy including tokenization, lemmatization, part-of-speech tagging, and named entity recognition.

#### Scenario: Process financial text
- **WHEN** raw financial news or documents are provided
- **THEN** the system SHALL tokenize, lemmatize, identify named entities (companies, people, locations), and extract structured information

#### Scenario: Recognize financial entities
- **WHEN** processing earnings calls or news articles
- **THEN** the system SHALL identify company names, ticker symbols, financial metrics, and dates using NER

### Requirement: Document-Term Matrix
The system SHALL create bag-of-words and TF-IDF document-term matrices from text corpora.

#### Scenario: Build TF-IDF matrix
- **WHEN** a corpus of financial documents is provided
- **THEN** the system SHALL create a TF-IDF weighted document-term matrix with configurable vocabulary size and n-grams

#### Scenario: Apply text preprocessing
- **WHEN** building document-term matrices
- **THEN** the system SHALL remove stop words, apply stemming or lemmatization, and filter by term frequency

### Requirement: Naive Bayes Text Classification
The system SHALL implement Naive Bayes classifiers for categorizing financial news and documents.

#### Scenario: Classify news articles
- **WHEN** labeled financial news articles are provided
- **THEN** the system SHALL train a Naive Bayes classifier and predict categories for new articles with probability scores

#### Scenario: Evaluate classification performance
- **WHEN** a text classifier is trained
- **THEN** the system SHALL report precision, recall, F1-score, and confusion matrix on test data

### Requirement: Sentiment Analysis
The system SHALL perform sentiment analysis on financial text including news, social media, and earnings transcripts.

#### Scenario: Analyze news sentiment
- **WHEN** financial news text is provided
- **THEN** the system SHALL compute sentiment scores (positive, negative, neutral) using lexicon-based or ML-based methods

#### Scenario: Aggregate sentiment scores
- **WHEN** multiple documents for a security are analyzed
- **THEN** the system SHALL aggregate sentiment scores over time and compute time-series sentiment indicators
