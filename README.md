# Embedding-Based Prediction of Neurological Deterioration in Intracerebral Hemorrhage
Model development for ICH deterioration via NLP-based approach.

## 1. Pre-Processing Pipeline

Loading required dependencies and initializing SpaCy for de-anonymization:

```python
import re
import spacy
import pandas as pd
import unicodedata

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')
```

Loading helper functions for de-anonymization and pre-processing (removal of NRICs, datetimes, special symbols):

```python
def deanonymize(text):
    doc = nlp(text)
    # Extract named entities
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # Replace identified names with placeholders in the original text
    anonymized_text = text
    for name in names:
        anonymized_text = anonymized_text.replace(name, "[NAME]")
    return anonymized_text

def preprocess(text):
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Remove NRICs
    text = re.sub(r'\b[A-Za-z]\d{7}[A-Za-z]\b', '', text)
    # Remove dates (pattern: dd/mm/yy, dd-mm-yy, dd/mm/yyyy, dd-mm-yyyy, month year)
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
    text = re.sub(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\b', '', text, flags=re.IGNORECASE)
    # Remove times (pattern: hh:mm, hh:mm:ss, with am/pm)
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM|am|pm)?\b', '', text)
    text = re.sub(r'\b\d{1,4}(am|pm)\b', '', text, flags=re.IGNORECASE)
    # Remove special symbols (===, +, /)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocessing_pipeline(text):
    anonymized_text = deanonymize(text)
    return preprocess(anonymized_text)
```

Applying pre-processing pipeline on ED note input data, export to new training environment:

```python
# Load ED Notes dataset
df = pd.read_csv("ed_notes.csv")
# Inspect loaded dataframe
pd.set_option('display.max_columns', None); df

# Applying preprocessing pipeline
df['PROCESSED_TEXT'] = df['NOTE_TEXT'].apply(preprocessing_pipeline)

# Exporting results for loading in new environment
df.to_csv("processed.csv", index = False)
```

## 2. Embedding Generation

Loading required dependencies and pre-trained model for embedding:

```python
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch

# Function to print separator line
def print_separator(text):
    print(f"\n{'-'*10} {text} {'-'*10}")

# Load a pre-trained model
print_separator("Loading Model")
modelPath = r'C:\Users\jonathanymj\Desktop\SBert Experimentation\BioBert'
# modelPath = r'C:\Users\jonathanymj\Desktop\SBert Experimentation\all-MiniLM-L6-v2'
model = SentenceTransformer(modelPath)
print(f"Model loaded: {model}")
```

Loading helper functions for generating embedding. Sliding Window and Mean Pooling approach adopted to overcome model max sequence length barrier. Resultant embedding dimensionality check defined.

```python
# Helper Functions 
def sliding_window(text, model, max_length, overlap=50):
    # Splits text into overlapping chunks based on maximum sequence length and overlap
    # ensures each chunk is within the token limit
    tokens = model.tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(model.tokenizer.convert_tokens_to_string(chunk))
    return chunks

def generate_embeddings(text, model, max_length, overlap=20, pooling_strategy='mean'):
    chunks = sliding_window(text, model, max_length, overlap)
    embeddings = [model.encode(chunk) for chunk in chunks]
    if pooling_strategy == 'mean':
        final_embedding = np.mean(embeddings, axis=0)
    elif pooling_strategy == 'max':
        final_embedding = np.max(embeddings, axis=0)
    else:
        raise ValueError("Unsupported pooling strategy. Choose 'mean' or 'max'.")
    return final_embedding

def check_embedding_dimensionality(df):
    # Check the shape of the first embedding
    first_embedding = df['EMBEDDINGS'].iloc[0]
    print(f"Shape of a single embedding: {first_embedding.shape}")
    
    # Check if all embeddings have the same shape
    shapes = df['EMBEDDINGS'].apply(lambda x: x.shape)
    if shapes.nunique() == 1:
        print("All embeddings have the same shape.")
    else:
        print("Warning: Embeddings have different shapes.")
        print(shapes.value_counts())
    
    return first_embedding.shape
```

Loading pre-processed notes and generating embeddings:

```python
# Load preprocessed dataset
df = pd.read_csv("processed.csv")
# Inspect loaded dataframe
pd.set_option('display.max_columns', None); df

# Generating embeddings
df['EMBEDDINGS'] = df['PROCESSED_TEXT'].apply(lambda x: generate_embeddings(x, model, model.max_seq_length))

# Check embedding dimensionality
embedding_shape = check_embedding_dimensionality(df)
print(f"Embedding dimensionality: {embedding_shape[0]}")
```

Creating dataframe for embedding matrix and target vector:

```python
# Input Embedding Array (convert to dataframe)
X = np.stack(df['EMBEDDINGS'].to_numpy())
X = pd.DataFrame(X)

# Target Column
y = df['Deterioration_Case']

# Combining Embedding + Target
clf_data = pd.concat([X,y], axis = 1)

# Exporting results for loading in new environment
clf_data.to_csv("embeddings.csv", index = False)
```

## 3. Classification Model

Loading embedding dataset:

```python
# Loading required packages
import pandas as pd
import numpy as np

# Load preprocessed dataset
df = pd.read_csv("embeddings.csv")
# Inspect loaded dataframe
pd.set_option('display.max_columns', None)
df
```

Loading pycaret classification tooling for ml training orchestration:

```python
# Import pycaret classification and init setup
from pycaret.classification import *
clf_setup = setup(
    data=df,
    target='Deterioration_Case',
    # Random Seed for Reproducibility
    session_id=123
)

# Train models
best = compare_models(sort = 'F1', exclude ='catboost')

# Predict on hold-out
predict_model(best)

# Inspect evaluation metrics
evaluate_model(best)
```
