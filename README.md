# Sentiment Analysis on IMDB Movie Reviews

[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com)

## Abstract

This research project presents a comprehensive comparative analysis of sentiment classification models on the IMDB movie reviews dataset. We evaluate four distinct approaches—TF-IDF with Logistic Regression, Word2Vec embeddings, BERT-based feature extraction, and fine-tuned BERT—to understand the trade-offs between model complexity, computational efficiency, and classification performance. Our findings demonstrate that while fine-tuned BERT achieves superior accuracy, TF-IDF provides a compelling baseline with significantly lower computational overhead.

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Citation](#citation)
- [Author](#author)

## Overview

**Objective**: Compare sentiment classification models to identify optimal approaches for different computational constraints.

**Dataset**: IMDB Reviews (50,000 samples)
- Training: 20,000 samples
- Validation: 5,000 samples  
- Testing: 25,000 samples

**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## Methodology

Our experimental approach follows standard machine learning practices:

1. **Data Preparation**
   - Load IMDB dataset from Hugging Face
   - Stratified train/validation/test split
   - Consistent preprocessing across all models

2. **Preprocessing Pipeline**
   - Lowercase conversion
   - HTML tag removal
   - Punctuation removal
   - Tokenization (for embedding-based models)

3. **Model Development**
   - Train four independent models
   - Consistent hyperparameters and optimization
   - GPU acceleration where applicable

4. **Evaluation**
   - Standardized metrics across all models
   - Error pattern analysis
   - Computational resource tracking

## Models

### 1. TF-IDF + Logistic Regression

**Description**: Traditional bag-of-words approach with TF-IDF vectorization.

**Specifications**:
- Max features: 20,000
- Classifier: Logistic Regression (max_iter=1000)
- Training time: 2-5 minutes
- Hardware: CPU only

**Advantages**:
- Fast training and inference
- Minimal memory requirements
- Interpretable feature importance
- Effective baseline

### 2. Word2Vec + Logistic Regression

**Description**: Word embedding approach with averaged vectors.

**Specifications**:
- Vector size: 100 dimensions
- Window size: 5
- Min count: 2
- Classifier: Logistic Regression (max_iter=1000)
- Training time: 5-10 minutes
- Hardware: CPU only

**Advantages**:
- Captures semantic relationships
- Moderate computational cost
- Pre-computed embeddings

**Limitations**:
- Averaging removes word order information
- Weak at detecting negation

### 3. BERT Embeddings + Logistic Regression

**Description**: Contextual embeddings from pre-trained BERT as features.

**Specifications**:
- Model: bert-base-uncased
- Extraction: CLS token embedding
- Batch size: 16
- Max length: 256 tokens
- Classifier: Logistic Regression (max_iter=1000)
- Training time: 15-30 minutes (with GPU)
- Hardware: GPU recommended

**Advantages**:
- Contextual word representations
- Better than Word2Vec performance
- Fast fine-tuning possible
- Pre-trained knowledge transfer

**Limitations**:
- Fixed representations
- Slower than TF-IDF/Word2Vec
- Higher memory requirements

### 4. Fine-tuned BERT

**Description**: End-to-end fine-tuning for sentiment classification.

**Specifications**:
- Model: BertForSequenceClassification
- Labels: 2 (positive/negative)
- Optimizer: AdamW
- Learning rate: 2e-5
- Epochs: 2
- Batch size: 16
- Training time: 30-60 minutes (with GPU)
- Hardware: GPU required (6GB+ VRAM)

**Advantages**:
- Highest performance
- Task-specific optimization
- Contextual fine-tuning
- State-of-the-art results

**Limitations**:
- High computational cost
- Longest training time
- Requires GPU
- Risk of overfitting on small datasets

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---|
| TF-IDF | High | High | High | High | 2-5 min |
| Word2Vec | Lower | Lower | Lower | Lower | 5-10 min |
| BERT Embeddings | Higher | Higher | Higher | Higher | 15-30 min |
| **Fine-tuned BERT** | **Highest** | **Highest** | **Highest** | **Highest** | 30-60 min |

### Computational Efficiency

| Metric | TF-IDF | Word2Vec | BERT Embeddings | Fine-tuned BERT |
|--------|--------|----------|-----------------|-----------------|
| Training Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Inference Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Memory Usage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| CPU Only | ✅ | ✅ | ❌ | ❌ |
| GPU Required | ❌ | ❌ | ⚠️ | ✅ |

## Key Findings

### 1. Fine-tuned BERT Achieves Best Performance
- Highest accuracy and F1-score through task-specific optimization
- End-to-end learning enables better contextual understanding
- Significant improvement over fixed embeddings

### 2. TF-IDF Provides Strong Baseline
- Surprisingly effective for IMDB dataset
- Keyword-based features are crucial for sentiment classification
- Massive computational advantage over deep learning approaches
- Recommended for resource-constrained environments

### 3. Word2Vec Limitations Evident
- Averaging embeddings eliminates word order
- Weak at detecting negation and sarcasm
- Outperformed by both TF-IDF and BERT approaches
- Limited practical value for this task

### 4. Trade-offs Clear
- Performance gains come at computational cost
- Fine-tuned BERT: Best accuracy but requires GPU
- TF-IDF: Practical choice for production systems
- Word2Vec: Middle ground but not recommended

### Error Pattern Analysis

**Common Challenges Across All Models**:
- Sarcasm and irony detection
- Mixed sentiment reviews (both positive and negative)
- Domain-specific language and slang

**Model-Specific Issues**:
- TF-IDF: Context-agnostic, struggles with subtle sentiment
- Word2Vec: Weak negation handling, word order loss
- BERT Embeddings: Fixed context, limited adaptation
- Fine-tuned BERT: Can overfit on small datasets

**Performance by Review Type**:
- Short reviews: All models perform well
- Long reviews: BERT-based models superior (contextual advantage)
- Complex sentiment: Fine-tuned BERT significantly better

## Installation

### System Requirements
- Python 3.7 or higher
- CUDA 11.8+ (for GPU acceleration)
- Minimum 8GB RAM (16GB recommended)
- GPU with 4GB+ VRAM (for BERT models)

### Dependencies

Clone and install:
```bash
git clone https://github.com/yourusername/sentiment-analysis-imdb.git
cd "Sentiment Analysis on IMDB Movie Reviews"
pip install -r requirements.txt
```

Or install manually:
```bash
pip install datasets==2.13.0
pip install scikit-learn==1.3.0
pip install nltk==3.8.1
pip install gensim==4.3.0
pip install transformers==4.30.0
pip install torch==2.0.0
pip install pandas==2.0.0
pip install numpy==1.24.0
```

## Usage

### Running the Notebook

1. **Google Colab** (Recommended):
   ```
   Upload the notebook to Google Colab
   Runtime → Change runtime type → GPU
   Run cells sequentially
   ```

2. **Local Jupyter**:
   ```bash
   jupyter notebook Sentiment_Analysis_on_IMDB_Movie_Reviews.ipynb
   ```

3. **Execution Steps**:
   - Cell 1-3: Initialize and set seeds
   - Cell 4-9: Load and preprocess data
   - Cell 10-12: Train TF-IDF model
   - Cell 13-20: Train Word2Vec model
   - Cell 21-28: Extract BERT embeddings
   - Cell 29-37: Evaluate all three models
   - Cell 38-52: Fine-tune BERT
   - Cell 53-56: Final comparison and analysis

### Expected Runtime
- TF-IDF: 2-5 minutes
- Word2Vec: 5-10 minutes
- BERT Embeddings: 15-30 minutes (GPU)
- Fine-tuned BERT: 30-60 minutes (GPU)

## Dataset

### IMDB Reviews Dataset

- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/imdb)
- **Size**: 50,000 reviews
- **Labels**: Binary (0=negative, 1=positive)
- **Split**:
  - Train: 20,000 (80% of original train)
  - Validation: 5,000 (20% of original train)
  - Test: 25,000 (original test set)

### Preprocessing

Text cleaning pipeline:
```python
def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
```

Applied to all text samples for consistency.

## Project Structure

```
Sentiment_Analysis_on_IMDB_Movie_Reviews.ipynb
├── 01: Setup & Initialization (Cells 1-3)
├── 02: Data Loading & Preprocessing (Cells 4-9)
├── 03: TF-IDF Model (Cells 10-12)
├── 04: Word2Vec Model (Cells 13-20)
├── 05: BERT Embeddings (Cells 21-28)
├── 06: Model Evaluation (Cells 29-37)
├── 07: Fine-tuned BERT (Cells 38-52)
└── 08: Results & Analysis (Cells 53-56)
```

## Hyperparameters

### TF-IDF
```python
TfidfVectorizer(max_features=20000)
LogisticRegression(max_iter=1000)
```

### Word2Vec
```python
Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2)
LogisticRegression(max_iter=1000)
```

### BERT Embeddings
```python
BertTokenizer.from_pretrained("bert-base-uncased")
BertModel.from_pretrained("bert-base-uncased")
batch_size = 16
max_length = 256
```

### Fine-tuned BERT
```python
BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 2
batch_size = 16
```

## Reproducibility

Fixed random seeds for consistent results:
```python
random.seed(42)
np.random.seed(42)
```

All splits use `seed=42` for reproducibility.

## References

- [Hugging Face Datasets](https://huggingface.co/datasets/imdb)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Gensim Word2Vec](https://radimrehal.com/gensim_models/word2vec.html)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK Documentation](https://www.nltk.org/)
- [PyTorch](https://pytorch.org/)

## Citation

If you use this project in your research, please cite:

```bibtex
@project{sentiment_analysis_imdb_2026,
  title={Comparative Analysis of Sentiment Classification Models on IMDB Reviews},
  author={Shuvo, Md. Mehedi Hasan},
  year={2026},
  institution={American International University-Bangladesh},
  type={Research Project}
}
```

## License

This project is provided for educational and research purposes.

---

## Author

**Md. Mehedi Hasan Shuvo**
- **Role**: Aspiring AI Engineer & Researcher
- **Institution**: American International University–Bangladesh (AIUB)
- **Major**: Computer Science and Engineering (Information Systems)
- **Focus**: Artificial Intelligence, Machine Learning, Deep Learning, Data Science, and Computer Vision
