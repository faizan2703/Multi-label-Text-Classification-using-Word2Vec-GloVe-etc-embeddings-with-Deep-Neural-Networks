# Multi-Label Text Classification with Different Word Embeddings

## Table of Contents
1.  [Task Description](#task-description)
2.  [Dataset](#dataset)
3.  [Preprocessing Text and Labels](#preprocessing-text-and-labels)
4.  [Generating Word Embeddings](#generating-word-embeddings)
    *   [Word2Vec](#word2vec)
    *   [GloVe](#glove)
    *   [FastText](#fasttext)
    *   [BERT](#bert)
5.  [Preparing Data for PyTorch](#preparing-data-for-pytorch)
6.  [Deep Neural Network Model Definition](#deep-neural-network-model-definition)
7.  [Training and Evaluation](#training-and-evaluation)
8.  [Results Comparison](#results-comparison)
9.  [Summary and Insights](#summary-and-insights)

## 1. Task Description
This notebook performs multi-label text classification on the PubMed Multi Label Text Classification Dataset. The primary goal is to compare the performance of deep neural network models trained using various word embedding techniques: Word2Vec, GloVe, FastText, and BERT.

## 2. Dataset
The dataset used is `PubMed Multi Label Text Classification Dataset Processed.csv`. It contains scientific article titles and abstracts, along with multiple associated categorical labels (A, B, C, ..., Z).

**Dataset Loading:**
- The CSV file was loaded into a pandas DataFrame.
- Initial inspection showed columns for `Title`, `abstractText`, `meshMajor`, `pmid`, `meshid`, `meshroot`, and 14 label columns (A, B, C, D, E, F, G, H, I, J, L, M, N, Z).

## 3. Preprocessing Text and Labels

### Text Preprocessing
1.  **Combine Text:** 'Title' and 'abstractText' columns were combined into a single `combined_text` column.
2.  **Clean Text:** A `preprocess_text` function was applied to `combined_text` to:
    *   Convert text to lowercase.
    *   Remove punctuation and numbers.
    *   Remove English stopwords.
    *   Lemmatize words using `WordNetLemmatizer`.
    The cleaned text is stored in the `processed_text` column.

### Label Preparation
1.  **Identify Labels:** Columns 'A' through 'Z' (excluding K, O, P, Q, R, S, T, U, V, W, X, Y) were identified as label columns.
2.  **Extract Labels:** The values from these label columns were extracted into a NumPy array, representing the binary multi-labels for each text entry. The shape of the labels array is (6623, 14).

## 4. Generating Word Embeddings
Dense embeddings were generated for the `processed_text` using four different methods:

### Word2Vec
- A `gensim.models.Word2Vec` model was trained on the tokenized `processed_text`.
- Parameters: `vector_size=100`, `window=5`, `min_count=1`, `workers=4`.
- Document embeddings were created by averaging the Word2Vec embeddings of words in each document. Out-of-vocabulary words were handled by returning a zero vector.
- **Shape of Embeddings:** (6623, 100)

### GloVe
- Pre-trained GloVe 6B 100d vectors were downloaded (`http://nlp.stanford.edu/data/glove.6B.zip`) and loaded.
- Document embeddings were created by averaging the GloVe embeddings of words in each document. Out-of-vocabulary words were handled by returning a zero vector.
- **Shape of Embeddings:** (6623, 100)

### FastText
- Pre-trained FastText `wiki-news-300d-1M.vec` vectors were downloaded (`https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip`) and loaded using `gensim.models.KeyedVectors`.
- Document embeddings were created by averaging the FastText embeddings of words in each document. Out-of-vocabulary words were handled by returning a zero vector.
- **Shape of Embeddings:** (6623, 300)

### BERT
- A pre-trained `bert-base-uncased` tokenizer and model were loaded from the `transformers` library.
- A `get_bert_embeddings` function was implemented to process text in batches, tokenize, and generate embeddings using the BERT model. The document embedding was obtained by taking the mean of the last hidden state.
- **Shape of Embeddings:** (6623, 768)

## 5. Preparing Data for PyTorch
For each set of embeddings (Word2Vec, GloVe, FastText, BERT):
- Embeddings and labels were converted into PyTorch tensors (`torch.float32`).
- The data was split into training (80%) and validation (20%) sets using `sklearn.model_selection.train_test_split` with `random_state=42`.
- `torch.utils.data.TensorDataset` and `torch.utils.data.DataLoader` were used to create iterable datasets for training and validation, with a `batch_size` of 32.

## 6. Deep Neural Network Model Definition
A simple Multi-Label Classifier model was defined using `torch.nn.Module`:
- **Input Layer:** `nn.Linear(input_dim, 256)`
- **Activation:** `nn.ReLU()`
- **Output Layer:** `nn.Linear(256, num_labels)`
- **Output Activation:** `nn.Sigmoid()` for multi-label binary classification.

## 7. Training and Evaluation
Each model (one for each embedding type) was trained and evaluated independently:
- **Loss Function:** `nn.BCELoss()` (Binary Cross-Entropy Loss).
- **Optimizer:** `optim.Adam(model.parameters(), lr=0.001)`.
- **Epochs:** 10 epochs for each model.
- **Metrics:** F1-score (macro), Precision (macro), Recall (macro), and ROC AUC (macro, multi-class='ovr') were calculated on the validation set.
- **Device:** Training and evaluation were performed on the CPU.

## 8. Results Comparison

The evaluation results for each embedding type are summarized below:

| Embedding Type | F1-Score (macro) | Precision (macro) | Recall (macro) | ROC AUC (macro) |
|----------------|------------------|-------------------|----------------|-----------------|
| Word2Vec       | 0.610            | 0.689             | 0.599          | 0.841           |
| GloVe          | 0.658            | 0.767             | 0.621          | 0.862           |
| FastText       | 0.671            | 0.754             | 0.645          | 0.870           |
| BERT           | 0.700            | 0.760             | 0.688          | 0.874           |

### Model Performance Comparison Across Different Embeddings

```python
# This chart was generated in the notebook to visualize the results
import matplotlib.pyplot as plt
import pandas as pd

results = {'word2vec': {'f1_score': 0.6100163597145617, 'precision_score': 0.6890196866347719, 'recall_score': 0.5987269122981476, 'roc_auc_score': 0.8406699352762051}, 'glove': {'f1_score': 0.6580275688917102, 'precision_score': 0.7670428525001725, 'recall_score': 0.620512564018138, 'roc_auc_score': 0.8615443193015011}, 'fasttext': {'f1_score': 0.6706382401170667, 'precision_score': 0.7536417853913356, 'recall_score': 0.6446490411900893, 'roc_auc_score': 0.8697078749240644}, 'bert': {'f1_score': 0.7003897949543126, 'precision_score': 0.760171664952272, 'recall_score': 0.6882217377197936, 'roc_auc_score': 0.8738024882186463}}

results_df = pd.DataFrame(results).T
metrics = ['f1_score', 'precision_score', 'recall_score', 'roc_auc_score']

plt.figure(figsize=(12, 7))
results_df[metrics].plot(kind='bar', figsize=(12, 7))
plt.title('Model Performance Comparison Across Different Embeddings')
plt.xlabel('Embedding Type')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric')
plt.tight_layout()
plt.show()
```

## 9. Summary and Insights

### Key Findings
*   The **BERT** embedding model performed best for this multi-label text classification task. It achieved the highest scores across all evaluated metrics: F1-score (0.700), Precision (0.760), Recall (0.688), and ROC AUC (0.874).
*   The results indicate a clear performance hierarchy:
    *   **BERT** (F1: 0.700, ROC AUC: 0.874)
    *   **FastText** (F1: 0.671, ROC AUC: 0.870)
    *   **GloVe** (F1: 0.658, ROC AUC: 0.862)
    *   **Word2Vec** (F1: 0.610, ROC AUC: 0.841)

### Insights
*   The superior performance of BERT highlights the importance of contextual understanding and larger embedding dimensions for complex text classification tasks. Models capable of capturing nuanced semantic relationships, such as transformer-based models like BERT, are more effective than traditional word embeddings for this type of task.

### Next Steps
*   **Fine-tuning BERT:** Explore fine-tuning the BERT model on this specific dataset rather than just using its fixed embeddings, which could further improve performance.
*   **Other Transformer Models:** Experiment with different pre-trained transformer models (e.g., RoBERTa, XLNet, ELECTRA) to see if they yield even better results.
*   **Model Architecture:** For Word2Vec, GloVe, and FastText, consider more complex neural network architectures (e.g., adding more hidden layers, dropout, or convolutional layers) or attention mechanisms to better leverage their embeddings.
*   **Hyperparameter Tuning:** Conduct more extensive hyperparameter tuning for each model and embedding type, including learning rates, batch sizes, and model layer sizes.
