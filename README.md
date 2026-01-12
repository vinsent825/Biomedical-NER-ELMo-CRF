
# Biomedical NER using ELMo, Char-CNN, and CRF

This repository contains a high-precision **Named Entity Recognition (NER)** system specifically designed for **vertical domains** (Biomedical/Healthcare). Unlike standard LLM-based solutions, this model is built for **on-premise deployment**, ensuring data privacy while maintaining robustness against specialized technical jargon.

##  The Challenge: The "Jargon" Problem

In specialized fields like biomedicine, traditional NER often fails due to:

* **Domain-Specific Jargon**: Terms like `Ivacaftor` or `ΔF508` don't exist in standard dictionaries.
* **Privacy Constraints**: Sensitive IP and patient data cannot be sent to external LLM APIs (e.g., GPT-4).
* **Tokenization Fragmentation**: Standard tokenizers often break technical terms into meaningless sub-units.

## ️ Technical Solution: The ELMo Advantage

This model revitalizes the "small-but-mighty" **ELMo (Embeddings from Language Models)** architecture to provide a 100% offline solution:

1. **Sub-word Resilience (Char-CNN)**: Uses a Character-level CNN to learn the "shape" of words (prefixes/suffixes). It identifies new jargon by morphological patterns rather than fixed vocabularies.
2. **Deep Contextualization (BiLM)**: A Bidirectional Language Model captures semantic flow, allowing the model to distinguish between common words and technical parameters based on surrounding context.
3. **Viterbi Decoding (CRF)**: A Conditional Random Field layer ensures global label consistency, preventing illegal transitions (e.g., an `I-TAG` following an `O` tag).

## ️ Model Architecture

```text
Input Sequence -> Char-CNN Embedding -> Highway Layers -> Bi-LSTM (Forward & Backward) -> ScalarMix -> Task-Specific Bi-LSTM -> CRF Layer -> Predicted Tags

```

##  Requirements

* Python 3.8+
* PyTorch
* NLTK
* NumPy

Install dependencies:

```bash
pip install torch nltk numpy

```

## ️ Usage

The script includes a full pipeline: vocabulary building, ELMo pre-training, and NER fine-tuning.

```bash
python biomedical_named_entity_recognition_using_elmo_bilm_charcnn_v3.py

```

### Example Output

* **Input**: `Patient diagnosed with cystic fibrosis and treated with Ivacaftor.`
* **Output**: `Patient diagnosed with [cystic fibrosis/B-DISEASE] and treated with [Ivacaftor/B-DRUG].`

##  Key Features

* **100% Private**: Runs entirely on local hardware (CPU or GPU).
* **Lightweight**: Can be trained and deployed without massive server clusters.
* **Secure**: No external API calls, making it suitable for enterprise "know-how" protection.

