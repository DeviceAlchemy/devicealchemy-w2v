# DeviceAlchemy Word2Vec Model
Word2Vec Mini Model - Training &amp; Inference


This model is a **sample** of one of the many building block models used to enable the materials stack prediction engine at [DeviceAlchemy.ai](https://www.devicealchemy.ai).

## Overview

This repository contains a domain-specific Word2Vec model trained on openly licensed (CC BY) scientific text. The model captures semantic relationships between materials, elements, physical phenomena, and device concepts.


## Model Details

| Property              | Value                          |
|-----------------------|--------------------------------|
| Model Type            | Word2Vec (Skip-gram)           |
| Vector Dimension      | 200                            |
| Training Corpus       | ~120 million words             |
| Domain                | Materials Science, Condensed Matter Physics, Spintronics, Electronic Devices |
| Framework             | Gensim                         |
| Files Included        | Full model + KeyedVectors      |

## Files in This Repository

| File                                      | Description                              |
|-------------------------------------------|------------------------------------------|
| `devicealchemy_w2v.model`                 | Full Word2Vec model (recommended)        |
| `devicealchemy_w2v.model.wv.vectors.npy`  | Word vectors (part of full model)        |
| `devicealchemy_w2v.model.syn1neg.npy`     | Output weights (part of full model)      |
| `devicealchemy_w2v.kv`                    | KeyedVectors only (lighter)              |
| `devicealchemy_bigrams.pkl`               | Bigram phrase model                      |
| `devicealchemy_trigrams.pkl`              | Trigram phrase model                     |

## Usage

### Load the Full Model (Recommended)

```python
from gensim.models import Word2Vec

model = Word2Vec.load("models/devicealchemy_w2v.model")

# Get vector for a material
vector = model.wv["CoFeB"]

# Find similar materials
similar = model.wv.most_similar("spin_orbit_torque", topn=10)
print(similar)
