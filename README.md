# Information-Retrieval-Hierarchical-Bloom-filter-tree-index

This repository contains an implementation in Python of the full-text retrieval algorithm described in the article:

> **"A Privacy-Preserved Full-Text Retrieval Algorithm Over Encrypted Data"**
> *Wei Song, Zhiguang Qin, Jin Li, Kui Ren, and Rongxing Lu*
> Published in: Future Generation Computer Systems (2016)

## 📖 Overview

This work focuses on enabling full-text search over encrypted documents without compromising user privacy. The implemented algorithm supports efficient keyword-based document retrieval using encryption and secure indexing, ensuring that the server cannot learn either the content of the documents or the search queries.

## 🧠 Features

* Secure index generation
* Encrypted keyword search
* Privacy preservation of both query and document content
* Full-text search functionality

## 🛡️ Security Model

The system assumes a **semi-honest server** that follows protocol but tries to infer private information. The client encrypts documents and indexes before uploading them. Queries are also encrypted using homomorphic techniques to preserve privacy during search.

## 📚 Reference

If you use this code or base your work on it, please cite the original paper:

```bibtex
@article{song2016privacy,
  title={A privacy-preserved full-text retrieval algorithm over encrypted data},
  author={Song, Wei and Qin, Zhiguang and Li, Jin and Ren, Kui and Lu, Rongxing},
  journal={Future Generation Computer Systems},
  volume={62},
  pages={148--159},
  year={2016},
  publisher={Elsevier}
}
