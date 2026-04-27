# 🚀 FinGPT: Quantitative Trading Signal Terminal

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![Milvus](https://img.shields.io/badge/VectorDB-Milvus-blueviolet)
![GCP Cloud Run](https://img.shields.io/badge/Deployed-GCP_Cloud_Run-green)

FinGPT is an end-to-end financial quantitative trading signal extraction engine. By integrating a **LoRA fine-tuned DeepSeek-R1** model with a **Milvus-powered Retrieval-Augmented Generation (RAG)** architecture, it intelligently parses vast amounts of financial news to output strict, machine-readable JSON trading signals. The project includes a local Gradio terminal and a production-ready deployment scheme for Google Cloud Run.

## ✨ Core Features

* 🧠 **LLM Reasoning Engine:** Powered by a fine-tuned `DeepSeek-R1-Distill-Llama-8B`, delivering robust financial logic reasoning via Chain-of-Thought (CoT) outputs.
* 📚 **RAG-Enhanced Knowledge Base:** Utilizes `Milvus-Lite` to store 2200+ professional financial audit rules and market sentiment logics, significantly reducing model hallucinations.
* 🧽 **Information Dehydration:** Automatically filters out PR fluff and ESG statements to extract only core financial, operational, and strategic facts.
* ⚡ **Production API + UI:** High-performance `FastAPI` backend with a `Gradio` terminal mounted at the root path for seamless human-AI interaction.
* ☁️ **Cloud Native Deployment:** Optimized `Dockerfile` with pre-warmed embedding models, designed for Google Cloud Run's `Gen2` execution environment.

## 🏗️ System Architecture

```text
[Long Financial News Input] 
           ↓ 
(Automated Fact Dehydration) -> Extracts hard strategic facts 
           ↓ 
[Sentence-Transformers] -> High-dimensional Vectorization
           ↓ 
[Milvus Vector DB] -> Semantic retrieval of audit rules (RAG)
           ↓ 
[DeepSeek-R1 + Structured Prompt] -> Generates Internal CoT
           ↓
[Signal Parser] -> Standard JSON Output {"direction": 1, "confidence": 95}
