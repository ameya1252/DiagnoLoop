# DiagnoLoop: A Hybrid LLM and Knowledge-Based System for Iterative Medical Diagnosis

**Authors** 

**Ameya Deshmukh**, Anish Parkhe, Bhushan Shankar Halasagi, Vedant Jayesh Oza, Akash Dudhane,  
Ambarish Kshirsagar 

University of Southern California  

---

## 🩺 Overview
**DiagnoLoop** is an AI-powered clinical assistant that combines **large language models (LLMs)**, **retrieval-augmented generation (RAG)**, and **knowledge graphs** to predict medical diagnoses from patient-reported symptoms.  

Our hybrid pipeline:
1. Accepts **voice or text symptom input**.
2. Transcribes audio using **OpenAI Whisper**.
3. Extracts clinical entities with **AWS Comprehend Medical**.
4. Retrieves relevant context via **FAISS** over a StatPearls-based knowledge base.
5. Generates ranked diagnoses using a **fine-tuned GPT-4.1-mini** model.
6. Refines results through an **Infermedica knowledge graph** and iterative **LLM-driven Q&A loops**.

<img width="715" height="421" alt="Screenshot 2025-09-16 at 6 08 15 PM" src="https://github.com/user-attachments/assets/3b49eaac-9e41-45a2-8ddf-75ac532330ea" />

---

## 🔑 Key Features
- **Hybrid AI System**: Combines symbolic reasoning with generative LLMs.  
- **Retrieval-Augmented Generation (RAG)**: Improves factual grounding via FAISS-based search.  
- **Iterative Diagnosis Refinement**: Asks targeted follow-up questions to boost accuracy.  
- **Fine-Tuned LLMs**: Achieved **77.6% accuracy** on a curated symptom–disease dataset.  
- **Explainability**: Provides interpretable diagnostic reasoning for clinicians.  

---

## 📊 Results Snapshot
| Model                     | Accuracy (%) |
|----------------------------|--------------|
| BioLinkBERT (baseline)      | 73           |
| GPT-4.1-mini (fine-tuned)    | 77           |
| + RAG                       | 80           |
| + RAG + Knowledge Graph      | 82           |

**Follow-up Question Loop**: Improved diagnosis accuracy from **61% → 78%** with iterative Q&A.  

---

## 🛠️ Tech Stack
- **LLMs**: GPT-4.1-mini, Mistral-7B, LLaMA-2  
- **ASR**: OpenAI Whisper  
- **NLP**: AWS Comprehend Medical  
- **Retrieval**: FAISS, Sentence Transformers  
- **Knowledge Graph**: Infermedica API  
- **Frameworks**: PyTorch, Hugging Face Transformers  

---

## 🚀 Usage (Basic Setup)
```bash
# Clone the repository
git clone https://github.com/ameya1252/healthAgent.git
cd healthAgent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run main application
python app.py
