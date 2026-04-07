# 🚀 Agentic Voice AI System

An intermediate-level project that builds a **real-time voice-based agentic system** capable of detecting and handling user interruptions intelligently using speech recognition and decision logic.

---

## 📌 Overview

This project demonstrates how modern AI systems can:
- Process audio input (offline + real-time simulation)
- Convert speech to text using Whisper
- Identify **filler vs meaningful speech**
- Trigger actions like stopping TTS based on user intent

The system is designed in a **modular + scalable way**, moving from basic classification to a near production-style pipeline.

---

## 🧠 Key Features

- 🎤 Audio input processing (real-time + file-based)
- 🧾 Speech-to-text using Whisper (Hugging Face)
- ⚡ Interrupt classification:
  - VALID
  - IGNORE (fillers/noise)
  - INTERRUPT
- 🛑 Smart interruption handling (stop TTS simulation)
- 🧩 Modular architecture (3 levels of complexity)
- 🔁 Clean pipeline design for scalability
- 
---

## ⚙️ Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers (Whisper)  
- NumPy  
- SciPy  
- SoundFile  

---

## 🔍 System Workflow

1. Audio input is captured or loaded  
2. Audio is preprocessed (resampling, normalization)  
3. Whisper model converts speech → text  
4. Text is cleaned and normalized  
5. Decision engine classifies:
   - **VALID** → normal speech  
   - **IGNORE** → filler/noise  
   - **INTERRUPT** → meaningful user input  
6. If INTERRUPT → agent stops speaking (TTS control)

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install torch transformers numpy scipy soundfile
