# **SenticWaveX: Real-time AI-driven Emotion Recognition** 🚀  

![Windows on Snapdragon](https://img.shields.io/badge/Windows%20on%20Snapdragon-Optimized-blue.svg)  
![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-Accelerated-purple.svg)  
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)  
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)  

## **📌 Overview**  
**SenticWaveX** is an advanced **real-time AI-driven emotion recognition system** optimized for **Windows on Snapdragon**.  
It processes live audio input, detects emotions using **WavLM**, and applies **ONNX optimization** for ultra-fast inference.  

🔹 **Deep Learning-powered Speech Emotion Analysis**  
🔹 **Optimized with ONNXRuntime for Snapdragon Acceleration**  
🔹 **Confidence Boosting & Emotion Smoothing for Accuracy**  
🔹 **Real-time Audio Processing with Sounddevice**  
🔹 **Multi-threaded Execution for Low Latency**  

---

## **🛠️ Technologies Used**  
- **Languages:** Python  
- **Frameworks & Libraries:** PyTorch, Transformers (Hugging Face), ONNXRuntime, Torchaudio, Sounddevice, NumPy  
- **APIs:** Hugging Face Model Hub (for WavLM), ONNX API, Sounddevice API  
- **Other Technologies:** ONNX, Qualcomm QNN Execution Provider, Multi-threading (Thread, Queue, Event)  

---

## **🎯 Features**  
✅ **Real-time Emotion Detection** – Classifies emotions: **Angry, Happy, Sad, Neutral, Surprised, Fearful, Disgusted**  
✅ **ONNX Optimization** – Converts WavLM model to ONNX for fast, hardware-accelerated inference  
✅ **Confidence Boosting** – Enhances prediction accuracy using **probability transition matrices**  
✅ **Adaptive Noise Handling** – Uses **SNR-based filtering** for cleaner speech processing  
✅ **Error Handling & Logging** – Custom exception handling with colored logs  

---

### **🔹 Prerequisites**  
Ensure you have the following installed:  
- **Python 3.10.8 or approx**  
- **pip** (Python package manager)  
- **ONNXRuntime** (`onnxruntime` for CPU, `onnxruntime-gpu` for NVIDIA, or `onnxruntime-qnn` for Snapdragon)  

### **🔹 Setup**  
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/RiturajSingh2004/SenticWaveX.git
cd SenticWaveX
```

2️⃣ Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
4️⃣ Download WavLM Model (Caches automatically)

```bash
python -c "from transformers import WavLMForSequenceClassification; WavLMForSequenceClassification.from_pretrained('microsoft/wavlm-base-plus')"
```
🚀 Usage
🔹 Running Real-time Emotion Detection
```bash
python SenticWaveX.py --verbose --confidence-threshold 0.6
```
🔹 Available Arguments
Argument	Description	Default
--duration	Duration to run the analysis (0 for unlimited)	0
--verbose	Enable detailed logs	False
--device	Specify an audio input device index	Default
--confidence-threshold	Confidence threshold for emotion detection (0.0-1.0)	0.6
--history-size	Number of past predictions to smooth results	5
Example:

```bash
python SenticWaveX.py --duration 10 --verbose --confidence-threshold 0.7
```
🖥️ How It Works
```bash
1️⃣ Audio Processing & Feature Extraction
Captures real-time audio from input devices using Sounddevice API
Applies preprocessing & normalization
Uses Wav2Vec2FeatureExtractor for speech feature extraction
2️⃣ Deep Learning-based Emotion Recognition
WavLM (Hugging Face) processes audio signals
Model predicts 7 emotion classes with softmax probabilities
3️⃣ ONNX Optimization & Hardware Acceleration
Converts PyTorch model to ONNX
Executes inference using ONNXRuntime
Uses QNN Execution Provider (Qualcomm) or CPU fallback
4️⃣ Confidence Boosting & Emotion Smoothing
Applies probability transition matrix to stabilize predictions
Uses signal-to-noise ratio (SNR) filtering for quality enhancement
Maintains prediction history (past 5 results) for consistent outputs
```
🛠️ Troubleshooting
🔹 Issue: No audio device detected

Try specifying a different device:
```bash
python SenticWaveX.py --device 1
```
Check available devices:
```python
import sounddevice as sd
print(sd.query_devices())
```
🔹 Issue: ONNX model conversion fails

Ensure ONNXRuntime is installed:
```bash
pip install onnxruntime
```
Try manually converting the model:
```python
python -c "import torch; from transformers import WavLMForSequenceClassification; model = WavLMForSequenceClassification.from_pretrained('microsoft/wavlm-base-plus'); torch.onnx.export(model, torch.randn(1, 8000), 'wavlm_optimized.onnx')"
```
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

