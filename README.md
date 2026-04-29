# VirtualBank Teller — FYP2

A conversational virtual bank teller built with Streamlit, Live2D, Whisper STT, Edge-TTS, and a local Ollama LLM.

---

## Prerequisites

Before running the app, install the following on your machine:

### 1. Python 3.10 – 3.13
Download from https://www.python.org/downloads/

### 2. Ollama
Download and install from https://ollama.com

### 3. Anaconda
Download and install from https://www.anaconda.com/download

After installing, pull the required model:
```
ollama pull hf.co/MaziyarPanahi/Qwen3-4B-GGUF:Q4_K_M
```
Then start the Ollama server (it runs automatically on install, but you can start it manually with):
```
ollama serve
```

### 3. FFmpeg (required for Whisper STT)
Whisper requires FFmpeg to decode audio. Install it and make sure it is on your PATH.

**Windows (via winget):**
```
winget install --id Gyan.FFmpeg
```
**Windows (via Chocolatey):**
```
choco install ffmpeg
```
After installing, open a new terminal and verify with `ffmpeg -version`.

### 4. CUDA (optional but recommended)
The app auto-detects an NVIDIA GPU for Whisper STT. Without CUDA it falls back to CPU, which is slower for transcription. Install the CUDA toolkit matching your driver from https://developer.nvidia.com/cuda-downloads

---

## Installation

### Step 1 — Create a virtual environment (recommended)
```
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install Python dependencies
```
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `--extra-index-url` for the CUDA-enabled PyTorch wheels. If you are on CPU only, you can replace the torch lines with the standard PyPI versions.

### Step 3 — Install PortAudio for microphone input
PyAudio requires PortAudio. The easiest way on Windows is to install it via Conda:
```
conda install -c conda-forge portaudio
```

---

## Running the App

Make sure the Ollama server is running, then from this folder:

```
streamlit run app.py
```

The app will open automatically at http://localhost:8501

**Default demo credentials:**
- User ID: `demo`
- Password: `VirtualBank2026`

---

## Features

| Feature | Description |
|---|---|
| Chat | Type natural language queries to the virtual teller |
| Voice input | Click the microphone button to speak (Whisper STT) |
| Text-to-speech | Teller responses are spoken aloud via Edge-TTS (requires internet) |
| Live2D avatar | Animated Haru Greeter avatar reacts to teller state (idle / thinking / speaking) |
| Workflows | Send Money, Pay Card, Pay Loan / Financing, Pay Bill |
| Modals | Promotions, Mailbox, Account Summary, System Maintenance |

---

## Project Structure

```
.
├── app.py                   # Main Streamlit application
├── action_registry.py       # Single source of truth for all clickable surfaces
├── teller_helpers.py        # State file and TTS cleanup utilities
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit theme and static serving config
├── workflows/
│   ├── send_money.py
│   ├── pay_card.py
│   ├── pay_loan.py
│   ├── pay_bill.py
│   └── modals.py
├── static/
│   ├── cubism4.min.js        # Pixi-live2d-display runtime (served locally)
│   ├── live2d_display.min.js
│   └── live2d/model/haru_greeter/   # Avatar model files
└── tests/                   # pytest test suite (72 tests)
```

---

## Running Tests

```
python -m pytest tests/ -v
```

Expected output: `72 passed`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on startup | Ensure Ollama is running: `ollama serve` |
| Avatar not loading | Make sure `streamlit.server.enableStaticServing = true` is in `.streamlit/config.toml` (it is by default) |
| Microphone not working | Check PyAudio is installed and your OS microphone permissions are enabled |
| TTS not playing | Edge-TTS requires outbound internet access to `speech.platform.bing.com` |
| Slow responses on CPU | Whisper `tiny.en` is used; install CUDA or reduce response verbosity |

---

## Credits

| Asset / Library | Author / Source | License |
|---|---|---|
| **Haru Greeter avatar model** (`haru_greeter_t05`) | Live2D Inc. | [Live2D Free Material License](https://www.live2d.com/download/sample-data/) — free for individual users and small businesses; non-public testing only for mid/large enterprises |
| **Live2D Cubism Core** (`cubism4.min.js`) | Live2D Inc. | [Live2D Proprietary Software License](https://www.live2d.com/en/sdk/license/) |
| **pixi-live2d-display** (`live2d_display.min.js`) | guansss | [MIT License](https://github.com/guansss/pixi-live2d-display/blob/master/LICENSE) |
| **Streamlit** | Streamlit Inc. | [Apache 2.0](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |
| **OpenAI Whisper** (`openai-whisper`) | OpenAI | [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) |
| **Edge-TTS** | `rany2` (community wrapper) | [GNU GPL v3](https://github.com/rany2/edge-tts/blob/master/LICENSE) — uses Microsoft Edge's TTS endpoint |
| **LangChain / LangChain-Ollama** | LangChain Inc. | [MIT License](https://github.com/langchain-ai/langchain/blob/master/LICENSE) |
| **Ollama** | Ollama Inc. | [MIT License](https://github.com/ollama/ollama/blob/main/LICENSE) |
| **Qwen3-4B-GGUF** (LLM) | Alibaba Cloud (original), quantised by MaziyarPanahi | [Apache 2.0](https://huggingface.co/Qwen/Qwen3-4B) |
| **SpeechRecognition** | Anthony Zhang | [BSD License](https://github.com/Uberi/speech_recognition/blob/master/LICENSE.txt) |
| **PyTorch** | Meta AI | [BSD License](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
