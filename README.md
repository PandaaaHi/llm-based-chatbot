# LLM-based-Chatbot
A simple voice chatbot built on **ASR** -> **LLM** -> **TTS**. Press Enter to record your speech, the audio is transcribed by PaddleSpeech, answered by a local LLM served through Ollama (`qwen2.5:7b`), and the reply is synthesized and played back.

## Requirements
### 1. Hardware / system dependencies
A GPU is required — the ASR is hard-coded to run on device='gpu', and the PaddlePaddle build installed by install.sh targets CUDA 12.0.

- NVIDIA GPU with a driver supporting CUDA 12.0
- `sox` (provides the `rec` command used for recording)
- `alsa-utils` (provides the `aplay` command used for playback)

```bash
sudo apt update
sudo apt install -y sox alsa-utils
```

### 2. Conda environment
Python 3.10 is recommended:
```bash
conda create -n llm python=3.10
conda install -n llm -c conda-forge cudnn=8.9.7.29
conda activate llm
```

### 3. Python dependencies
```bash
sh install.sh
```
This installs PaddleSpeech (ASR + TTS), LangChain and the Ollama integration.

### 4. Ollama and the LLM model
```bash
# install Ollama first
ollama pull qwen2.5:7b
```

## Usage
Run the following script to chat with LLM:
```bash
conda activate llm
python voice_assistant.py
```