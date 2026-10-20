<div align="center">

# Qwen PRIME 4B AI

**A Qwen3-4B model fine-tuned for coding assistance — with a quantized GGUF export for local inference.**

[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3--4B-615CED?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
[![Fine-tuning](https://img.shields.io/badge/Fine--tuning-Unsloth-FFB000?style=for-the-badge)](https://github.com/unslothai/unsloth)
[![Method](https://img.shields.io/badge/Method-QLoRA%20%2B%20LoRA-168C8C?style=for-the-badge)](https://huggingface.co/docs/peft/index)
[![Runtime](https://img.shields.io/badge/Runtime-Ollama-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

![Google Colab](https://img.shields.io/badge/Training-Google%20Colab%20T4-4285F4?style=flat-square&logo=googlecolab&logoColor=white)
![GGUF](https://img.shields.io/badge/Export-GGUF%20Q4_K_M-7A3E9D?style=flat-square)

</div>

## Overview

Qwen PRIME 4B AI adapts [`Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) using **supervised fine-tuning (SFT)** and **QLoRA/LoRA**.

- **Training:** Google Colab with an NVIDIA T4 GPU
- **Dataset:** conversational JSONL records using a `messages` field
- **Training stack:** Unsloth, Transformers, TRL, PEFT, Datasets
- **Export:** quantized `Q4_K_M` GGUF
- **Inference:** local Ollama runtime

> This is a fine-tuned model, not a model trained from scratch. Training uses Colab; inference can run locally after the model is downloaded.

## Pipeline

```mermaid
flowchart TD
    A["Conversational JSONL"] --> B["Train / Validation / Test Split"]
    B --> C["Load Qwen3-4B in 4-bit"]
    C --> D["Baseline Evaluation"]
    D --> E["Attach LoRA Adapters"]
    E --> F["Supervised Fine-Tuning"]
    F --> G["Save / Resume Checkpoints"]
    G --> H["Evaluate Fine-Tuned Model"]
    D --> I["Compare Results"]
    H --> I
    I --> J["Export GGUF Q4_K_M"]
    J --> K["Run with Ollama"]
```

## Dataset format

Each JSONL line contains a conversation:

```json
{
  "messages": [
    {"role": "user", "content": "Write a Python function to reverse a string."},
    {"role": "assistant", "content": "def reverse_string(text):\n    return text[::-1]"}
  ]
}
```

The training workflow uses separate `train`, `validation`, and `test` splits.

- Keep test examples out of training.
- Update dataset paths to match your environment.
- The dataset is not included by default.

## Results

Reported ROUGE scores from the completed run:

| Metric | Base model | Fine-tuned |
|---|---:|---:|
| ROUGE-1 | 21.37% | 30.64% |
| ROUGE-2 | 2.85% | 6.29% |
| ROUGE-L | 9.92% | 18.43% |

**Note:** These scores indicate text overlap, not code correctness. The evaluation used a small test sample, so treat the results as preliminary rather than a comprehensive benchmark.

## Key highlights

- **Parameter-efficient training:** trains LoRA adapters instead of updating every base-model weight.
- **Lower memory use:** 4-bit model loading supports resource-conscious fine-tuning.
- **Checkpoint recovery:** saves progress to persistent Google Drive storage.
- **Before/after comparison:** evaluates the base and fine-tuned model on held-out examples.
- **Portable inference:** exports a quantized GGUF model for Ollama.

## Run locally with Ollama

Requirements: [Ollama](https://ollama.com/) and the exported `.gguf` file.

1. Put the GGUF file in a directory.
2. Create a `Modelfile` beside it:

```dockerfile
FROM ./Qwen3-4B-Instruct-2507.Q4_K_M.gguf
```

3. Import and run:

```bash
ollama create qwen-prime-4b-ai -f ./Modelfile
ollama run qwen-prime-4b-ai
```

Check the installed model:

```bash
ollama list
```

If your GGUF filename differs, update the `FROM` path. Actual speed and memory use depend on context length, system RAM, and GPU offload.

## Reproducing training

1. Open the training notebook in Google Colab and select a GPU runtime.
2. Mount Google Drive if your dataset or checkpoints are stored there.
3. Install compatible library versions and configure dataset/output paths.
4. Confirm all dataset splits load correctly.
5. Run baseline evaluation, fine-tuning, and post-training comparison.
6. Export to GGUF and copy the completed file to persistent storage before the Colab session ends.

Checkpoint recovery requires checkpoint files to have been saved successfully.

## Limitations & privacy

- Fine-tuning does not guarantee correct, secure, or hallucination-free code.
- Results depend on dataset quality and coverage.
- Local Ollama inference can run offline once the model is installed; training is **not fully offline** because it uses Colab and downloads required assets.
- Keep private data and credentials out of public commits.
- Check the base model's licence before redistributing model weights.

## Technology

[![Qwen](https://img.shields.io/badge/Qwen-Model-615CED?style=flat-square)](https://huggingface.co/Qwen)
[![Unsloth](https://img.shields.io/badge/Unsloth-Fine--tuning-FFB000?style=flat-square)](https://github.com/unslothai/unsloth)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![TRL](https://img.shields.io/badge/TRL-SFT-8A63D2?style=flat-square)](https://github.com/huggingface/trl)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-168C8C?style=flat-square)](https://github.com/huggingface/peft)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-black?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com/)

---

<div align="center">

**Qwen PRIME 4B AI** · Fine-tuned in the cloud. Ready for local inference.

</div>
