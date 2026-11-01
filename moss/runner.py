"""MOSS-Music inference, kept out of the ComfyUI process.

bitsandbytes' int8 matmul segfaults under ComfyUI's VRAM manager, so the node talks to this file as a
worker over stdin/stdout (one JSON job per line) and the CLI imports it directly.
"""
import json
import os
import sys

import av
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

if __package__:
    from .modeling_moss_music import MossMusicModel
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from moss.modeling_moss_music import MossMusicModel

# Kept in full precision: the audio side is small and quantizing it hurts chord/key perception.
SKIP_QUANT = ["audio_encoder", "audio_adapter", "deepstack_audio_merger_list", "lm_head"]
SAMPLE_RATE = 16000


def load(path, precision):
    kwargs = {"dtype": torch.bfloat16}
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_QUANT)
        kwargs["device_map"] = {"": 0}
    elif precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=SKIP_QUANT)
        kwargs["device_map"] = {"": 0}
    else:
        # bf16 does not fit next to the KV cache on 16 GB cards, so let accelerate spill the tail to RAM.
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: int(torch.cuda.mem_get_info()[0] * 0.8), "cpu": "48GiB"}
    # The modeling file shipped with the weights encodes the whole clip at once; the vendored upstream
    # model encodes 4 s windows like training did, so full songs also fit its 2 min position table.
    model = MossMusicModel.from_pretrained(path, **kwargs).eval()
    # The processor's own from_pretrained loads the tokenizer without trust_remote_code, which prompts on stdin.
    processor_cls = get_class_from_dynamic_module("processing_moss_music.MossMusicProcessor", path)
    processor = processor_cls(AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True), enable_time_marker=True)
    return model, processor


def read_audio(path):
    """Mono float32 at 16 kHz from a .npy written by the node, or from any file PyAV decodes."""
    if path.endswith(".npy"):
        return np.load(path)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    with av.open(path) as container:
        chunks = [f.to_ndarray()[0] for frame in container.decode(audio=0) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray()[0] for f in resampler.resample(None)]
    return np.concatenate(chunks)


def split_thinking(text):
    if "</think>" in text:
        thinking, answer = text.split("</think>", 1)
        return thinking.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


def ask(model, processor, wav, question, max_new_tokens=4096, temperature=0.6, repetition_penalty=1.05, seed=0):
    inputs = processor(text=question, audios=[torch.from_numpy(wav)], return_tensors="pt").to(model.device)
    inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

    gen = {"max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty, "use_cache": True}
    if temperature > 0:
        torch.manual_seed(seed)
        gen.update(do_sample=True, temperature=temperature, top_p=0.95, top_k=20)
    else:
        gen["do_sample"] = False
    ids = model.generate(**inputs, **gen)
    return split_thinking(processor.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def serve(path, precision):
    # Libraries print to stdout; keep it for the protocol and send everything else to stderr.
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    model, processor = load(path, precision)
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        job = json.loads(line)
        try:
            thinking, answer = ask(model, processor, read_audio(job.pop("audio")), **job)
            out.write(json.dumps({"thinking": thinking, "answer": answer}) + "\n")
        except Exception as e:
            out.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        out.flush()


if __name__ == "__main__":
    serve(sys.argv[1], sys.argv[2])
