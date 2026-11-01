import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
import torchaudio

import comfy.model_management
import folder_paths

folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

RUNNER = os.path.join(os.path.dirname(__file__), "moss", "runner.py")

_worker = {}


def model_names():
    names = []
    for root in folder_paths.get_folder_paths("LLM"):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            config = os.path.join(root, name, "config.json")
            if os.path.isfile(config):
                with open(config) as f:
                    if json.load(f).get("model_type") == "moss_music":
                        names.append(name)
    return names


def model_path(name):
    for root in folder_paths.get_folder_paths("LLM"):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    raise FileNotFoundError(f"MOSS-Music model '{name}' not found in the LLM model folders.")


def worker(name, precision):
    key = (name, precision)
    if _worker.get("key") == key and _worker["proc"].poll() is None:
        return _worker["proc"]
    unload()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    logging.info(f"Starting MOSS-Music worker {name} ({precision})")
    proc = subprocess.Popen([sys.executable, RUNNER, model_path(name), precision],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    reply(proc)
    _worker.update(key=key, proc=proc)
    return proc


def reply(proc):
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"MOSS-Music worker exited with code {proc.wait()}; see the console above.")
    result = json.loads(line)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def unload():
    proc = _worker.pop("proc", None)
    _worker.clear()
    if proc is not None and proc.poll() is None:
        proc.stdin.close()
        proc.wait()


def ask(audio, question, name, precision, max_new_tokens, temperature, repetition_penalty, seed):
    proc = worker(name, precision)
    wav = torchaudio.functional.resample(audio["waveform"][0].float().mean(dim=0), audio["sample_rate"], 16000)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "audio.npy")
        np.save(path, wav.cpu().numpy())
        proc.stdin.write(json.dumps({"audio": path, "question": question, "max_new_tokens": max_new_tokens,
                                     "temperature": temperature, "repetition_penalty": repetition_penalty, "seed": seed}) + "\n")
        proc.stdin.flush()
        result = reply(proc)
    return result["thinking"], result["answer"]
