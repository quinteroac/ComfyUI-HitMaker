import os
import shutil
import time
import uuid

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT_DIR = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")


class Comfy:
    def __init__(self, server="http://127.0.0.1:8188"):
        self.server = server.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def run(self, graph, timeout=3600):
        r = requests.post(f"{self.server}/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the prompt: {r.text}")
        prompt_id = r.json()["prompt_id"]
        start = time.time()
        while time.time() - start < timeout:
            history = requests.get(f"{self.server}/history/{prompt_id}").json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    errors = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise RuntimeError(f"ComfyUI execution failed: {errors}")
                return entry["outputs"]
            time.sleep(2)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout}s")

    def free(self):
        try:
            requests.post(f"{self.server}/free", json={"unload_models": True, "free_memory": True})
        except requests.ConnectionError:
            pass  # No server running, so nothing holds the GPU.


def stage_input(path, name):
    """LoadAudio only lists the top level of input/, so audio is copied there under a flat name."""
    dest = os.path.join(INPUT_DIR, name + os.path.splitext(path)[1])
    shutil.copyfile(path, dest)
    return os.path.basename(dest)


def output_file(entry):
    return os.path.join(OUTPUT_DIR, entry.get("subfolder", ""), entry["filename"])
