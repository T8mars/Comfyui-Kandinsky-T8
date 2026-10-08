"""Submit a Comfy API graph and save its real execution receipt."""
import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def request(url, data=None):
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Content-Type": "application/json"} if payload else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, payload, headers), timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="replace")) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path, help="API-format graph, from example_workflows/api")
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--model", help="Override the diffusion filename (no downloading)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--filename-prefix")
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text(encoding="utf-8"))
    for node in graph.values():
        inputs, kind = node["inputs"], node["class_type"]
        if kind == "UNETLoader" and args.model:
            inputs["unet_name"] = args.model
        if kind == "SaveVideo" and args.filename_prefix:
            inputs["filename_prefix"] = args.filename_prefix
        if kind in ("KSampler", "Kandinsky6Sampler"):
            for field in ("seed", "steps"):
                value = getattr(args, field)
                if value is not None:
                    inputs[field] = value
        if kind in ("Kandinsky6EmptyLatent", "ImageScale"):
            for field in ("width", "height"):
                value = getattr(args, field)
                if value is not None:
                    inputs[field] = value
            if kind == "Kandinsky6EmptyLatent" and args.frames is not None:
                inputs["length"] = args.frames
    url = args.url.rstrip("/")
    submitted = request(url + "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    prompt_id = submitted["prompt_id"]
    print(f"QUEUED {prompt_id}", flush=True)
    started = time.monotonic()
    next_log = started + 30
    while True:
        history = request(url + "/history/" + prompt_id)
        if prompt_id in history:
            receipt = {"prompt_id": prompt_id, "workflow": str(args.workflow), "elapsed_seconds": time.monotonic() - started,
                       "graph": graph, "history": history[prompt_id]}
            path = args.receipt or args.workflow.with_suffix(".receipt.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            status = receipt["history"]["status"]
            print(json.dumps({"prompt_id":prompt_id,"status":status,"outputs":receipt["history"]["outputs"],"receipt":str(path)}), flush=True)
            if status["status_str"] != "success":
                raise RuntimeError("Workflow failed; see the saved execution receipt")
            break
        if time.monotonic() >= next_log:
            print(f"RUNNING {prompt_id} elapsed={time.monotonic()-started:.0f}s", flush=True)
            next_log = time.monotonic() + 30
        time.sleep(3)


if __name__ == "__main__":
    main()
