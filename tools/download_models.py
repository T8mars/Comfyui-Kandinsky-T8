"""Download only the selected K6 components, with resumable and verified files."""
import argparse
import concurrent.futures
import hashlib
from http.client import IncompleteRead
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from filelock import FileLock, Timeout

VARIANTS = {
    "lite": ("Kandinsky-6.0-Lite-5s-Diffusers", "5686616f04318cbec7b0a4978b4ff1697395de15"),
    "lite-distill": ("Kandinsky-6.0-Lite-distill-5s-Diffusers", "efbaeb9961770cc82fece5b9cb2197049d8c983c"),
    "pro": ("Kandinsky-6.0-Pro-5s-Diffusers", "295ca976810e318fb45c870ba4fc72c9a37fd548"),
    "pro-distill": ("Kandinsky-6.0-Pro-distill-5s-Diffusers", "1018d5828716e8bea3de99609f45c924de9c5564"),
}
COMPONENTS = (
    ("Comfy-Org/Qwen-Image_ComfyUI", "1f12b17be14c89b026c51a91d67c32f84bb047bc", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", "text_encoders/qwen_2.5_vl_7b.safetensors"),
    ("Comfy-Org/HunyuanVideo_repackaged", "533a43813f01517d9e624f9e824a85bb94094120", "split_files/text_encoders/clip_l.safetensors", "text_encoders/clip_l.safetensors"),
    ("Comfy-Org/HunyuanVideo_repackaged", "533a43813f01517d9e624f9e824a85bb94094120", "split_files/vae/hunyuan_video_vae_bf16.safetensors", "vae/hunyuan_video_vae_bf16.safetensors"),
    ("hkchengrex/MMAudio", "eb13a1a98fdbec91753775c57b074ccdfc60587c", "ext_weights/v1-44.pth", "audio_vae/v1-44.pth"),
    ("nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb", "bigvgan_generator.pt", "audio_vae/bigvgan_vocoder/bigvgan_generator.pt"),
    ("nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb", "config.json", "audio_vae/bigvgan_vocoder/config.json"),
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def describe(repo, revision, filename, destination):
    if revision == "main":
        revision = read_json(f"https://huggingface.co/api/models/{repo}")["sha"]
    parent = filename.rsplit("/", 1)[0] if "/" in filename else ""
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    if parent:
        url += "/" + parent
    entries = read_json(url)
    entry = next(item for item in entries if item["path"] == filename)
    return {"repo": repo, "revision": revision, "filename": filename,
            "destination": destination, "size": entry["size"],
            "sha256": entry.get("lfs", {}).get("oid")}


def download(item, root):
    root = Path(root).resolve()
    destination = (root / item["destination"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(destination) + ".download.lock")
    artifacts = (destination, destination.with_suffix(destination.suffix + ".part"),
                 destination.with_suffix(destination.suffix + ".download.json"))
    if any(lock_path.resolve() == path.resolve() or (
        lock_path.exists() and path.exists() and lock_path.samefile(path)
    ) for path in artifacts):
        raise ValueError("Download lock must differ from the model, partial and receipt files")
    lock = FileLock(lock_path, timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"A download is already running for {destination}") from error
    try:
        return _download_locked(item, root, destination)
    finally:
        lock.release()


def _download_locked(item, root, destination):
    receipt_path = destination.with_suffix(destination.suffix + ".download.json")
    if destination.exists():
        actual = digest(destination)
        if destination.stat().st_size != item["size"] or (item["sha256"] and actual != item["sha256"]):
            raise ValueError(f"Existing file does not match upstream: {destination}")
        print(f"REUSED {item['destination']}", flush=True)
    else:
        partial = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(8):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item["size"]:
                if not item["sha256"] or digest(partial) == item["sha256"]:
                    break
                # This file belongs to the downloader. Restart a complete but
                # corrupt partial without touching an existing installed model.
                with partial.open("wb"):
                    pass
                offset = 0
            if offset > item["size"]:
                with partial.open("wb"):
                    pass
                offset = 0
            if shutil.disk_usage(root).free < item["size"] - offset + (2 << 30):
                raise OSError(f"Not enough free space for {destination}")
            url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['filename']}?download=true&resume={offset}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise ValueError("Server did not honor the resume offset")
                    last_log = time.monotonic()
                    with partial.open("ab" if offset else "wb") as stream:
                        while block := response.read(8 << 20):
                            stream.write(block)
                            offset += len(block)
                            if offset > item["size"]:
                                # Discard this downloader-owned oversized response
                                # so a corrected server can be retried later.
                                stream.truncate(0)
                                raise ValueError("Download exceeded upstream size")
                            if time.monotonic() - last_log >= 20:
                                print(f"DOWNLOAD {item['destination']} {offset / 1e9:.2f}/{item['size'] / 1e9:.2f} GB", flush=True)
                                last_log = time.monotonic()
                if offset != item["size"]:
                    raise OSError(f"Incomplete response: {offset}/{item['size']}")
                break
            except (OSError, urllib.error.URLError, IncompleteRead) as exc:
                print(f"RETRY {item['destination']} attempt={attempt + 1} {exc}", flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
        actual = digest(partial)
        if partial.stat().st_size != item["size"] or (item["sha256"] and actual != item["sha256"]):
            with partial.open("wb"):
                pass
            raise ValueError(f"Downloaded file failed verification: {partial}")
        if os.name == "nt":
            os.rename(partial, destination)
        else:
            os.link(partial, destination)
            try:
                partial.unlink()
            except OSError as error:
                print(f"Downloaded model installed; partial retained at {partial}: {error}", flush=True)
        print(f"VERIFIED {item['destination']}", flush=True)
    receipt = {**item, "actual_sha256": actual}
    temporary_receipt = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=receipt_path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary_receipt = Path(stream.name)
            stream.write(json.dumps(receipt, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_receipt, receipt_path)
    finally:
        if temporary_receipt is not None:
            try:
                temporary_receipt.unlink(missing_ok=True)
            except OSError as error:
                print(f"Could not remove download receipt temporary file {temporary_receipt}: {error}", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["lite"])
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    items = []
    for variant in dict.fromkeys(args.variants):
        repo, revision = VARIANTS[variant]
        items.append(describe("kandinskylab/" + repo, revision, "transformer/diffusion_pytorch_model.safetensors", f"sources/{variant}.safetensors"))
    if args.components:
        items.extend(describe(repo, revision, filename, dest) for repo, revision, filename, dest in COMPONENTS)
    plan = args.models / "download_plan.json"
    plan.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item, args.models) for item in items]
        receipts = [future.result() for future in futures]
    (args.models / "download_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE {len(receipts)} verified files", flush=True)


if __name__ == "__main__":
    main()
