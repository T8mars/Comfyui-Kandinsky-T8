"""Create minimal native workflows from the licensed upstream node layouts."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    ("2", "models", "qwen_2.5_vl_7b.safetensors", "text_encoders", "Comfy-Org/Qwen-Image_ComfyUI", "1f12b17be14c89b026c51a91d67c32f84bb047bc", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
    ("2", "models", "clip_l.safetensors", "text_encoders", "Comfy-Org/HunyuanVideo_repackaged", "533a43813f01517d9e624f9e824a85bb94094120", "split_files/text_encoders/clip_l.safetensors"),
    ("3", "models", "hunyuan_video_vae_bf16.safetensors", "vae", "Comfy-Org/HunyuanVideo_repackaged", "533a43813f01517d9e624f9e824a85bb94094120", "split_files/vae/hunyuan_video_vae_bf16.safetensors"),
    ("4", "kandinsky6_required_files", "v1-44.pth", "audio_vae", "hkchengrex/MMAudio", "eb13a1a98fdbec91753775c57b074ccdfc60587c", "ext_weights/v1-44.pth"),
    ("4", "kandinsky6_required_files", "bigvgan_vocoder/bigvgan_generator.pt", "audio_vae", "nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb", "bigvgan_generator.pt"),
    ("4", "kandinsky6_required_files", "bigvgan_vocoder/config.json", "audio_vae", "nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb", "config.json"),
)


def make_graph(image, distilled, family="lite"):
    name = f"k6_{family}{'_distill' if distilled else ''}_int8_convrot.safetensors"
    scale = 0.417 if distilled else 0.5302
    caption = (
        "A cinematic close-up of a young adult woman in soft daylight. She looks into the camera and clearly says "
        "<S>Look there!<E>, her lips moving naturally with the words. She smoothly turns her head toward a window "
        "and finishes with a warm smile. Static camera, continuous shot."
        if image else "Two scarlet macaws with brilliant red, yellow and blue feathers perch on a mossy branch in a "
        "lush tropical rainforest. One spreads its wings and flaps them, then both turn their heads and gently "
        "preen each other. Sunlight streams through the green canopy. Cinematic nature documentary shot."
    )
    audio_caption = "A clear, natural female voice. Quiet indoor ambience, no music." if image else "Loud macaw squawks, rustling feathers and leaves, lively rainforest ambience with distant birdsong."
    graph = {}

    def add(key, kind, **inputs):
        graph[str(key)] = {"class_type": kind, "inputs": inputs}

    add(1, "UNETLoader", unet_name=name, weight_dtype="default")
    add(2, "DualCLIPLoader", clip_name1="qwen_2.5_vl_7b.safetensors", clip_name2="clip_l.safetensors", type="kandinsky5", device="default")
    add(3, "VAELoader", vae_name="hunyuan_video_vae_bf16.safetensors")
    add(4, "Kandinsky6AudioVAELoader", tod_vae="v1-44.pth", bigvgan_dir="bigvgan_vocoder", mode="44k", scaling_factor=scale,
        model_variant="pretrain / distill" if distilled else "base")
    add(5, "Kandinsky6TextEncode", clip=["2", 0], video_caption=caption, audio_caption=audio_caption)
    add(6, "Kandinsky6TextEncode", clip=["2", 0], video_caption="Static, cartoon, worst quality, low quality, deformed", audio_caption="")
    add(7, "Kandinsky6EmptyLatent", width=864, height=480, length=121, fps=24.0, batch_size=1)
    sampler, video_decode, audio_decode, video_create, video_save = (12, 14, 15, 18, 19) if image else (8, 9, 10, 11, 12)
    positive, negative, latent = ["5", 0], ["6", 0], ["7", 0]
    if image:
        add(8, "LoadImage", image="kandinsky6/kandinsky6_i2va_portrait.png")
        add(9, "ImageScale", image=["8", 0], upscale_method="bilinear", width=864, height=480, crop="center")
        add(10, "VAEEncode", pixels=["9", 0], vae=["3", 0])
        add(11, "Kandinsky6ImageToVideoAudio", positive=positive, negative=negative, empty_latent=latent, reference_latent=["10", 0])
        positive, negative, latent = ["11", 0], ["11", 1], ["11", 2]
    add(sampler, "Kandinsky6Sampler" if distilled else "KSampler", model=["1", 0], seed=202 if image else 42,
        steps=10 if distilled else 50, cfg=1.0 if distilled else 5.0, sampler_name="euler", scheduler="simple",
        positive=positive, negative=negative, latent_image=latent, denoise=1.0)
    decoded_latent = [str(sampler), 0]
    if image:
        add(13, "Kandinsky6RemoveReferenceLatent", joint_latent=decoded_latent)
        decoded_latent = ["13", 0]
    add(video_decode, "VAEDecode", samples=decoded_latent, vae=["3", 0])
    add(audio_decode, "Kandinsky6AudioVAEDecode", audio_vae=["4", 0], joint_latent=decoded_latent)
    add(video_create, "CreateVideo", images=[str(video_decode), 0], fps=["7", 1], audio=[str(audio_decode), 0])
    add(video_save, "SaveVideo", video=[str(video_create), 0], filename_prefix=f"kandinsky6/{family}{'_distill' if distilled else ''}_{'i2av' if image else 't2av'}",
        format="mp4", **{"format.codec": "h264", "format.codec.encoding": "auto"})
    return graph


def make_gui(graph, image, distilled, title):
    template_name = "Kandinsky 6.0 Image to Video+Audio.json" if image else "Kandinsky 6.0 Text to Video+Audio.json"
    template = json.loads((ROOT / "upstream_examples" / template_name).read_text(encoding="utf-8"))
    source_nodes = {str(node["id"]): node for node in template["nodes"]}
    nodes, links = [], []
    for key, api_node in graph.items():
        node = copy.deepcopy(source_nodes[key])
        node["type"] = api_node["class_type"]
        node["properties"] = {"Node name for S&R": node["type"]}
        node["inputs"] = [port for port in node.get("inputs", []) if port["name"] in api_node["inputs"] and isinstance(api_node["inputs"][port["name"]], list)]
        for port in node["inputs"]:
            port["link"] = None
        for port in node.get("outputs", []):
            port["links"] = []
        values = api_node["inputs"]
        kind = node["type"]
        widgets = [value for value in values.values() if not isinstance(value, list)]
        if kind in ("KSampler", "Kandinsky6Sampler"):
            widgets.insert(1, "fixed")
        if kind == "CreateVideo":
            widgets = [24.0, "auto", "sRGB", "none"]
        if kind == "SaveVideo":
            widgets = [values["filename_prefix"], "mp4", "h264", "auto"]
        if kind == "LoadImage":
            widgets.append("image")
        node["widgets_values"] = widgets
        node["order"] = len(nodes)
        nodes.append(node)
    by_id = {str(node["id"]): node for node in nodes}
    for key, api_node in graph.items():
        target = by_id[key]
        for name, value in api_node["inputs"].items():
            if not isinstance(value, list):
                continue
            source, slot = by_id[value[0]], value[1]
            index = next((i for i, port in enumerate(target["inputs"]) if port["name"] == name), None)
            if index is None:
                index = len(target["inputs"])
                target["inputs"].append({"name": name, "type": "FLOAT", "link": None, "widget": {"name": name}})
            port = target["inputs"][index]
            link_id = len(links) + 1
            port["link"] = link_id
            source["outputs"][slot]["links"].append(link_id)
            links.append([link_id, source["id"], slot, target["id"], index, port["type"]])
    sources = {(item["repo"], item["revision"], item["filename"]): item
               for item in json.loads((ROOT / "MODEL_SOURCES.json").read_text(encoding="utf-8"))}
    for key, field, name, directory, repo, revision, filename in COMPONENTS:
        source = sources[(repo, revision, filename)]
        by_id[key]["properties"].setdefault(field, []).append({"name": name, "directory": directory,
            "url": f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
            "size": source["size"], "sha256": source["actual_sha256"]})
    # Compact the original positions into a left-to-right native pipeline.
    positions = {"1":(50,50),"2":(50,280),"3":(50,490),"4":(50,690),"5":(470,50),"6":(470,390),"7":(470,720)}
    if image:
        positions.update({"8":(470,1000),"9":(850,1000),"10":(1190,1000),"11":(1200,450),"12":(1580,100),
                          "13":(1960,100),"14":(2300,50),"15":(2300,280),"18":(2660,100),"19":(3000,100)})
    else:
        positions.update({"8":(900,100),"9":(1300,50),"10":(1300,280),"11":(1660,100),"12":(2000,100)})
    for key, node in by_id.items():
        node["pos"] = list(positions[key])
    note_id = 30
    nodes.append({"id":note_id,"type":"MarkdownNote","pos":[50,1420 if image else 1030],"size":[680,260],"flags":{},"order":len(nodes),"mode":0,
                  "inputs":[],"outputs":[],"properties":{"kandinsky6_download_package":"kandinsky6"},"widgets_values":[
                      f"# {title}\nComfyUI 0.39+ with comfy-kitchen 0.2.37+. Download a ready INT8 DiT from "
                      "[Kandinsky-Comfy](https://huggingface.co/t8star/Kandinsky-Comfy) into diffusion_models. "
                      "The download button supplies independent TE/VAE/audio components. Optional conversion: tools/convert_int8.py.\n\n"
                      + ("PiFlow: 10 steps, CFG=1, denoise=1, audio scale=0.417. Sampler/scheduler dropdowns are unused for PiFlow."
                         if distilled else "Stock KSampler: 50 steps, CFG=5, Euler/simple, audio scale=0.5302, flow shift=5.")
                      + "\n\n864×480, 121 frames at 24 fps. "
                      + ("Choose Lite distill or Pro distill in UNETLoader."
                         if distilled else "Use Lite for Base; use the PiFlow examples for Lite distill or Pro distill.")
                  ]})
    return {"last_node_id":note_id,"last_link_id":len(links),"nodes":nodes,"links":links,"groups":[],
            "config":{},"extra":{"ds":{"scale":0.45,"offset":[20,30]}},"version":0.4}


def main():
    destination = ROOT / "example_workflows"
    (destination / "api").mkdir(parents=True, exist_ok=True)
    for image in (False, True):
        for distilled in (False, True):
            title = f"K6 INT8 {'PiFlow' if distilled else 'Base'} {'Image' if image else 'Text'} to Video+Audio"
            graph = make_graph(image, distilled)
            (destination / "api" / (title + ".json")).write_text(json.dumps(graph, indent=2)+"\n", encoding="utf-8")
            (destination / (title + ".json")).write_text(json.dumps(make_gui(graph,image,distilled,title),indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
