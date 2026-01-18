#!/usr/bin/env python3
"""
Qwen Text-to-Image Generator for Focal Psychology Book
Uses ComfyUI API with proper Qwen workflow nodes
"""

import json
import time
import urllib.request
from pathlib import Path
import random

COMFYUI_URL = "http://127.0.0.1:8190"
OUTPUT_DIR = Path("/mnt/c/Users/PC/focal-psychology/images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Chapter prompts
CHAPTER_PROMPTS = {
    1: "abstract human silhouette with glowing masks floating away revealing pure light within, dark ethereal background, cinematic, digital art",
    2: "human eye reflecting smaller eye inside infinite mirror effect, cosmic space, purple gold colors, surreal art",
    3: "figure at crossroads with multiple transparent versions looking from different angles, geometric patterns mystical atmosphere",
    4: "beam of golden light from eye narrow and wide simultaneously, particles in light, dark starry background ethereal",
    5: "iron filings around magnet forming patterns, magnets are glowing emotion spheres, abstract visualization",
    6: "person transforming into what they observe, gradient metamorphosis, butterfly from caterpillar concept artistic",
    7: "eye looking at itself through mirror portal, recursive infinite depth, glowing edges cosmic purple atmosphere",
    8: "glowing energy flowing through chakra points some bright some dim, energy economy visualization ethereal body",
    9: "flowing river of light with slow and fast sections, inertia visualization, golden blue gradients abstract motion",
    10: "compass needle pointing to glowing center, calibration to zero concept, minimalist cosmic background gold accents",
    11: "figure surrounded by translucent ghost versions, quantum superposition visualization, ethereal glow",
    12: "circle of masked advisors around central figure, council of voices, dramatic lighting theatrical atmosphere",
    13: "wave function collapsing into single particle, quantum physics art, purple gold energy decision moment",
    14: "scales balancing with glowing orbs fading and brightening, price of choice visualization",
    15: "rewinding spiral staircase, figure walking backwards and forwards simultaneously, time manipulation art",
    16: "infinite feedback loop spiral, ouroboros modern interpretation, glowing neon lines cyberpunk mystical",
    17: "breaking chain with explosion of light particles, pattern breaking moment, dramatic energy release",
    18: "new star forming from cosmic dust, attractor creation, gravitational pull visualization cosmic birth",
    19: "filter screens with information streams blocked and passing, information diet concept digital art",
    20: "anchor made of light embedded in ground, decision anchoring concept, golden glow stable foundation",
    21: "topographical map with glowing contour lines, consciousness territories, aerial mind landscape view",
    22: "golden cradle floating in void, point zero concept, ultimate safety visualization warm light emanating",
    23: "internal theater with actor versions of self, stage and audience same person, meta-theatrical",
    24: "border between waking and dreaming, half realistic half surreal landscape, liminal space art",
    25: "kaleidoscope of social masks rotating, colorful personality fragments, identity carousel dynamic motion",
    26: "eagle eye view from above clouds, strategic overview of life paths below, golden hour lighting",
    27: "workshop with tools for soul, internal craftsman space, warm workshop lighting, tools made of light",
    28: "kaleidoscope turning, pattern shift moment, reality fragments rearranging, colorful geometric",
    29: "phase transition ice to water, consciousness state change visualization, crystal structure melting",
    30: "health symbol with consciousness map, vital signs as territory markers, medical mystical art",
    31: "body and glowing spirit separating reuniting, duality concept, ethereal separation art",
    32: "empty chair with ghostly figure conversation, gestalt hot seat technique, dramatic single spotlight",
    33: "emotion as colorful shape inside transparent body, body-image work visualization, internal anatomy art",
    34: "old map redrawn with new lines, imprinting rewrite concept, parchment with glowing new paths",
    35: "conscious mind handing key to subconscious shadow, trust transfer visualization, yin yang energy",
    36: "table with chess figures representing life elements, constellation placement, strategic arrangement",
    37: "open book with fairy tale characters emerging, story diagnostic tool, magical realism art",
    38: "person dialoguing with symptom as entity, conversation with illness, empathetic meeting",
    39: "person acting as-if already changed, future self overlay on present, potential visualization",
    40: "embodiment of emotion through movement, psychodrama dance, energy flowing through posed body",
    41: "clear boundary line with warning signs, method limits visualization, protective barrier art",
    42: "helping hands reaching stopping at ethical line, where help ends, compassionate restraint",
    43: "controlled spark between electrodes, provocation as catalyst, dangerous contained energy",
    44: "hands cupping protective space around small light, holding space visualization, nurturing energy",
    45: "crystal clear water with perfect reflections, clean language concept, pristine communication art",
    46: "wall transforming into door, resistance becoming ally, obstacle metamorphosis, hopeful light",
    47: "mirror showing past face overlaid on present, transference recognition, temporal ghost effect",
    48: "ecosystem web with one element changing ripple effects, ecology of change, interconnected art",
    49: "doorway with gentle light, completion and exit, sunset through door, peaceful transition",
    50: "path continuing into glowing horizon, journey continues concept, endless road of growth golden light",
    "og": "book cover focal psychology, eye with concentric circles, purple gold cosmic theme, premium design typography",
    "hero": "abstract consciousness visualization, human silhouette with light rays layers, psychology art premium",
    "topology": "five concentric rings golden center, consciousness topology diagram, ethereal glowing circles cosmic"
}

def create_workflow(prompt: str, seed: int = None) -> dict:
    """Create ComfyUI workflow for Qwen text-to-image"""
    if seed is None:
        seed = random.randint(0, 2147483647)

    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            }
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
                "strength_model": 1.0
            }
        },
        "5": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["4", 0],
                "shift": 3.1
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": ""
            }
        },
        "8": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": 1328,
                "height": 1328,
                "batch_size": 1
            }
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0]
            }
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": "focal"
            }
        }
    }

def queue_prompt(workflow: dict) -> str:
    """Queue a workflow and return the prompt_id"""
    data = json.dumps({"prompt": workflow}).encode('utf-8')
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data)
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())['prompt_id']

def get_history(prompt_id: str) -> dict:
    """Get the history for a prompt"""
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    """Download generated image"""
    url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
    with urllib.request.urlopen(url) as response:
        return response.read()

def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Wait for prompt to complete"""
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(3)
    raise TimeoutError(f"Timeout for {prompt_id}")

def generate_image(name, prompt: str) -> str:
    """Generate a single image and save it"""
    print(f"Generating: {name}")
    print(f"  Prompt: {prompt[:50]}...")

    workflow = create_workflow(prompt)
    prompt_id = queue_prompt(workflow)
    print(f"  Queued: {prompt_id}")

    result = wait_for_completion(prompt_id)

    outputs = result.get('outputs', {})
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            for img in node_output['images']:
                data = get_image(img['filename'], img.get('subfolder', ''), img.get('type', 'output'))
                fname = f"chapter-{name:02d}.jpg" if isinstance(name, int) else f"{name}.jpg"
                path = OUTPUT_DIR / fname
                with open(path, 'wb') as f:
                    f.write(data)
                print(f"  Saved: {path}")
                return str(path)

    raise ValueError(f"No images for {name}")

if __name__ == "__main__":
    print("=" * 60)
    print("Focal Psychology Qwen Image Generator")
    print(f"Total: {len(CHAPTER_PROMPTS)} images")
    print("=" * 60)

    # Check ComfyUI
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats") as r:
            stats = json.loads(r.read())
            print(f"ComfyUI: {stats['system']['comfyui_version']}")
            print(f"PyTorch: {stats['system']['pytorch_version']}")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

    print("=" * 60)

    generated = []
    for name, prompt in CHAPTER_PROMPTS.items():
        try:
            generate_image(name, prompt)
            generated.append(name)
            time.sleep(2)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nGenerated {len(generated)} / {len(CHAPTER_PROMPTS)} images")
