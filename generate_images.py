#!/usr/bin/env python3
"""
Image Generator for Focal Psychology Book
Uses ComfyUI API with Qwen workflow to generate chapter illustrations
"""

import json
import requests
import time
import os
import urllib.request
import uuid
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8190"
OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Chapter prompts - artistic, metaphorical images for each chapter
CHAPTER_PROMPTS = {
    # Part I: Nature of Attention
    1: "abstract human silhouette with glowing masks floating away, revealing pure light within, dark ethereal background, cinematic lighting, digital art",
    2: "human eye reflecting a smaller eye inside, infinite mirror effect, cosmic space background, purple and gold colors, surreal art",
    3: "figure standing at crossroads with multiple transparent versions looking from different angles, geometric patterns, mystical atmosphere",
    4: "beam of golden light from eye, narrow and wide simultaneously, particles in the light, dark background with stars, ethereal",
    5: "iron filings around magnet forming patterns, but magnets are glowing spheres of different emotions, abstract visualization",
    6: "person transforming into what they observe, gradient metamorphosis, butterfly emerging from caterpillar concept, artistic",
    7: "eye looking at itself through mirror portal, recursive infinite depth, glowing edges, cosmic purple atmosphere",
    8: "glowing energy flowing through body chakra points, some bright some dim, energy economy visualization, ethereal body",
    9: "flowing river of light with slow and fast sections, inertia visualization, golden and blue gradients, abstract motion",
    10: "compass needle pointing to glowing center point, calibration to zero concept, minimalist cosmic background, gold accents",

    # Part II: Mechanics of Choice
    11: "figure surrounded by multiple translucent ghost versions of themselves, quantum superposition visualization, ethereal glow",
    12: "circle of masked advisors around central figure, council of voices, dramatic lighting, theatrical atmosphere",
    13: "wave function collapsing into single particle, quantum physics art, purple and gold energy, moment of decision",
    14: "scales balancing with glowing orbs, some orbs fading as others brighten, price of choice visualization",
    15: "rewinding spiral staircase, figure walking backwards and forwards simultaneously, time manipulation art",
    16: "infinite feedback loop spiral, ouroboros modern interpretation, glowing neon lines, cyberpunk meets mystical",
    17: "breaking chain with explosion of light particles, pattern breaking moment, dramatic energy release",
    18: "new star forming from cosmic dust, attractor creation, gravitational pull visualization, cosmic birth",
    19: "filter screens with different information streams, some blocked some passing, information diet concept, digital art",
    20: "anchor made of light embedded in ground, decision anchoring concept, golden glow, stable foundation",

    # Part III: Topography of Consciousness
    21: "topographical map with glowing contour lines, consciousness territories, aerial view of mind landscape",
    22: "golden cradle floating in void, point zero concept, ultimate safety visualization, warm light emanating",
    23: "internal theater with multiple actor versions of self, stage and audience are same person, meta-theatrical",
    24: "border between waking and dreaming, half realistic half surreal landscape, liminal space art",
    25: "kaleidoscope of social masks rotating, colorful personality fragments, identity carousel, dynamic motion",
    26: "eagle eye view from above clouds, strategic overview of life paths below, golden hour lighting",
    27: "workshop with tools for soul, internal craftsman space, warm workshop lighting, tools made of light",
    28: "kaleidoscope turning, moment of pattern shift, reality fragments rearranging, colorful geometric",
    29: "phase transition like ice to water, consciousness state change visualization, crystal structure melting",
    30: "health symbol combined with consciousness map, vital signs as territory markers, medical meets mystical",

    # Part IV: Toolkit
    31: "body and glowing spirit image separating and reuniting, duality concept, ethereal separation art",
    32: "empty chair with ghostly figure conversation, gestalt hot seat technique, dramatic single spotlight",
    33: "emotion as colorful shape inside transparent body, body-image work visualization, internal anatomy art",
    34: "old map being redrawn with new lines, imprinting rewrite concept, parchment with glowing new paths",
    35: "conscious mind handing key to subconscious shadow, trust transfer visualization, yin yang energy",
    36: "table with chess-like figures representing life elements, constellation placement, strategic arrangement",
    37: "open book with fairy tale characters emerging, story as diagnostic tool, magical realism art",
    38: "person in dialogue with their own symptom as entity, conversation with illness, empathetic meeting",
    39: "person acting as-if already changed, future self overlay on present, potential visualization",
    40: "embodiment of emotion through movement, psychodrama dance, energy flowing through posed body",

    # Part V: Mastery and Boundaries
    41: "clear boundary line with warning signs, method limits visualization, protective barrier art",
    42: "helping hands reaching but stopping at ethical line, where help ends, compassionate restraint",
    43: "controlled spark between electrodes, provocation as catalyst, dangerous but contained energy",
    44: "hands cupping protective space around small light, holding space visualization, nurturing energy",
    45: "crystal clear water with perfect reflections, clean language concept, pristine communication",
    46: "wall transforming into door, resistance becoming ally, metamorphosis of obstacle, hopeful light",
    47: "mirror showing past face overlaid on present, transference recognition, temporal ghost effect",
    48: "ecosystem web with one element changing, ripple effects visible, ecology of change, interconnected",
    49: "doorway with gentle light, completion and exit, sunset through door, peaceful transition",
    50: "path continuing into glowing horizon, journey continues concept, endless road of growth, golden light",

    # Additional images
    "og": "book cover focal psychology, eye with concentric circles, purple and gold cosmic theme, title typography, premium design",
    "hero": "abstract consciousness visualization, human silhouette with light rays and layers, psychology art, premium quality",
    "topology": "five concentric rings with golden center, consciousness topology diagram, ethereal glowing circles, cosmic background"
}

# Workflow template based on the Qwen workflow
def create_workflow(prompt: str, seed: int = None) -> dict:
    if seed is None:
        seed = int(time.time() * 1000) % 2147483647

    return {
        "prompt": {
            "60": {
                "inputs": {
                    "images": ["75", 0],
                    "filename_prefix": "focal_psychology"
                },
                "class_type": "SaveImage"
            },
            "75": {
                "inputs": {
                    "unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
                    "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
                    "width": 1328,
                    "height": 1328,
                    "batch_size": 1,
                    "seed": seed,
                    "steps": 4,
                    "text": prompt
                },
                "class_type": "2c61139d-9c34-4c7e-a083-7a67cc4770ad"
            }
        }
    }

def queue_prompt(prompt: dict) -> str:
    """Queue a prompt and return the prompt_id"""
    data = json.dumps(prompt).encode('utf-8')
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data)
    req.add_header('Content-Type', 'application/json')

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        return result['prompt_id']

def get_history(prompt_id: str) -> dict:
    """Get the history for a prompt"""
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    """Download generated image"""
    url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
    with urllib.request.urlopen(url) as response:
        return response.read()

def wait_for_completion(prompt_id: str, timeout: int = 120) -> dict:
    """Wait for prompt to complete and return outputs"""
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} did not complete in {timeout} seconds")

def generate_image(name: str, prompt: str) -> str:
    """Generate a single image and save it"""
    print(f"Generating: {name}")
    print(f"  Prompt: {prompt[:60]}...")

    workflow = create_workflow(prompt)
    prompt_id = queue_prompt(workflow)
    print(f"  Queued: {prompt_id}")

    result = wait_for_completion(prompt_id)

    # Get the output image
    outputs = result.get('outputs', {})
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            for image_info in node_output['images']:
                image_data = get_image(
                    image_info['filename'],
                    image_info.get('subfolder', ''),
                    image_info.get('type', 'output')
                )

                # Determine output filename
                if isinstance(name, int):
                    output_filename = f"chapter-{name:02d}.jpg"
                else:
                    output_filename = f"{name}.jpg"

                output_path = OUTPUT_DIR / output_filename
                with open(output_path, 'wb') as f:
                    f.write(image_data)

                print(f"  Saved: {output_path}")
                return str(output_path)

    raise ValueError(f"No images in output for {name}")

def main():
    print("=" * 60)
    print("Focal Psychology Image Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"ComfyUI URL: {COMFYUI_URL}")
    print(f"Total images to generate: {len(CHAPTER_PROMPTS)}")
    print("=" * 60)

    # Check ComfyUI connection
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats") as response:
            stats = json.loads(response.read())
            print(f"ComfyUI version: {stats['system']['comfyui_version']}")
            print(f"PyTorch: {stats['system']['pytorch_version']}")
    except Exception as e:
        print(f"ERROR: Cannot connect to ComfyUI at {COMFYUI_URL}")
        print(f"  {e}")
        return

    print("=" * 60)

    generated = []
    errors = []

    for name, prompt in CHAPTER_PROMPTS.items():
        try:
            path = generate_image(name, prompt)
            generated.append((name, path))
            time.sleep(1)  # Small delay between generations
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((name, str(e)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated: {len(generated)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for name, error in errors:
            print(f"  {name}: {error}")

if __name__ == "__main__":
    main()
