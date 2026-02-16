import runpod
import torch
import os
from PIL import Image
import base64
from io import BytesIO

# Global model cache
current_model_name = None
pipe = None

def load_model(model_type):
    global pipe, current_model_name
    
    if model_type == current_model_name:
        return pipe
    
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
    
    from diffusers import Flux2KleinPipeline
    
    if model_type == "distilled":
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16
        )
    else:
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-base-9B",
            torch_dtype=torch.bfloat16
        )
    
    pipe.to("cuda")
    current_model_name = model_type
    return pipe

def load_lora(pipe, lora_path):
    pipe.load_lora_weights(lora_path)
    return pipe

def get_reference_images(image_path):
    images = []
    for f in sorted(os.listdir(image_path))[:3]:
        if f.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            images.append(Image.open(f"{image_path}/{f}"))
    return images

def handler(job):
    input_data = job["input"]
    
    prompt = input_data["prompt"]
    image_path = input_data["image_path"]
    model_type = input_data.get("model", "distilled")
    use_lora = input_data.get("use_lora", False)
    aspect_ratio = input_data.get("aspect_ratio", "1:1")
    
    ratios = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 896),
        "3:4": (896, 1152)
    }
    width, height = ratios.get(aspect_ratio, (1024, 1024))
    
    pipe = load_model(model_type)
    
    if model_type == "base" and use_lora:
        load_lora(pipe, "/app/loras/my_lora.safetensors")
    
    ref_images = get_reference_images(image_path)
    
    steps = 4 if model_type == "distilled" else 50
    
    result = pipe(
        prompt=prompt,
        image=ref_images,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=1.0 if model_type == "distilled" else 3.5,
    ).images[0]
    
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {"image": img_b64}

runpod.serverless.start({"handler": handler})