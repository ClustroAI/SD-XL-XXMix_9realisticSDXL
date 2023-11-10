import json
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline
import torch

model = hf_hub_download(repo_id="Remilistrasza/XXMix_9realisticSDXL", filename="xxmix9realisticsdxl_v10.safetensors")
base = StableDiffusionXLPipeline.from_single_file(
    model,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

base.safety_checker = None
#base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

def invoke(input_text):
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    negative_prompt = input_json['negative_prompt'] if 'negative_prompt' in input_json else ""
    steps = int(input_json['steps']) if 'steps' in input_json else 30
    guidance_scale  = int(input_json['guidance_scale']) if 'guidance_scale' in input_json else 8
    
    negative_prompt_template = f'''(worst quality, low quality, illustration, 3d, 2d, 
                            painting, cartoons, sketch), tooth, open mouth, three fingers, four fingers, {negative_prompt}'''

    image = base(
        prompt=prompt, 
        #prompt_2=prompt_2, 
        negative_prompt=negative_prompt_template,
        #negative_prompt_2=negative_prompt_2,
        height=1280,
        width=768,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    image.save("generated_image.png")
    return "generated_image.png"
