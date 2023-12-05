import json
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline
import torch

model = hf_hub_download(
    repo_id="Remilistrasza/XXMix_9realisticSDXL", 
    filename="xxmix9realisticsdxl_v10.safetensors", 
    revision="41e7c1aa4e871961f7188b431a6e8218adbced6b"
)
base = StableDiffusionXLPipeline.from_single_file(
    model,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

base.safety_checker = None
#base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

def invoke(input_text):
    try:
        input_json = json.loads(input_text)
        prompt = input_json['prompt']
        negative_prompt = input_json.get('negative_prompt', "")
        steps = int(input_json.get('steps', 30))
        guidance_scale = int(input_json.get('guidance_scale', 8))
    except:
        prompt = input_text + " (SimplepositiveXLv1:0.7)"
        negative_prompt = ""
        steps = 30
        guidance_scale = 8
    
    negative_prompt_template = f'''(worst quality, low quality, illustration, 3d, 2d, 
                            painting, cartoons, sketch), tooth, open mouth, three fingers, four fingers, {negative_prompt}'''

    image = base(
        prompt=prompt, 
        #prompt_2=prompt_2, 
        negative_prompt=negative_prompt_template,
        #negative_prompt_2=negative_prompt_2,
        height=1000,
        width=768,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    image.save("generated_image.png")
    return "generated_image.png"
