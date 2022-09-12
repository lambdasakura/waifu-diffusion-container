from torch import autocast
import random
import datetime
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import json

# StableDiffusionPipeline's parameter
#
#  prompt: Union[str, List[str]],
#  height: Optional[int] = 512,
#  width: Optional[int] = 512,
#  num_inference_steps: Optional[int] = 50,
#  guidance_scale: Optional[float] = 7.5,
#  eta: Optional[float] = 0.0,
#  generator: Optional[torch.Generator] = None,
#  latents: Optional[torch.FloatTensor] = None,
#  output_type: Optional[str] = "pil",

model = StableDiffusionPipeline.from_pretrained(
        "hakurei/waifu-diffusion",
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=DDIMScheduler(
          beta_start=0.00085,
          beta_end=0.012,
          beta_schedule="scaled_linear",
          clip_sample=False,
          set_alpha_to_one=False,
        ),
).to("cuda")

prompt=''
num = 10

for i in range(num):
  params = {}
  params['prompt'] = prompt
  params['seed'] = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)
  generator = torch.Generator("cuda").manual_seed(params['seed'])
  params['step'] = 150
  params['scale'] = 7.5
  # samples = 1
  with autocast("cuda"):
    image = model(prompt,
              num_inference_steps=params['step'],
              guidance_scale=params['scale'],
              generator=generator
              )["sample"][0]
    now = datetime.datetime.now()
    base_filename = now.strftime('%Y%m%d_%H%M%S')
    path3 = base_filename + ".png"
    json_file = open("outputs/" + base_filename + ".json", mode="w")
    json.dump(params, json_file, indent=2, ensure_ascii=False)
    json_file.close()
    image.save("outputs/" + base_filename + ".png")
    