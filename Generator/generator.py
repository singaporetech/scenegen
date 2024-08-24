import torch

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from realesrgan import RealESRGANer
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import os,gc

def flush():
  gc.collect()
  torch.cuda.empty_cache()

def upscale(filename):
  img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
  model = RRDBNet(
    num_in_ch=3, 
    num_out_ch=3, 
    num_feat=64, 
    num_block=23, 
    num_grow_ch=32, 
    scale=4)
  upsampler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",
    model=model,
    tile=1024,
    half=True)
  output, _ = upsampler.enhance(img, outscale=2)

  cv2.imwrite(str(filename).strip(".png")+"-upscaled.png", output)
  flush()
  
def generate(inputPrompt, count):
    prompt = "scenegen, "+ inputPrompt + ", hires, 4k, ultra realist, high res, realistic, google street view, 360, ultra - wide shot, a photorealistic rendering, extremely clear and coherent, photorealistic, ultradetailed, 7 0 mm imax"
    negetivePrompt ="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    print(prompt)
    loraLocation = os.path.abspath("paranomicScenegen-small-04.safetensors")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",subfolder="scheduler",use_karras_sigmas=True,euler_at_final=True)
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16, cache_dir="sdxl-vae-cache", use_safetensors=True).to("cuda")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0", 
       torch_dtype=torch.float16, 
       use_safetensors=True, 
       variant="fp16", 
       cache_dir="sdxl-cache",
       scheduler = scheduler,
       vae = vae
       ).to("cuda")
    
    
    pipeline.load_lora_weights(loraLocation)
    pipeline.enable_xformers_memory_efficient_attention()
    seed = 61411 #int.from_bytes(os.urandom(2), "big")
    print("seed: "+ str(seed))

    pipeline.vae.use_tiling = False

    image = pipeline(prompt, 
                    negetive_prompt = negetivePrompt,
                    width = 1024, 
                    height = 512,
                    target_size =(1024,512),
                    num_inference_steps = 100,
                    guidance_scale=5.0, 
                    generator=torch.manual_seed(seed)
                    ).images[0]
    image.save("initial.png")
    del pipeline
    flush()

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    initialimage = Image.open("initial.png")
    image = refiner(prompt, image=initialimage).images[0]
    image.save("initial.png")
    del refiner
    flush()

    # Upscale image to 2048x1024 for impainting
    upscale("initial.png")

    #Split image and inpaint seam
    image = Image.open("initial-upscaled.png")
    width, height = image.size
    quarter = width // 4
    first_quarter = image.crop((0, 0, quarter, height))
    remainder = image.crop((quarter, 0, width, height))
    image.paste(remainder, (0, 0))
    image.paste(first_quarter, (quarter*3, 0))
    image.save("joined.png")

    #get seam
    seam_image = image.crop((quarter*2, 0, width, height))
    remainder = image.crop((0, 0, quarter*2, height))
    seam_image.save("seam-img.png")

    inpainterScheduler = DPMSolverMultistepScheduler.from_pretrained( "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",subfolder="scheduler",use_karras_sigmas=True)
    inpainterScheduler.set_timesteps(num_inference_steps = 50)
    inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
       "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
       torch_dtype=torch.float16, 
       variant="fp16", 
       cache_dir="sdxl-inpaint-cache",
       use_safetensors=True,
       scheduler = inpainterScheduler
       ).to("cuda")
    inpainter.enable_xformers_memory_efficient_attention()
    mask = Image.open("mask.png")
    inpaintPrompt = "Inpaint the image to seamlessly blend the edges and ensure all parts of the image are connected and cohesive. Remove any visible seams or artifacts and create a natural transition between the disconnected parts."
    inpaintedImg = inpainter(
       prompt=inpaintPrompt, 
       image=seam_image, 
       mask_image=mask, 
       guidance_scale=8.0,
       strength=0.99,
       generator=torch.manual_seed(seed)
       ).images[0]
    inpaintedImg.save("inpainted_seam.png")
    del inpainter
    flush()

    seam_image = Image.open("inpainted_seam.png")
    image.paste(remainder, (0, 0))
    image.paste(seam_image, (quarter*2, 0))
    image.save(f"scaleReady.png")

    # upscale to 4k
    upscale("scaleReady.png")
    os.replace("scaleReady-upscaled.png","output.png")
    return(os.path.abspath("output.png"))

if __name__ =="__main__":
  inputPrompt = input()
  generate(inputPrompt,1)
