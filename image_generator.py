import torch
from diffusers import DiffusionPipeline
import argparse

def setup_device():
    if torch.cuda.is_available():
        print("Using GPU for image generation")
        return "cuda"
    print("GPU not available, using CPU for image generation")
    return "cpu"

def load_model(device, base_model="black-forest-labs/FLUX.1-dev", lora_model="creatorchain/Pudgy_2.0", lora_scale=0.7):
    print(f"Loading base model: {base_model}")
    pipe = DiffusionPipeline.from_pretrained(base_model)
    
    print(f"Loading LoRA weights: {lora_model}")
    pipe.load_lora_weights(lora_model)
    
    if hasattr(pipe, "set_adapters_scale"):
        pipe.set_adapters_scale(lora_scale)
    
    pipe = pipe.to(device)
    return pipe

def generate_image(pipe, prompt, output_path="generated_image.png", height=512, width=512, num_steps=50):
    try:
        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps
        ).images[0]
        
        image.save(output_path)
        print(f"Image successfully generated and saved as {output_path}")
        return True
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output image path")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--base-model", type=str, default="black-forest-labs/FLUX.1-dev", help="Base model ID or path")
    parser.add_argument("--lora-model", type=str, default="creatorchain/Pudgy_2.0", help="LoRA model ID or path")
    parser.add_argument("--lora-scale", type=float, default=0.7, help="Scale factor for LoRA weights (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    device = setup_device()
    print("Loading model...")
    pipe = load_model(
        device,
        base_model=args.base_model,
        lora_model=args.lora_model,
        lora_scale=args.lora_scale
    )
    
    print(f"Generating image with prompt: {args.prompt}")
    print(f"Using LoRA model: {args.lora_model} with scale: {args.lora_scale}")
    generate_image(pipe,args.prompt,args.output,args.height,args.width,args.steps)

if __name__ == "__main__":
    main()