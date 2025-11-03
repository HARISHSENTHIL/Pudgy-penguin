import torch
from diffusers import (
    FluxPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from diffusers.utils import load_image, export_to_video
from PIL import Image
import subprocess
import os
from pathlib import Path
from openai import OpenAI
import json
from src.prompts import (
    PROMPT_ENHANCER_SYSTEM,
    get_prompt_enhancer_user_message,
    get_fallback_prompts
)
from src.config import Config


class PudgyGIFGenerator:
    def __init__(
        self,
        flux_model=None,
        lora_id=None,
        cogvideo_model=None,
        device=None,
        dtype=None,
        openai_api_key=None
    ):
        """
        Initialize the Pudgy GIF Generator pipeline.
        All parameters default to Config values from .env file.

        Args:
            flux_model: Base FLUX model (defaults to Config)
            lora_id: Pudgy Penguin LoRA adapter (defaults to Config)
            cogvideo_model: CogVideoX image-to-video model (defaults to Config)
            device: cuda/cpu (defaults to Config)
            dtype: torch dtype (defaults to Config)
            openai_api_key: OpenAI API key (defaults to Config)
        """
        # Use config defaults if not provided
        self.device = device or Config.DEVICE
        self.dtype = dtype or Config.get_dtype()
        self.output_dir = Config.OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

        # Thread locks for concurrent inference safety
        import threading
        self.flux_lock = threading.Lock()
        self.video_lock = threading.Lock()

        flux_model = flux_model or Config.FLUX_MODEL
        lora_id = lora_id or Config.LORA_ID
        cogvideo_model = cogvideo_model or Config.COGVIDEO_MODEL

        # Initialize OpenAI client for prompt enhancement
        self.openai_client = None
        api_key = openai_api_key or Config.OPENAI_API_KEY
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            print("🤖 OpenAI prompt enhancer enabled!")
        else:
            print("⚠️  OpenAI API key not provided - prompt enhancement disabled")

        print("🐧 Initializing Pudgy GIF Generator...")
        print(f"   Device: {self.device}")
        print(f"   Dtype: {self.dtype}")

        # Step 1: Load FLUX with LoRA and IP-Adapter
        print("\n📦 Loading FLUX.1-dev with Pudgy LoRA...")
        self.flux_pipe = FluxPipeline.from_pretrained(
            flux_model,
            torch_dtype=self.dtype
        ).to(self.device)

        # Load Pudgy LoRA
        print(f"   Loading LoRA: {lora_id}")
        self.flux_pipe.load_lora_weights(
            lora_id,
            adapter_name="pudgy"
        )

        # Set adapter scales
        self.flux_pipe.set_adapters(["pudgy"], adapter_weights=[0.8])

        # Note: IP-Adapter disabled - XLabs IP-Adapter not yet compatible with standard Diffusers
        # Use LoRA + seeds for consistency instead (see examples)
        self.ip_adapter_available = False

        print("✅ FLUX pipeline ready!")

        # Step 2: Load CogVideoX Image-to-Video
        print(f"\n🎬 Loading CogVideoX I2V: {cogvideo_model}")
        self.video_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            cogvideo_model,
            torch_dtype=self.dtype
        ).to(self.device)

        # Use DPM++ scheduler for better quality
        self.video_pipe.scheduler = CogVideoXDPMScheduler.from_config(
            self.video_pipe.scheduler.config
        )

        # Enable memory optimizations
        
        # self.video_pipe.enable_model_cpu_offload()
        self.video_pipe.vae.enable_slicing()
        self.video_pipe.vae.enable_tiling()

        print("✅ CogVideoX pipeline ready!")
        print("\n🚀 All systems ready! Ready to generate GIFs.\n")

    def enhance_prompt(self, simple_prompt):
        """
        Use OpenAI to enhance a simple user prompt into detailed image and motion prompts.

        Args:
            simple_prompt: Simple user description (e.g., "pudgy penguin as ironman")

        Returns:
            dict with 'image_prompt' and 'motion_prompt'
        """
        if not self.openai_client:
            print("⚠️  Prompt enhancement unavailable - using fallback prompts")
            return get_fallback_prompts(simple_prompt)

        print(f"🤖 Enhancing prompt: '{simple_prompt}'")

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": PROMPT_ENHANCER_SYSTEM},
                    {"role": "user", "content": get_prompt_enhancer_user_message(simple_prompt)}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )

            result = json.loads(response.choices[0].message.content)

            print("✅ Prompts enhanced!")
            print(f"   📝 Image: {result['image_prompt'][:80]}...")
            print(f"   🎬 Motion: {result['motion_prompt'][:80]}...")

            return result

        except Exception as e:
            print(f"❌ Prompt enhancement failed: {e}")
            print("   Falling back to simple prompts...")
            return get_fallback_prompts(simple_prompt)

    def generate_image(
        self,
        prompt,
        negative_prompt="blurry, low quality, distorted, deformed",
        width=None,
        height=None,
        num_inference_steps=None,
        guidance_scale=None,
        lora_scale=None,
        seed=None
    ):
        """
        Generate a single branded image using FLUX + LoRA.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid
            width/height: Image dimensions (defaults to Config)
            num_inference_steps: Quality vs speed (defaults to Config)
            guidance_scale: Prompt adherence (defaults to Config)
            lora_scale: LoRA strength (defaults to Config)
            seed: Random seed for reproducibility

        Returns:
            PIL Image
        """
        # Use Config defaults if not provided
        width = width or Config.DEFAULT_IMAGE_WIDTH
        height = height or Config.DEFAULT_IMAGE_HEIGHT
        num_inference_steps = num_inference_steps or Config.DEFAULT_IMAGE_STEPS
        guidance_scale = guidance_scale or Config.DEFAULT_IMAGE_GUIDANCE
        lora_scale = lora_scale or Config.DEFAULT_LORA_SCALE

        print(f"🎨 Generating image: {prompt[:50]}...")

        # Set LoRA scale
        self.flux_pipe.set_adapters(["pudgy"], adapter_weights=[lora_scale])

        # Prepare generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"   Seed: {seed}")

        # Generate with thread lock for concurrent inference safety
        with self.flux_lock:
            result = self.flux_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = result.images[0]

        print(f"✅ Image generated ({width}x{height})")
        return image

    def animate_to_video(
        self,
        image,
        motion_prompt,
        num_frames=None,
        num_inference_steps=None,
        guidance_scale=None,
        seed=None
    ):
        """
        Animate a static image into a video using CogVideoX.

        Args:
            image: PIL Image to animate
            motion_prompt: Describe the motion/animation
            num_frames: Video length (defaults to Config)
            num_inference_steps: Quality (defaults to Config)
            guidance_scale: Motion adherence (defaults to Config)
            seed: Random seed

        Returns:
            List of PIL Images (video frames)
        """
        # Use Config defaults if not provided
        num_frames = num_frames or Config.DEFAULT_NUM_FRAMES
        num_inference_steps = num_inference_steps or Config.DEFAULT_VIDEO_STEPS
        guidance_scale = guidance_scale or Config.DEFAULT_VIDEO_GUIDANCE

        print(f"🎬 Animating to video: {motion_prompt[:50]}...")
        print(f"   Frames: {num_frames} ({num_frames/8:.1f}s @ 8fps)")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"   Seed: {seed}")

        # Generate video frames with thread lock for concurrent inference safety
        with self.video_lock:
            video_frames = self.video_pipe(
                prompt=motion_prompt,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]

        print(f"✅ Video generated ({len(video_frames)} frames)")
        return video_frames

    def video_to_gif(
        self,
        frames,
        output_path,
        fps=None,
        max_width=None,
        optimize=True
    ):
        """
        Convert video frames to optimized GIF using ffmpeg.

        Args:
            frames: List of PIL Images
            output_path: Output .gif path
            fps: Frames per second (defaults to Config)
            max_width: Max width in pixels (defaults to Config)
            optimize: Use high-quality palette

        Returns:
            Path to created GIF
        """
        # Use Config defaults if not provided
        fps = fps or Config.DEFAULT_GIF_FPS
        max_width = max_width or Config.DEFAULT_GIF_WIDTH

        print(f"🎞️  Converting to GIF...")
        print(f"   FPS: {fps}, Max width: {max_width}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save frames as temporary MP4
        temp_mp4 = output_path.with_suffix('.mp4')
        export_to_video(frames, str(temp_mp4), fps=fps)

        if optimize:
            # High-quality GIF with palettegen (no banding)
            print("   Applying high-quality palette optimization...")
            cmd = [
                'ffmpeg', '-y', '-i', str(temp_mp4),
                '-vf',
                f'fps={fps},scale={max_width}:-1:flags=lanczos,'
                'split[s0][s1];[s0]palettegen=stats_mode=diff[p];'
                '[s1][p]paletteuse=dither=floyd_steinberg',
                '-loop', '0',
                str(output_path)
            ]
        else:
            # Fast conversion
            cmd = [
                'ffmpeg', '-y', '-i', str(temp_mp4),
                '-vf', f'fps={fps},scale={max_width}:-1',
                '-loop', '0',
                str(output_path)
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Clean up temp MP4
            temp_mp4.unlink()

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ GIF created: {output_path}")
            print(f"   Size: {file_size_mb:.2f} MB")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr.decode()}")
            raise

    def generate_gif(
        self,
        prompt,
        motion_prompt=None,
        output_name=None,
        # Prompt enhancement
        use_prompt_enhancer=True,
        # Image generation params
        image_width=None,
        image_height=None,
        image_steps=None,
        image_guidance=None,
        lora_scale=None,
        # Video generation params
        num_frames=None,
        video_steps=None,
        video_guidance=None,
        # GIF params
        gif_fps=None,
        gif_width=None,
        # Seeds
        image_seed=None,
        video_seed=None
    ):
        """
        End-to-end: Generate branded image → animate → create GIF.
        All parameters default to Config values from .env file.

        Args:
            prompt: Simple description (e.g., "pudgy penguin as ironman")
            motion_prompt: Description of motion (auto-enhanced if None and enhancer enabled)
            output_name: Output filename (auto-generated if None)
            use_prompt_enhancer: Use OpenAI to enhance prompts (default True)
            All other params: Default to Config values if not provided

        Returns:
            dict with 'image', 'frames', 'gif_path'
        """
        # Use Config defaults if not provided
        image_width = image_width or Config.DEFAULT_IMAGE_WIDTH
        image_height = image_height or Config.DEFAULT_IMAGE_HEIGHT
        image_steps = image_steps or Config.DEFAULT_IMAGE_STEPS
        image_guidance = image_guidance or Config.DEFAULT_IMAGE_GUIDANCE
        lora_scale = lora_scale or Config.DEFAULT_LORA_SCALE
        num_frames = num_frames or Config.DEFAULT_NUM_FRAMES
        video_steps = video_steps or Config.DEFAULT_VIDEO_STEPS
        video_guidance = video_guidance or Config.DEFAULT_VIDEO_GUIDANCE
        gif_fps = gif_fps or Config.DEFAULT_GIF_FPS
        gif_width = gif_width or Config.DEFAULT_GIF_WIDTH

        print("\n" + "="*60)
        print("🐧 PUDGY GIF GENERATION PIPELINE")
        print("="*60 + "\n")

        # Enhance prompts if enabled and OpenAI is available
        if use_prompt_enhancer and self.openai_client:
            enhanced = self.enhance_prompt(prompt)
            image_prompt = enhanced['image_prompt']
            if motion_prompt is None:
                motion_prompt = enhanced['motion_prompt']
            print()
        else:
            # Use simple prompt as-is
            image_prompt = prompt
            if motion_prompt is None:
                motion_prompt = f"{prompt}, gentle movement, subtle animation, smooth motion"

        # Step 1: Generate image
        print("STEP 1: Generating branded image with FLUX + LoRA")
        print("-" * 60)
        image = self.generate_image(
            prompt=image_prompt,
            width=image_width,
            height=image_height,
            num_inference_steps=image_steps,
            guidance_scale=image_guidance,
            lora_scale=lora_scale,
            seed=image_seed
        )

        # Save image
        if output_name is None:
            output_name = f"pudgy_{hash(prompt) % 10000}"

        image_path = self.output_dir / f"{output_name}_image.png"
        image.save(image_path)
        print(f"   Saved: {image_path}\n")

        # Step 2: Animate to video
        print("STEP 2: Animating with CogVideoX Image-to-Video")
        print("-" * 60)
        frames = self.animate_to_video(
            image=image,
            motion_prompt=motion_prompt,
            num_frames=num_frames,
            num_inference_steps=video_steps,
            guidance_scale=video_guidance,
            seed=video_seed
        )
        print()

        print("STEP 3: Converting to optimized GIF")
        print("-" * 60)
        gif_path = self.output_dir / f"{output_name}.gif"
        self.video_to_gif(
            frames=frames,
            output_path=gif_path,
            fps=gif_fps,
            max_width=gif_width,
            optimize=True
        )

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print(f"📁 Outputs:")
        print(f"   Image: {image_path}")
        print(f"   GIF:   {gif_path}")
        print()

        return {
            "image": image,
            "frames": frames,
            "gif_path": gif_path,
            "image_path": image_path
        }


if __name__ == "__main__":
    # Option 1: With OpenAI prompt enhancement (recommended)
    generator = PudgyGIFGenerator(
        device="cuda",
        dtype=torch.bfloat16,
        openai_api_key="your-openai-api-key-here"  # Or set OPENAI_API_KEY env var
    )

    result = generator.generate_gif(
        prompt="pudgy penguin as ironman",  
        output_name="pudgy_ironman",
        lora_scale=0.8,
        image_seed=42,
        video_seed=123,
        use_prompt_enhancer=True  # AI generates detailed prompts
    )