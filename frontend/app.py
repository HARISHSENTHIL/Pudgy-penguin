import gradio as gr
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8001"

def generate_gif(prompt, use_enhancer, lora_scale, image_seed, video_seed):
    if not prompt.strip():
        return None, "Please enter a prompt"

    try:
        image_seed_val = int(image_seed) if image_seed else None
        video_seed_val = int(video_seed) if video_seed else None
    except ValueError:
        return None, "Seeds must be integers"

    payload = {
        "prompt": prompt,
        "use_prompt_enhancer": use_enhancer,
        "lora_scale": lora_scale,
        "image_seed": image_seed_val,
        "video_seed": video_seed_val
    }

    try:
        response = requests.post(f"{API_URL}/generate", json=payload)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data["job_id"]

        status_text = "⏳ Job queued..."
        yield None, status_text

        while True:
            status_response = requests.get(f"{API_URL}/status/{job_id}")
            status_response.raise_for_status()
            status_info = status_response.json()

            if status_info["status"] == "completed":
                gif_response = requests.get(f"{API_URL}/download/{job_id}")
                gif_response.raise_for_status()

                output_path = Path(f"temp_{job_id}.gif")
                output_path.write_bytes(gif_response.content)

                yield str(output_path), f"✅ Complete!"
                break

            elif status_info["status"] == "failed":
                error = status_info.get("error_message", "Unknown error")
                yield None, f"❌ Failed: {error}"
                break

            elif status_info["status"] == "processing":
                status_text = f"🎨 Generating... (Job: {job_id})"
                yield None, status_text

            time.sleep(3)

    except requests.exceptions.RequestException as e:
        yield None, f"❌ API Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(), title="Pudgy GIF Generator") as app:
    gr.Markdown("# 🐙 OPEN Pudgy GIF Generator")
    gr.Markdown("Create animated Pudgy Penguin GIFs with OpenLedger")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="pudgy penguin as ironman",
                lines=3
            )

            with gr.Accordion("Advanced Options", open=False):
                use_enhancer = gr.Checkbox(label="Use AI Prompt Enhancer", value=True)
                lora_scale = gr.Slider(0.6, 1.2, value=0.8, step=0.1, label="LoRA Scale")
                image_seed = gr.Textbox(label="Image Seed (optional)", placeholder="42")
                video_seed = gr.Textbox(label="Video Seed (optional)", placeholder="123")

            generate_btn = gr.Button("Generate GIF", variant="primary", size="lg")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Status", interactive=False)
            gif_output = gr.Image(label="Generated GIF", type="filepath")

    generate_btn.click(
        fn=generate_gif,
        inputs=[prompt_input, use_enhancer, lora_scale, image_seed, video_seed],
        outputs=[gif_output, status_output]
    )

    # gr.Markdown("---")
    # gr.Markdown("Powered by FLUX + Pudgy LoRA + CogVideoX")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
