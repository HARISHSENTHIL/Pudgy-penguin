PROMPT_ENHANCER_SYSTEM = """You are a creative prompt engineer specializing in generating prompts for AI image and video generation.
Your task is to take a simple user description and create:
1. A detailed IMAGE prompt for generating a high-quality Pudgy Penguin character image
2. A detailed MOTION prompt describing how the character should animate

Rules for IMAGE prompts:
- Always include "Pudgy Penguin" or "cute penguin mascot"
- Add visual details: lighting, style, quality descriptors
- Keep the core concept from user's input
- Make it suitable for FLUX model (detailed, descriptive)
- Include artistic style terms like "digital art", "professional", "vibrant colors"
- Add environmental context if relevant to the character's role
- Mention texture and material details when appropriate

Rules for MOTION prompts:
- Describe realistic, smooth movements appropriate for the character's role/action
- Include motion descriptors: "gentle", "smooth", "dynamic", "fluid", "natural"
- Keep movements coherent with the character's concept
- Suitable for CogVideoX model (describe the action, not just repeat the scene)
- Avoid static descriptions, focus on what moves and how
- Consider physics and natural movement patterns
- Include environmental interactions if relevant (flying through sky, swimming, etc.)

Return your response as a JSON object with keys: "image_prompt" and "motion_prompt"."""


def get_prompt_enhancer_user_message(simple_prompt: str) -> str:
    """
    Generate the user message for prompt enhancement.

    Args:
        simple_prompt: The simple user input (e.g., "pudgy penguin as ironman")

    Returns:
        Formatted user message string
    """
    return f"""Simple prompt: "{simple_prompt}"

Generate enhanced prompts for this Pudgy Penguin concept. Think about:
- What would this character look like in detail?
- What movements/actions would make sense for this concept?
- How can we make this visually appealing and animated smoothly?
- What environmental elements or effects would enhance the scene?

Return JSON only with "image_prompt" and "motion_prompt" keys."""


# Fallback prompts when OpenAI is unavailable
def get_fallback_prompts(simple_prompt: str) -> dict:
    """
    Generate basic fallback prompts when OpenAI is unavailable.

    Args:
        simple_prompt: The simple user input

    Returns:
        dict with 'image_prompt' and 'motion_prompt'
    """
    return {
        "image_prompt": f"A cute Pudgy Penguin mascot {simple_prompt}, high quality, detailed digital art, vibrant colors, professional illustration",
        "motion_prompt": f"{simple_prompt}, gentle movement, smooth animation, dynamic motion, natural flow"
    }
