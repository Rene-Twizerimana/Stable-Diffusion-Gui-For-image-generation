# -----------------------------
# app.py
# -----------------------------

# 1️⃣ Import needed libraries
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

# 2️⃣ Set device and dtype
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# 3️⃣ The dictionary mapping style names to style strings
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting, --ar 9:16 --hd --q 2",
    "watercolor": "painting, watercolors, pastel, composition",
}

# 4️⃣ Load Stable Diffusion model
@st.cache_resource(show_spinner=False)
def load_model():
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype
    )
    pipeline = pipeline.to(device)
    return pipeline

# 5️⃣ Generate images function
def generate_images(prompt, pipeline, n, guidance=7.5, steps=50, style="none"):
    style_text = style_dict.get(style, "")
    final_prompt = prompt + (", " + style_text if style_text else "")
    
    output = pipeline(
        [final_prompt] * n,
        guidance_scale=guidance,
        num_inference_steps=steps
    )
    return output.images

# 6️⃣ Main Streamlit GUI
def main():
    st.title("Stable Diffusion GUI")
    st.write("Generate images from text prompts using Stable Diffusion!")

    # Sidebar inputs
    num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=10, value=1)
    prompt = st.sidebar.text_area("Text-to-Image Prompt", value="A serene mountain landscape")

    guidance_help = "Lower values follow the prompt less strictly. Higher values risk distorted images."
    guidance = st.sidebar.slider("Guidance", 2.0, 15.0, 7.5, help=guidance_help)

    steps_help = "More steps produces better images but takes longer."
    steps = st.sidebar.slider("Steps", 10, 150, 50, help=steps_help)

    style = st.sidebar.selectbox("Style", options=list(style_dict.keys()))

    generate = st.sidebar.button("Generate Images")
    if generate:
        with st.spinner("Generating images..."):
            pipeline = load_model()
            images = generate_images(
                prompt, pipeline, num_images, guidance, steps, style
            )
            for im in images:
                st.image(im)

# 7️⃣ Run the app
if __name__ == "__main__":
    main()
