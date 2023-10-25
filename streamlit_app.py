# --- Required Libraries and Modules --- #
import replicate
import streamlit as st
import requests
import zipfile
import io

from utils import icon
from streamlit_image_select import image_select

# --- UI Configurations --- #
st.set_page_config(page_title="Replicate Image Generator",
                   page_icon=":bridge_at_night:",
                   layout="wide")

st.markdown("## Text to Image Generator")

# --- Initialize session state for generated images --- #
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

# --- Secret Sauce (API Tokens and Endpoints) --- #
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
REPLICATE_MODEL_ENDPOINTSTABILITY = st.secrets["REPLICATE_MODEL_ENDPOINTSTABILITY"]

# --- Placeholders for Images and Gallery --- #
generated_images_placeholder = st.empty()

# --- Sidebar Elements --- #
with st.sidebar:
    with st.form("my_form"):
        with st.expander("**Refine your output here**"):
            # Advanced Settings (for the curious minds!)
            width = st.number_input("Width of output image", value=1024)
            height = st.number_input("Height of output image", value=1024)
            num_outputs = st.slider(
                "Number of images to output", value=1, min_value=1, max_value=4)
            scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete',
                                                   'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
            num_inference_steps = st.slider(
                "Number of denoising steps", value=50, min_value=1, max_value=500)
            guidance_scale = st.slider(
                "Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
            prompt_strength = st.slider(
                "Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)", value=0.8, max_value=1.0, step=0.1)
            refine = st.selectbox(
                "Select refine style to use (left out the other 2)", ("expert_ensemble_refiner", "None"))
            high_noise_frac = st.slider(
                "Fraction of noise to use for `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
        prompt = st.text_area(
            ":orange[**Enter prompt:**]",
            value="Create an image of an astronaut on the moon with a massive rocket in the background")
        negative_prompt = st.text_area(":orange[**Negative prompt, what you don't want to see in the result**]",
                                       value="The absolute worst quality, distorted features")

        submitted = st.form_submit_button(
            "Submit", type="primary", use_container_width=True)

# --- Image Generation --- #
if submitted:
    with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
        st.write("⚙️ Model initiated")
        st.write("🙆‍♀️ Stand up and strecth in the meantime")
        try:
            # Only call the API if the "Submit" button was pressed
            if submitted:
                # Calling the replicate API to get the image
                with generated_images_placeholder.container():
                    all_images = []  # List to store all generated images
                    output = replicate.run(
                        REPLICATE_MODEL_ENDPOINTSTABILITY,
                        input={
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "num_outputs": num_outputs,
                            "scheduler": scheduler,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "prompt_stregth": prompt_strength,
                            "refine": refine,
                            "high_noise_frac": high_noise_frac
                        }
                    )
                    if output:
                        st.toast('Your image has been generated!', icon='😍')
                        # Save generated image to session state
                        st.session_state.generated_image = output

                        # Displaying the image
                        for image in st.session_state.generated_image:
                            with st.container():
                                st.image(image, caption="Generated Image",
                                         use_column_width=True)
                                # Add image to the list
                                all_images.append(image)

                                response = requests.get(image)
                    # Save all generated images to session state
                    st.session_state.all_images = all_images

                    # Create a BytesIO object
                    zip_io = io.BytesIO()

                    # Download option for each image
                    with zipfile.ZipFile(zip_io, 'w') as zipf:
                        for i, image in enumerate(st.session_state.all_images):
                            response = requests.get(image)
                            if response.status_code == 200:
                                image_data = response.content
                                # Write each image to the zip file with a name
                                zipf.writestr(
                                    f"output_file_{i+1}.png", image_data)
                            else:
                                st.error(
                                    f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}", icon="🚨")
                    # Create a download button for the zip file
                    st.download_button(
                        ":red[**Download All Images**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
            status.update(label="✅ Images generated!",
                          state="complete", expanded=False)
        except Exception as e:
            st.error(f'Encountered an error: {e}', icon="🚨")
else:
    pass