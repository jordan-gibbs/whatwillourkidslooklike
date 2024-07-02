import streamlit as st
import base64
import requests
from openai import OpenAI
import replicate

# Initialize OpenAI client
client = OpenAI()

# Function to encode the image
def encode_image(image):
    return base64.b64encode(image).decode('utf-8')

# Function to get the image description from GPT-4
def get_image_description(encoded_image, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this person in the image based solely on the aspects of their face. Hair color, hair texture, eye color, skin color, and dominant facial features (nothing in regards to facial hair, hair style, or glasses or piercings). Focus on face shape and permanent facial fixtures). Do not mention clothing. make sure you mention whether they are generally attractive, average, or below average"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())
    return response.json()

# Function to generate the prompt
def prompt_gen(person1, person2):
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a prompt writer that will write a Stable Diffusion image prompt "
                                          "based on the descriptions of two people. You will output the most dominant facial features,"
                                          " the mixed skin color, and the mixed race if applicable. You are basically generating what "
                                          "a child would look like if two people were to have kids together. Try to ensure that the most"
                                          " dominant and unique traits are kept in the final description. You shall only output commas"
                                          "and no other punctuation. Here is an example output:\nsoutheast asian-white, light brown skin, square face,"
                                          "brown eyes, chiseled jaw, thin lips, high cheekbones, freckles, straight black hair"},
            {"role": "user", "content": f"Person 1: {person1}\nPerson 2: {person2}"}
        ]
    )
    return completion.choices[0].message.content

# Function to generate the image
def generate_image(prompt, age, gender, image1, image2, merge_strength, merge_noise):
    rep_token = st.secrets('REPLICATE_API_TOKEN')
    rep_client = replicate.Client(api_token=rep_token)

    output = rep_client.run(
        "fofr/image-merge-sdxl:5fd9159399134ae0dd7b06bbbaabe7e7c15dbfec8b038eddef2ca3aa03355620",
        input={
            "steps": 40,
            "width": 512,
            "height": 512,
            "prompt": f"an uncropped photo of a {age} year old {gender}, {prompt}, white background, photo cropped at shoulders",
            "image_1": image1,
            "image_2": image2,
            "base_model": "albedobaseXL_v13.safetensors",
            "batch_size": 1,
            "merge_strength": merge_strength,
            "negative_prompt": "wrinkles, old skin, elderly, unrealistic, drawing, art, painting, sketch, nsfw, porn, "
                               "nipple, genitals, shirtless, smile lines, crows feet, close up",
            "added_merge_noise": merge_noise,
            "disable_safety_checker": True
        }
    )
    return output

# Streamlit UI
# st.set_page_config(layout='wide')
st.title("What will your kids look like?")
# st.subheader("👨➕👩🟰❓")

with open('custom.css') as f:
    css = f.read()

# Inject CSS into Streamlit app using st.markdown
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    image1 = st.file_uploader("Upload a portrait of the father", type=["jpg", "jpeg", "png"])
    if image1 is not None:
        st.image(image1, caption='👨 The Father', width=150)

with col2:
    image2 = st.file_uploader("Upload a portrait of the mother", type=["jpg", "jpeg", "png"])
    if image2 is not None:
        st.image(image2, caption='👩 The Mother', width=150)

age = st.slider("Select the age", 2, 20, 11)
gender = st.radio("Select the gender", ["Boy", "Girl"])
if age <= 5:
    gender = f"toddler, {gender}"
    merge_strength = 0.2
    merge_noise = 1
elif age >= 6 and age <= 12:
    merge_strength = 0.4
    merge_noise = .9
else:
    merge_strength = 0.7
    merge_noise = 0.8

if st.button("Go!"):
    if image1 is not None and image2 is not None:
        # Encode the images
        with st.spinner("Analyzing face attributes and genetic traits..."):
            encoded1 = encode_image(image1.read())
            encoded2 = encode_image(image2.read())

            api_key = st.secrets('OPENAI_API_KEY')
            person1 = get_image_description(encoded_image=encoded1, api_key=api_key)
            person1 = person1['choices'][0]['message']['content']

            person2 = get_image_description(encoded_image=encoded2, api_key=api_key)
            person2 = person2['choices'][0]['message']['content']

        with st.spinner("Combining faces..."):
            SDXL_prompt = prompt_gen(person1, person2)

        with st.spinner("Creating a photo of the child..."):
            output_image = generate_image(SDXL_prompt, age, gender, image1, image2, merge_strength, merge_noise)

        st.image(output_image[0], caption=f"Predicted child at age {age}", width=300)

        # Download button for the generated image
        if output_image is not None:
            st.download_button(
                label="Download Image",
                data=output_image[0],
                file_name=f"predicted_child.png",
                mime="image/png"
            )

        if st.button("Try Again"):
            st.success("Change settings above.")
    else:
        st.warning("Please upload both images.")

