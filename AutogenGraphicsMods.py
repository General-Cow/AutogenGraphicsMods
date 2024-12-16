import os
import argparse

# BLIP imports
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import re

# Image Generation pipeline
from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline
import torch


def prompt_generator(filename, max_length=50, predownloaded=False):
    """
    Uses BLIP to autogenerate a prompt for an image.
    """

    if predownloaded == False:
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    elif predownloaded == True:
        processor = AutoProcessor.from_pretrained("./models/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("./models/blip-image-captioning-base")
        
    image = Image.open(filename)
    text = "A picture of"

    inputs = processor(images=image, text=text, return_tensors="pt")
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=max_length)
    generated_prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_prompt
    
    
def style_token(generated_prompt, style="", predownloaded=False):
    """
    This function takes the generated prompt and returns 3 variables.
    style_model tells the image generator what model to use to generate a new image
    model_type is used by the image generator for handling different models according to how they need to be called.
    style_prompt append the appropriate language to the original prompt
    """

    if style == "manual":
        if predownloaded == False:
            style_model = input("Enter the style model you want loaded: ")
        elif predownloaded == True:
            style_model = input("Enter the path to the model you wish to use: ")
            
        model_type = input("Input model_type per style_token function description: ")
        style_prompt = input("Enter the appropriate modification to the prompt: ")
        
    elif style == "gta5":
        if predownloaded == False:
            style_model = "sd-concepts-library/gta5-artwork"
        elif predownloaded == True:
            style_model = "./models/gta5-artwork"
            
        model_type = "token"
        style_prompt = "<gta5-artwork> style "
        
    elif style == "ghibli":
        if predownloaded == False:
            style_model = "nitrosocke/Ghibli-Diffusion"
        elif predownloaded == True:
            style_model = "./models/Ghibli-Diffusion"
            print("style_model predown = True")
        model_type = "model"
        style_prompt = "ghibli style "
        
    elif style == "anime":
        if predownloaded == False:
            style_model = "cagliostrolab/animagine-xl-3.1"
        elif predownloaded == True:
            style_model = "./models/animagine-xl-3.1"
            
        model_type = "adv_model"
        print("cagliostrolab's animagine-xL-3.1 is a more advanced model, as such it requires more specific prompting.")
        print("For the prompt use the following format: 1girl/1boy, character name, from what series, everything else in any order.")
        style_prompt = input("Your input: ")
    
    elif style == "anime2":
        if predownloaded == False:
            style_model = "hakurei/waifu-diffusion"
        elif predownloaded == True:
            style_model = "./models/waifu-diffusion"
            
        model_type = "adv_model"
        print(generated_prompt)
        print("Manually convert your initial prompt into booru tags.")
        print("A full list can be found at Danbooru or other boorus. NSFW Warning")
        print("Use your own discretion visiting this website. https://danbooru.donmai.us/")
        print("Initial prompt: ", generated_prompt)
        style_prompt = input("Your input: ")
    
    elif style == "realism":
        if predownloaded == False:
            style_model = "emilianJR/epiCRealism"
        elif predownloaded == True:
            style_model = "./models/epiCRealism"
            
        model_type = "model"
        style_prompt = ""

    else:
        pass

    return style_model, model_type, style_prompt
    
    
def image_generator(
    filename,
    style="",
    num_inference_steps=25,
    autogen_prompt=True,
    max_length=50,
    predownloaded=False,
    **kwargs
    ):

    """
    The image generator sets up and generates the image in the desired style.

    Parameters:
    - filename (str): The input file to process.
    - style (str): The desired style for image generation.
    - num_inference_steps (int): Number of inference steps for the model.
    - autogen_prompt (bool): Whether to auto-generate the prompt or take manual input.
    - max_length (int): Maximum length of the prompt.
    - predownloaded (bool): Whether to use a predownloaded model.
    - **kwargs: Additional arguments for the pipeline call.

    Returns:
    - PIL.Image: The generated image.
    """
    
    # Creates prompt through autogeneration of prompt or direct input
    if autogen_prompt:
        prompt = prompt_generator(filename, max_length=max_length, predownloaded=predownloaded)
    else:
        prompt = input("Input your basic prompt: ")
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

    # Auto appends for model_types token and model but skips for adv models to
    # allow for more complex prompting later.
    style_model, model_type, style_prompt = None, None, ""
    if style:
        style_model, model_type, style_prompt = style_token(prompt, style=style, predownloaded=predownloaded)
        if model_type != "adv_model":
            prompt = f"{style_prompt}{prompt}"


    model_path = "./models/stable-diffusion-v1-5" if predownloaded else "runwayml/stable-diffusion-v1-5"
    pipeline = AutoPipelineForText2Image.from_pretrained(
        style_model if model_type in ["model", "adv_model"] else model_path,
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    # Handle specific model types
    if model_type == "token":
        pipeline.load_textual_inversion(style_model)
    elif model_type == "adv_model":
        if style == "anime":
            append_loc = input("Append front or back of original prompt with f or b: ")
            if append_loc == "f":
                prompt = f"{style_prompt}, {prompt}"
            elif append_loc == "b":
                prompt = f"{prompt}, {style_prompt}"
            else:
                raise ValueError("Invalid input. Use input 'f' or 'b'.")
        elif style == "anime2":
            prompt = style_prompt

    return pipeline(prompt, num_inference_steps=num_inference_steps, **kwargs).images[0]
    
    
def generate_graphic_mods(
    filepath,
    save_folder="./mods/",
    style="",
    num_inference_steps=25,
    max_length=50,
    autogen_prompt=True,
    predownloaded=True,
    full_auto=False,
    **kwargs
    ):
    """
    This function calls all the functions on a directory of images  allowing you to create a whole
    set of images to replace the old ones. It is HIGHLY recommended you have all the models you will use
    predownloaded otherwise connection issues can cause the function to fail early.
    """
    
    # Obtain filenames for images to be modded.
    image_names = os.listdir(filepath)
    
    # Generate modded images for each original image.
    image_list = []
    for img in image_names:
        image_list.append(image_generator(filename=f"{filepath}{img}", style=style, num_inference_steps=num_inference_steps,
                                          max_length=max_length, autogen_prompt=autogen_prompt, predownloaded=predownloaded,
                                          **kwargs))

    # Satisfaction check for each image followed by regenerating unsatisfacotry images until satisfied.
    # Default is set to skip this.
    if full_auto == False:
        satisfaction_list = []
        while sum(satisfaction_list) < len(satisfaction_list):
            for img in image_list:
                img
                satisfaction_check = input("Are you satisfied with this image? y/n: ")
                if satisfaction_check == "y":
                    satisfaction_list.append(1)
                elif satisfaction_check == "n":
                    satisfaction_list.append(0)
            #insert redo of specific indices
            redo_indices = [index for index, value in enumerate(satisfaction_list) if value == 0]
            if redo_indices != []:
                for idx in redo_indices:
                    image_list[idx] = image_generator(style=style, num_inference_steps=num_inference_steps,
                                                      max_length=max_length, autogen_prompt=autogen_prompt,
                                                      predownloaded=predownloaded, **kwargs)

    # Saves each image with style and original filename
    for idx in range(len(image_list)):
        image_list[idx].save(f"{save_folder}{style}_{image_names[idx]}")

    return image_list
    
    
def parse_args():
    # General settings
    parser = argparse.ArgumentParser(description="Autogenerate Graphics Mods")
    parser.add_argument("--filepath", type=str, default="./test/", help="filepath to input images")
    parser.add_argument("--save_folder", type=str, default="./mods/", help="Save location for output")
    
    # Generation settings
    parser.add_argument("--style", type=str, default="", help="Style of image generation")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of Inference Steps")
    parser.add_argument("--max_length", type=int, default=50, help="Max str length of prompt generator function")
    
    # Flags
    parser.add_argument("--autogen_prompt", action="store_true", help="Whether to use autogen prompt feature")
    parser.add_argument("--no_autogen_prompt", action="store_false", dest="autogen_prompt", help="Don't use autogen prompt feature")
    parser.set_defaults(autogen_prompt=True)
    
    parser.add_argument("--predownloaded", action="store_true", help="Model is predownloaded")
    parser.add_argument("--not_predownloaded", action="store_false", dest="predownloaded", help="Model is not predownloaded")
    parser.set_defaults(predownloadedt=True)
    
    parser.add_argument("--full_auto", action="store_false", help="Full auto toggle")
    parser.add_argument("--full_auto_on", action="store_true", dest="full_auto", help="Full auto toggle")
    parser.set_defaults(full_auto=False)

    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    
    generate_graphic_mods(
                    filepath=args.filepath,
                    save_folder=args.save_folder,
                    style=args.style,
                    num_inference_steps=args.num_inference_steps,
                    max_length=args.max_length,
                    autogen_prompt=args.autogen_prompt,
                    predownloaded=args.predownloaded,
                    full_auto=args.full_auto
                    )

    # Example call in the command line using different values from preset
    # python AutogenGraphicsMods.py --filepath './images/' --save_folder './out/' --style ghibli etc etc