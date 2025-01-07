# AutogenGraphicsMods

## Project Overview
AutogenGraphicsMods is a Python-based tool that automates the generation of graphics mods for images. By utilizing image captioning and text-to-image generation techniques, it produces styled graphics based on predefined or user-specified themes. The project is designed for users who want to create unique image assets efficiently.

## Features

- Prompt Generation: Automatically generate captions for input images using BLIP (Bootstrapped Language-Image Pretraining).
- Style Customization: Apply various artistic styles such as "Ghibli," "Anime," "GTA5," or realism to images.
- Diffusion Models: Leverages Stable Diffusion and other pipelines for high-quality image generation.
- Batch Processing: Process multiple images in a directory, with an option for user satisfaction checks.
- Customizable Parameters: Modify settings like inference steps, prompt length, and style preferences.
- Save Options: Save modded images to a specified directory with customized filenames.

## Installation/Dependencies
Ensure the following Python libraries are installed:

torch
transformers
diffusers
Pillow
Install them using pip:


`pip install torch transformers diffusers Pillow`

## Usage

Command Line Interface
Run the script using the command line with optional arguments:

`python AutogenGraphicsMods.py --filepath './images' --style anime --num_inference_steps 30`

## Arguments

Arguments used in the script

| Argument | Description | Default |
|----------|-------------|---------|
| --filepath |	Path to the directory containing input images. | './' |
| --save_folder |	Path to save the generated images. | './mods' |
| --style |	Style to apply (e.g., ghibli, anime, realism). | '' |
| --num_inference_steps |	Number of inference steps for image generation. |	25 |
| --max_length | Maximum length of the generated prompt string.	| 50 |
| --autogen_prompt | Whether to generate prompts automatically. | True |
| --no_autogen_prompt | Toggle autogen prompt feature off. | False |
| --predownloaded | Whether models are pre-downloaded to avoid runtime downloads. |	True |
| --not_predownloaded | Toggle pre-downloaded feature off. |	False |
| --full_auto | Skip satisfaction checks for a fully automated process. |	False |
| --full_auto_on | Toggle full auto feature on. |	True |

## How It Works
1. Prompt Generation:
   - Uses BLIP to analyze an image and generate a descriptive prompt.
   - Users can opt for manual prompt input if preferred.

2. Style Tokenization:
   - Based on the chosen style, a token or model is applied to adjust the image's artistic theme.

3. Image Generation:
   - Text-to-image diffusion models generate modded images.
   - Users can review generated images, providing feedback to refine them.

4. Save Results:
   - Outputs are saved in the specified directory, named with the applied style as a prefix.

## References
The following Huggingface models where used in this script. They are linked below.

- runwayml/stable-diffusion-v1-5
   - Now deprecated
- Salesforce/blip-image-captioning-base
   - https://huggingface.co/Salesforce/blip-image-captioning-base
- sd-concepts-library/gta5-artwork
   - https://huggingface.co/sd-concepts-library/gta5-artwork
- nitrosocke/Ghibli-Diffusion
   - https://huggingface.co/nitrosocke/Ghibli-Diffusion
- cagliostrolab/animagine-xl-3.1
   - https://huggingface.co/cagliostrolab/animagine-xl-3.1
- hakurei/waifu-diffusion
   - https://huggingface.co/hakurei/waifu-diffusion
- emilianJR/epiCRealism
   - https://huggingface.co/emilianJR/epiCRealism

## Future Work
- Implement new stable diffusion model as runwayml/stable-diffusion-v1-5 is now deprecated.
- Add new styles and new features in the style_token function.
- Implement a system for creating paper doll/layer styled images that allow for customization. ex. the paper doll style for unit sprites in Shadow Empire.
- Implement better handling of generated images that are flagged as NSFW. Currently they are simply returned as black images.

## Contact
For questions or contributions, please reach out via GitHub.
