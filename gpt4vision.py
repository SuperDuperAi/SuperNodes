import re
import os
import base64
import torch
from openai import OpenAI
import numpy as np
from datetime import datetime
from PIL import Image
import io
import json
class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")
CUSTOM_CATEGORY = "comfyui_superduperai"

class SuperDuperVision:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        default_prompt = """
You're assistant for counting peoples on photo.
Find main person.
Return as dict:
1.  index - count from left-to-right, (index start zero). Very important RETURN JUST ONLY NUMBER WITHOUT ANY STRING, YOU HAVE TO RETURN INTEGER.Interval(0-10).
2. gender from ['male','female','no']

Example answer: {"index": 1, "gender": "female"}
 
"""
        return {
            "required": {
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default":default_prompt}),
            }
        }

    RETURN_TYPES = (
        "STRING",  # Full response content
        "STRING",  # Extracted summary (first two sentences)
        "IMAGE",   # Faded image tensor
        # "INTEGER", #RETURN INT AS OUTPUT
    )
    RETURN_NAMES = ("full_response", "summary",)
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        # Determine orientation based on aspect ratio
        aspect_ratio = images[0].width / images[0].height
        vertical_stack = aspect_ratio > 1

        if vertical_stack:
            # Vertical stacking for wider images
            fade_height = int(images[0].height * (fade_percentage / 100))
            total_height = sum(img.height for img in images) - fade_height * (
                    len(images) - 1
            )
            max_width = max(img.width for img in images)
            combined_image = Image.new("RGB", (max_width, total_height))

            y_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    y_offset = img.height - fade_height
                else:
                    for y in range(fade_height):
                        factor = y / fade_height
                        for x in range(max_width):
                            if x < images[i - 1].width and x < img.width:
                                pixel1 = images[i - 1].getpixel(
                                    (x, images[i - 1].height - fade_height + y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x, y_offset + y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((0, fade_height, img.width, img.height)),
                        (0, y_offset + fade_height),
                    )
                    y_offset += img.height - fade_height
        else:
            # Horizontal stacking for taller images
            fade_width = int(images[0].width * (fade_percentage / 100))
            total_width = sum(img.width for img in images) - fade_width * (
                    len(images) - 1
            )
            max_height = max(img.height for img in images)
            combined_image = Image.new("RGB", (total_width, max_height))

            x_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    x_offset = img.width - fade_width
                else:
                    for x in range(fade_width):
                        factor = x / fade_width
                        for y in range(max_height):
                            if y < images[i - 1].height and y < img.height:
                                pixel1 = images[i - 1].getpixel(
                                    (images[i - 1].width - fade_width + x, y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x_offset + x, y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((fade_width, 0, img.width, img.height)),
                        (x_offset + fade_width, 0),
                    )
                    x_offset += img.width - fade_width

        return combined_image

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_images(
        self,
        images,
        custom_prompt="",
        additive_prompt="",
        dynamic_prompt=False,
        words=100,
        fade_percentage=15.0,
    ):
        try:
            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##WORDS##", str(words))

                if additive_prompt:
                    full_prompt = f"{additive_prompt} {custom_prompt}"
                else:
                    full_prompt = (
                        custom_prompt if custom_prompt else "Analyze this image."
                    )

            # Prepare the messages for OpenAI API
            messages = [
                {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
            ]

            # Convert tensor images to PIL
            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)
            base64_image = self.encode_image(combined_image)

            # Add the image to the messages
            # Note: Adjust this part according to the OpenAI API's requirements
            # This is a placeholder for how you might include the image
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

            # Call the OpenAI API
            # Replace this with your actual API call
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages
            )

            faded_image_tensor = self.pil2tensor(combined_image)

            return (
                response.choices[0].message.content,
                self.extract_first_two_sentences(response.choices[0].message.content),
                faded_image_tensor,
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (
                error_message,
                error_message[:100],
                self.pil2tensor(error_image)
            )

class SuperDuperReactorOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_faces_index": (any, {})

            },
        }


    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "execute"
    CATEGORY = CUSTOM_CATEGORY

    def execute(self, input_faces_gpt):
        json_input = json.loads((input_faces_gpt))
        if not input_faces_gpt:
            input_faces_index = 0
        options = {
            "input_faces_order":"left-right",
            "input_faces_index": str(input_faces_index),
            "detect_gender_input":"no",
            "source_faces_order":"left-right",
            "source_faces_index":"0",
            "detect_gender_source":"no",
            "console_log_level":"1",
        }
        return (options,)

NODE_CLASS_MAPPINGS = {
    "SuperDuperVision":SuperDuperVision,  
    "SuperDuperReactorOptions":SuperDuperReactorOptions,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperDuperVision":"SuperDuperVision",
    "SuperDuperReactorOptions":"SuperDuperReactorOptions",
}