import os
import json
import commentjson
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from aiohttp import ClientSession
from PIL import Image
from io import BytesIO
import re
import concurrent.futures
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class LearnTools:

    # ----------------- Existing Utility Functions -----------------
    def monitor(self, data):
        print("~" * 100)
        print(json.dumps(data))
        print("#" * 100)

    def clean_string(self, item):
        return re.sub(r"[^a-zA-Z0-9\s\-_]", "", item)

    def load_data(self, data_dir):
        json_data = {}

        def load_file(file):
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf-8") as data:
                return file_name, commentjson.load(data)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            json_data.update(
                dict(
                    executor.map(
                        load_file,
                        [f for f in os.listdir(data_dir) if f.endswith(".json")],
                    )
                )
            )
        return json_data

    def removeIntegers(self, input_string):
        return re.sub(r"\s*\d", "", input_string)

    async def get_image(self, img_url, padding=100):
        try:
            async with ClientSession() as session:
                async with session.get(img_url) as response:
                    response.raise_for_status()
                    image = Image.open(BytesIO(await response.read())).convert("RGB")

                    scale = 1 / math.sqrt(10)  # 10 times smaller image
                    new_size = (
                        max(1, int(image.width * scale)),
                        max(1, int(image.height * scale)),
                    )

                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    return image

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Invalid URL ==>>{img_url}<<== : {str(e)}")

    def pil_to_cv2(self, pil_image):
        cv_image = np.array(pil_image)  # Convert PIL to NumPy array (RGB)
        if pil_image.mode == "RGBA":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        else:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Dhaka"))
        return dt.strftime(datefmt) if datefmt else dt.strftime("%Y-%m-%d %I:%M:%S %p")


    # ----------------- Image Display Functions -----------------
    def show_image(self, ax, img=None, title=None, cmap=None):
        """
        Display a single image on the given matplotlib axis.
        Automatically converts BGR to RGB for color images.
        Uses 'gray' cmap for 2D images if not provided.
        """
        if img is None:
            raise ValueError("No image provided.")

        # Determine if grayscale
        if len(img.shape) == 2:
            ax.imshow(img, cmap=cmap or "gray")
        # Color image BGR -> RGB
        elif len(img.shape) == 3 and img.shape[2] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img)  # fallback

        if title:
            ax.set_title(title)
        ax.axis("off")


    def show_multiple_images(self, image_plotting_data=[]):
        """
        Display multiple images dynamically from a list of dictionaries.

        Parameters:
            image_plotting_data (list of dicts): Each dict must have keys:
                - 'image': the image array
                - 'title': title for the image
                - 'cmap': optional colormap (can be None)
        """
        if not image_plotting_data:
            return  # Nothing to show

        n = len(image_plotting_data)

        # Automatically determine figure size
        fig, axs = plt.subplots(1, n, figsize=(10*n, 10))

        # Handle single image case
        if n == 1:
            axs = [axs]

        for i, data in enumerate(image_plotting_data):
            img = data.get("image")
            title = data.get("title", f"Image {i+1}")
            cmap = data.get("cmap", None)
            if img is not None:
                self.show_image(axs[i], img, title=title, cmap=cmap)

        plt.show()
