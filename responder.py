import math
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class Responder:

    TILE_W    = 336   
    TILE_H    = 336   
    TILE_COLS = 3     

    MODEL_ID  = "llava-hf/llava-v1.6-vicuna-13b-hf"

    # Initializes the responder mode and loads or reuses the LLaVA-Next model and processor.
    def __init__(
        self,
        mode: str = "single",
        model=None,
        processor=None,
    ):
        assert mode in ("single", "tiled"), "mode must be 'single' or 'tiled'"
        self.mode = mode

        if model is not None and processor is not None:
            self.model     = model
            self.processor = processor
            print(f"[Responder] Reusing shared model (mode={mode}).")
        else:
            print(f"[Responder] Loading LLaVA-Next from {self.MODEL_ID} (mode={mode})...")
            self.processor = LlavaNextProcessor.from_pretrained(self.MODEL_ID)
            self.model     = LlavaNextForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print(f"[Responder] Model loaded.")

    # Opens and returns the first (top-ranked) frame from the provided path list.
    def _load_single(self, frame_paths: list[str]) -> Image.Image:
        return Image.open(frame_paths[0]).convert("RGB")

    # Stitches multiple frames into a single, chronologically sorted grid image.
    def _build_tile_grid(self, frame_paths: list[str]) -> Image.Image:
        def _frame_num(path: str) -> int:
            return int(str(path).split("frame_")[1].replace(".jpg", ""))
        
        frame_paths = sorted(frame_paths, key=_frame_num)

        n      = len(frame_paths)
        cols   = self.TILE_COLS
        rows   = math.ceil(n / cols)
        grid_w = cols * self.TILE_W
        grid_h = rows * self.TILE_H

        grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

        for i, path in enumerate(frame_paths):
            frame = (
                Image.open(path)
                .convert("RGB")
                .resize((self.TILE_W, self.TILE_H), Image.LANCZOS)
            )
            col = i % cols
            row = i // cols
            grid.paste(frame, (col * self.TILE_W, row * self.TILE_H))

        return grid

    # Generates a prediction letter (A, B, C, or D) for a question by running visual language model inference.
    def answer(
        self,
        frame_paths: list[str],
        question: str,
        options: list[str],
    ) -> str:
        if not frame_paths:
            return "X"

        if self.mode == "single":
            image      = self._load_single(frame_paths)
            image_desc = (
                "This is a single frame extracted from a video at the most "
                "semantically relevant moment for the question below."
            )
        else:
            image      = self._build_tile_grid(frame_paths)
            image_desc = (
                f"This image contains {len(frame_paths)} frames extracted from "
                f"a video, arranged in chronological order from left to right "
                f"and top to bottom. Each cell represents a different moment "
                f"in the video. Use all frames together as visual evidence "
                f"to answer the question below."
            )

        options_text = "\n".join(options)
        prompt = (
            f"USER: <image>\n"
            f"{image_desc} "
            f"Select the best answer to the following multiple-choice "
            f"question based on the video. "
            f"Respond with only the letter (A, B, C, or D) of the correct option.\n\n"
            f"Question: {question}\n\n"
            f"{options_text}\n\n"
            f"The best answer is:\n"
            f"ASSISTANT:"
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device, torch.float16)

        input_len = inputs["input_ids"].shape[1]
        print(f"[Responder] Input tokens: {input_len} (mode={self.mode})")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,      
            )

        new_tokens = output[0][input_len:]
        raw        = self.processor.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        print(f"[Responder] Raw output: '{raw}'")

        for letter in ["A", "B", "C", "D"]:
            if letter in raw.upper():
                return letter
        return "X"