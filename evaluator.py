import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from config import RESPONSES_DIR, DEFAULT_K
from src.indexer import Indexer
from src.retriever import Retriever
from src.responder import Responder


class Evaluator:

    # Initialises the evaluator, verifies the index, and sets up shared models.
    def __init__(self, video: dict, k: int = DEFAULT_K, output_dir: str = None):
        self.video      = video
        self.video_id   = video["video_id"]
        self.k          = k
        self.output_dir = output_dir or RESPONSES_DIR

        indexer = Indexer(self.video_id)
        if not indexer.already_indexed():
            raise RuntimeError(
                f"[Evaluator] No FAISS index found for video {self.video_id}. "
                f"Run Indexer.build() before creating Evaluator."
            )
        index, names   = indexer.load()
        self.retriever = Retriever(index, names)

        print("[Evaluator] Loading LLaVA-Next model (shared)...")
        self.responder_single = Responder(mode="single")
        self.responder_tiled  = Responder(
            mode="tiled",
            model=self.responder_single.model,
            processor=self.responder_single.processor,
        )
        print("[Evaluator] Model ready.")

    # Copies a list of frame image files to a target directory.
    def _copy_frames(self, frame_paths: list[str], target_dir: str) -> None:
        os.makedirs(target_dir, exist_ok=True)
        for path in frame_paths:
            src_path = Path(path)
            if src_path.exists():
                dst_path = os.path.join(target_dir, src_path.name)
                shutil.copy2(src_path, dst_path)
            else:
                print(f"  [Warning] Frame file missing from disk: {path}")

    # Runs evaluation for all questions using clip_single, clip_tiled, and uniform strategies.
    def run(self) -> dict:
        results = []

        os.makedirs(self.output_dir, exist_ok=True)
        existing = [
            f for f in os.listdir(self.output_dir)
            if f.startswith(self.video_id) and f.endswith(".json")
        ]
        run_num = len(existing) + 1
        
        run_frames_dir = os.path.join("saved_frames", f"{self.video_id}_run{run_num}")

        for q in self.video["questions"]:
            question    = q["question"]
            options     = q["options"]
            truth       = q["answer"]
            question_id = q["question_id"]

            print(f"\n[Evaluator] Question: {question[:80]}...")

            clip_frames    = self.retriever.clip(question, options, k=self.k)
            uniform_frames = self.retriever.uniform(k=self.k)

            clip_save_dir = os.path.join(run_frames_dir, question_id, "clip")
            unif_save_dir = os.path.join(run_frames_dir, question_id, "uniform")
            
            self._copy_frames(clip_frames, clip_save_dir)
            self._copy_frames(uniform_frames, unif_save_dir)

            clip_single_pred = self.responder_single.answer(
                clip_frames, question, options
            )
            clip_tiled_pred = self.responder_tiled.answer(
                clip_frames, question, options
            )
            uniform_pred = self.responder_tiled.answer(
                uniform_frames, question, options
            )

            results.append({
                "question_id":  question_id,
                "task_type":    q["task_type"],
                "question":     question,
                "options":      options,
                "ground_truth": truth,
                "clip_single": {
                    "frames":    clip_frames,
                    "predicted": clip_single_pred,
                    "correct":   clip_single_pred == truth,
                },
                "clip_tiled": {
                    "frames":    clip_frames,
                    "predicted": clip_tiled_pred,
                    "correct":   clip_tiled_pred == truth,
                },
                "uniform": {
                    "frames":    uniform_frames,
                    "predicted": uniform_pred,
                    "correct":   uniform_pred == truth,
                },
            })

            print(
                f"  CLIP single : {clip_single_pred} | "
                f"CLIP tiled : {clip_tiled_pred} | "
                f"Uniform : {uniform_pred} | "
                f"Truth : {truth}"
            )

        total          = len(results)
        single_correct = sum(r["clip_single"]["correct"] for r in results)
        tiled_correct  = sum(r["clip_tiled"]["correct"]  for r in results)
        unif_correct   = sum(r["uniform"]["correct"]     for r in results)

        output = {
            "video_id":  self.video_id,
            "url":       self.video["url"],
            "duration":  self.video["duration"],
            "domain":    self.video["domain"],
            "k":         self.k,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total":                total,
                "clip_single_correct":  single_correct,
                "clip_single_accuracy": round(single_correct / total, 3) if total else 0,
                "clip_tiled_correct":   tiled_correct,
                "clip_tiled_accuracy":  round(tiled_correct  / total, 3) if total else 0,
                "uniform_correct":      unif_correct,
                "uniform_accuracy":     round(unif_correct   / total, 3) if total else 0,
            },
            "results": results,
        }

        self._save(output, run_num)
        return output

    # Saves the structured evaluation summary to a JSON file.
    def _save(self, output: dict, run_num: int) -> str:
        filename = os.path.join(
            self.output_dir, f"{self.video_id}_run{run_num}.json"
        )

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[Evaluator] Results saved to {filename}")
        return filename