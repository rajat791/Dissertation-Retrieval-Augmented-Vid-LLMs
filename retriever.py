import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CLIP_MODEL, DEFAULT_K


class Retriever:

    # Initializes the Retriever with a FAISS index, an array of frame names, and the CLIP model.
    def __init__(self, index, frame_names: np.ndarray):
        self.index       = index
        self.frame_names = frame_names
        self.model       = SentenceTransformer(CLIP_MODEL, device="cpu")

    # Extracts the numerical frame identifier from the path string.
    @staticmethod
    def _get_frame_num(path: str) -> int:
        return int(str(path).split("frame_")[1].replace(".jpg", ""))

    # Retrieves semantic frames matching the prompt using a diversity filter to prevent temporal clustering.
    def clip(
        self,
        question: str,
        options: list[str],
        k: int = DEFAULT_K,
        min_gap: int = None,
    ) -> list[str]:
        if min_gap is None:
            total_frames = len(self.frame_names)
            min_gap = max(30, total_frames // (k * 2))

        query        = question + " " + " ".join(options)
        query_vector = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)

        n_candidates       = min(k * 4, len(self.frame_names))
        distances, indices = self.index.search(query_vector, k=n_candidates)
        candidate_indices  = indices[0]
        candidate_frames   = [self.frame_names[i] for i in candidate_indices]

        selected       = []
        selected_times = []

        for frame_path in candidate_frames:
            fn = self._get_frame_num(frame_path)
            if all(abs(fn - t) >= min_gap for t in selected_times):
                selected.append(str(frame_path))
                selected_times.append(fn)
            if len(selected) == k:
                break

        if len(selected) < k:
            for frame_path in candidate_frames:
                fp = str(frame_path)
                if fp not in selected:
                    selected.append(fp)
                if len(selected) == k:
                    break

        print(
            f"[Retriever] CLIP selected frames at times: "
            f"{[self._get_frame_num(f) for f in selected]} "
            f"(min_gap={min_gap})"
        )
        return selected

    # Samples frames at static, regular intervals across the entire duration of the video.
    def uniform(self, k: int = DEFAULT_K) -> list[str]:
        total   = len(self.frame_names)
        indices = np.linspace(0, total - 1, k, dtype=int)
        frames  = [str(self.frame_names[i]) for i in indices]

        print(
            f"[Retriever] Uniform selected frames at times: "
            f"{[self._get_frame_num(f) for f in frames]}"
        )
        return frames