import os
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from config import INDEXES_DIR, CLIP_MODEL, BATCH_SIZE


class Indexer:

    # Initializes the Indexer with the video ID, file paths, and the CLIP model.
    def __init__(self, video_id: str):
        self.video_id   = video_id
        self.index_path = os.path.join(INDEXES_DIR, f"{video_id}.faiss")
        self.names_path = os.path.join(INDEXES_DIR, f"{video_id}_names.npy")
        self.model      = SentenceTransformer(CLIP_MODEL, device="cpu")

    # Checks if the FAISS index and names files already exist on disk.
    def already_indexed(self) -> bool:
        return (
            os.path.exists(self.index_path)
            and os.path.exists(self.names_path)
        )

    # Embeds video frames in batches, normalizes them, and saves the FAISS index to disk.
    def build(self, frame_paths: list[str], force: bool = False) -> None:
        if self.already_indexed() and not force:
            print(f"[Indexer] Index already exists for {self.video_id}, skipping.")
            return

        print(f"[Indexer] Embedding {len(frame_paths)} frames...")
        embeddings = []

        for i in range(0, len(frame_paths), BATCH_SIZE):
            batch      = frame_paths[i : i + BATCH_SIZE]
            images     = [Image.open(f).convert("RGB") for f in batch]
            batch_emb  = self.model.encode(
                images,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.append(batch_emb)
            print(
                f"[Indexer] Embedded "
                f"{min(i + BATCH_SIZE, len(frame_paths))} / {len(frame_paths)}"
            )

        embeddings = np.vstack(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        os.makedirs(INDEXES_DIR, exist_ok=True)
        faiss.write_index(index, self.index_path)
        np.save(self.names_path, np.array(frame_paths))

        print(f"[Indexer] Saved index ({index.ntotal} vectors) to {self.index_path}")

    # Loads and returns the saved FAISS index along with the frame paths array.
    def load(self) -> tuple:
        if not self.already_indexed():
            raise RuntimeError(
                f"[Indexer] No index found for video {self.video_id}. "
                f"Run build() first."
            )
        index       = faiss.read_index(self.index_path)
        frame_names = np.load(self.names_path, allow_pickle=True)
        print(f"[Indexer] Loaded index with {index.ntotal} vectors.")
        return index, frame_names