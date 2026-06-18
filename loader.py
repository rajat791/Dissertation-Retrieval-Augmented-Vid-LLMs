import pandas as pd
from datasets import load_dataset
from config import DATASET_NAME, DATASET_SPLIT


class VideoLoader:

    # Initializes the VideoLoader with an empty DataFrame.
    def __init__(self):
        self.df = None

    # Downloads the dataset and caches it as a pandas DataFrame.
    def load(self):
        ds      = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        self.df = pd.DataFrame(ds)
        return self

    # Returns a sorted list of unique values for a specified column.
    def get_unique_values(self, column: str, df: pd.DataFrame = None) -> list:
        source = df if df is not None else self.df
        return sorted(source[column].dropna().unique().tolist())

    # Filters the DataFrame based on optional duration, domain, sub-category, or task type criteria.
    def filter(
        self,
        duration=None,
        domain=None,
        sub_category=None,
        task_type=None,
    ) -> pd.DataFrame:
        filtered = self.df.copy()
        if duration:
            filtered = filtered[filtered["duration"] == duration]
        if domain:
            filtered = filtered[filtered["domain"] == domain]
        if sub_category:
            filtered = filtered[filtered["sub_category"] == sub_category]
        if task_type:
            filtered = filtered[filtered["task_type"] == task_type]
        return filtered

    # Extracts a structured list of unique video dictionaries from a filtered DataFrame.
    def get_videos(self, filtered_df: pd.DataFrame) -> list[dict]:
        videos = []
        for url in filtered_df["url"].unique():
            rows = filtered_df[filtered_df["url"] == url]
            videos.append({
                "video_id":  rows.iloc[0]["video_id"],
                "url":       url,
                "duration":  rows.iloc[0]["duration"],
                "domain":    rows.iloc[0]["domain"],
                "questions": rows[
                    ["question_id", "question", "options", "answer", "task_type"]
                ].to_dict("records"),
            })
        return videos

    # Retrieves a single structured video dictionary using its unique video ID.
    def get_video_by_id(self, video_id: str) -> dict:
        rows = self.df[self.df["video_id"] == video_id]
        if rows.empty:
            raise ValueError(
                f"[VideoLoader] Video ID '{video_id}' not found in dataset."
            )
        url = rows.iloc[0]["url"]
        return {
            "video_id":  video_id,
            "url":       url,
            "duration":  rows.iloc[0]["duration"],
            "domain":    rows.iloc[0]["domain"],
            "questions": rows[
                ["question_id", "question", "options", "answer", "task_type"]
            ].to_dict("records"),
        }