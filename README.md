# Enhancing Long-Form Video Question Answering via CLIP-Driven Semantic Frame Retrieval

This project investigates whether **question-aware semantic frame retrieval** can improve long-form video question answering compared to the standard approach of **uniform frame sampling**.

Video Large Language Models are constrained by fixed context windows, making it infeasible to pass every frame from a long video into the model. The common workaround is to sample a small number of frames uniformly across the video. However, this approach is question-agnostic and may miss the specific visual evidence needed to answer a query.

To address this, this project implements a **training-free, single-pass retrieval-augmented pipeline** that uses **CLIP** and **FAISS** to retrieve the most semantically relevant frames for each question before passing them to **LLaVA-Next 13B** for multiple-choice reasoning.

## Overview

Given a long-form video and a multiple-choice question, the pipeline:

1. Extracts frames from the video at 1 FPS.
2. Encodes each frame using CLIP ViT-B/32.
3. Stores frame embeddings in a FAISS vector index.
4. Encodes the question and answer options as a CLIP text embedding.
5. Retrieves the top `k = 5` semantically relevant frames.
6. Passes the retrieved frames to LLaVA-Next 13B.
7. Compares the generated answer against the Video-MME ground truth.

The core hypothesis is that selecting frames based on the question should provide stronger visual evidence than selecting frames at fixed temporal intervals.

## Motivation

Long-form videos create a context bottleneck for Video Large Language Models. A 45-minute video sampled at 1 FPS produces thousands of frames, which translates into hundreds of thousands or even millions of visual tokens depending on the model architecture.

Since models such as LLaVA-Next have fixed context windows, only a small subset of frames can be processed. Uniform sampling is simple but does not consider the question being asked. As a result, it can discard the precise frames needed for accurate question answering.

This project explores whether semantic retrieval can reduce this issue by selecting frames that are visually and semantically aligned with the query.

## Pipeline Architecture

The system is divided into three main stages.

### 1. Preprocessing and Indexing

Each video is processed once. Frames are extracted at 1 FPS and encoded using CLIP ViT-B/32 into 512-dimensional embeddings. These embeddings are stored in a FAISS index for efficient nearest-neighbour search.

### 2. Query-Aware Retrieval

At inference time, the question and answer options are concatenated and encoded as a CLIP text embedding. FAISS retrieves the top `k = 5` frame embeddings with the highest semantic similarity to the query.

An adaptive temporal windowing mechanism is used to reduce clustering, ensuring that retrieved frames are distributed across relevant moments rather than concentrated around a single timestamp.

### 3. Multimodal Reasoning

Retrieved frames are passed to LLaVA-Next 13B for answer generation.

Two semantic retrieval modes are evaluated:

* **CLIP Single**: passes only the highest-ranked retrieved frame.
* **CLIP Tiled**: arranges the top 5 retrieved frames into a tiled composite image and passes it as a single visual input.

A **uniform sampling baseline** is also implemented, using the same frame budget of `k = 5`.

## Evaluation

The pipeline is evaluated on a stratified sample of the **Video-MME benchmark**, covering:

* 590 videos
* 1,770 multiple-choice questions
* All three duration tiers: short, medium, and long
* All six Video-MME domains
* Multiple question categories, including object recognition, attribute perception, temporal reasoning, action reasoning, and spatial reasoning

The primary metric is multiple-choice answer accuracy.

## Results

| Method           |    Correct | Overall Accuracy |
| ---------------- | ---------: | ---------------: |
| Uniform Sampling | 742 / 1770 |            41.9% |
| CLIP Single      | 796 / 1770 |            45.0% |
| CLIP Tiled       | 805 / 1770 |            45.5% |

Semantic retrieval outperformed uniform sampling overall.

The best-performing condition, **CLIP Tiled**, achieved a **3.6 percentage point improvement** over the uniform baseline. Per-video win-rate analysis also showed that this improvement was consistent rather than being driven by a small number of outlier videos.

## Key Findings

### Semantic retrieval improves overall accuracy

Both CLIP-based retrieval strategies outperformed uniform sampling while using the same frame budget and the same downstream reasoning model. This shows that the frame selection strategy alone can improve long-form video question answering.

### The largest gains occur on visually grounded tasks

Semantic retrieval performed especially well on question types requiring specific visual evidence, including:

* Attribute perception
* Object reasoning
* Object recognition

These tasks align well with CLIP’s visual-semantic embedding space, where static visual concepts can be matched effectively against text queries.

### Temporal reasoning remains challenging

CLIP-based retrieval struggled on tasks requiring motion, causality, or event progression. This is because CLIP encodes frames independently and does not model temporal relationships between frames.

As a result, uniform sampling sometimes performed better on:

* Action reasoning
* Spatial reasoning
* Temporally dependent questions

### Longer videos did not produce a larger retrieval advantage

The initial expectation was that semantic retrieval would become increasingly useful as video duration increased. However, this was not fully supported.

The largest gain over uniform sampling appeared in short videos, while medium and long videos showed smaller improvements. This was likely because longer Video-MME videos contain more temporal reasoning questions, where static CLIP frame embeddings are less effective.

## Comparison with Published Baselines

| Model            | Params | Overall Accuracy |
| ---------------- | -----: | ---------------: |
| Random Baseline  |      — |            25.0% |
| Video-LLaVA      |     7B |            39.9% |
| LLaVA-NeXT-Video |     7B |            43.3% |
| Uniform Sampling |    13B |            41.9% |
| CLIP Single      |    13B |            45.0% |
| CLIP Tiled       |    13B |            45.5% |
| LLaVA-NeXT-Video |    34B |            52.5% |

CLIP Tiled exceeded the published LLaVA-NeXT-Video 7B result while using only five semantically selected frames. This suggests that question-aware retrieval can compensate for a restricted frame budget.

## Limitations

The project has several limitations:

* The evaluation covered 590 of the 900 Video-MME videos due to YouTube access and bot-detection issues.
* All methods used a fixed frame budget of `k = 5`; no ablation was conducted across different values of `k`.
* CLIP encodes static images and does not capture temporal motion or causality.
* LLaVA-Next was not specifically trained to interpret the tiled composite frame format.
* Runtime and latency were not formally benchmarked across all pipeline stages.

## Future Work

Potential extensions include:

* Incorporating audio, subtitles, or captions as additional retrieval signals.
* Replacing CLIP with a temporally aware video encoder.
* Testing the retrieval pipeline with larger reasoning backbones such as LLaVA-NeXT-Video 34B or Qwen2-VL.
* Evaluating different frame budgets to determine how performance scales with `k`.
* Running the full Video-MME benchmark for stronger statistical reliability.
* Measuring runtime and efficiency against iterative retrieval methods such as VideoAgent.

## Tech Stack

* Python
* CLIP ViT-B/32
* FAISS
* LLaVA-Next 13B
* PyTorch
* Video-MME
* SLURM / HPC batch evaluation

## Conclusion

This project demonstrates that **training-free, single-pass, question-aware semantic frame retrieval** can improve long-form video question answering compared to uniform sampling.

The results confirm that retrieving frames based on the question provides better visual evidence for downstream reasoning, especially for object-centric and visually grounded tasks. However, temporal reasoning remains a key limitation due to CLIP’s static frame-level representation.

Overall, the project shows that retrieval strategy is an important factor in long-form video understanding and provides a practical foundation for future work on efficient, scalable Video-LLM pipelines.
