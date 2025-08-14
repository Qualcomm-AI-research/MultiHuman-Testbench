# Multi-Human-Testbench: A Benchmark for Subject-Driven Image Generation with Multiple Identities 👥🖼️

This is the official repository for the **Multi-Human-Testbench** project.

![Main illustration for Multi-Human-Testbench](assets/main.jpg)

## Abstract 📝
Generation of images containing multiple humans, performing complex actions, while preserving their facial identities, is a significant challenge. A major factor contributing to this is the lack of a a dedicated benchmark. To address this, we introduce MultiHuman-Testbench, a novel benchmark for rigorously evaluating generative models for multi-human generation. The benchmark comprises 1800 samples, including carefully curated text prompts, describing a range of simple to complex human actions. These prompts are matched with a total of 5,550 unique human face images, sampled uniformly to ensure diversity across  age, ethnic background, and gender. Alongside captions, we provide human-selected pose conditioning images which accurately match the prompt. We propose a multi-faceted evaluation suite employing four key metrics to quantify face count, ID similarity, prompt alignment, and action detection. We conduct a thorough evaluation of a diverse set of models, including zero-shot approaches and training-based methods, with and without regional priors. We also propose novel techniques to incorporate image and region isolation using human segmentation and Hungarian matching, significantly improving ID similarity. Our proposed benchmark and key findings provide valuable insights and a standardized tool for advancing research in multi-human image generation.

---

## Environment ⚙️

We provide a `Dockerfile` in the `docker/` directory to set up the required environment. This ensures consistency and simplifies dependencies management.
1.  Build the dockerfile:
    ```bash
    docker build -t NAME:TAG -f docker/Dockerfile .
    ```
2.  Run the Dockerfile:
    ```bash
    docker run -it NAME:TAG  bash
    ```

---

## Data Preparation 🗂️

Before running the benchmark, you need to prepare the necessary data. Follow these steps:

1.  Download the required facial data:
    ```bash
    python data_generation/download_faces.py
    ```
*(Note: On rare occasions, HuggingFace throws the "Too many requests" error. Please re-try after sometime in this case).*

2.  (Optional) If you are not using docker, unzip the pose.zip images:
    ```bash
    cd data/
    unzip poses.zip 
    ```

---

## (Optional) Clone Third-Party Baseline (OmniGen) ⬇️

We provide an example script to run with the OmniGen baseline. If you are not using Docker, please clone this repo.

```bash
sh download_third_party.sh
```

*(This script will clone the OmniGen repository into a designated location, `third_party/`)*.

---

## Run Inference ▶️


Once the environment is set up, data is prepared, and the baseline is cloned, you can run inference to generate multi-human images. **If you would like to access these images outside the container, please make sure to copy them to a local directory.**

1. On Task 1: Reference based generation in the wild:
    ```bash
    python inference/generate_omnigen_images.py --output_path=OUTPUT_PATH
    ```
*(Note: If you are running this for the first time, HF might throw a "Consistency Check Failed" error. Please re-try running the same command and HF will resume download.).*

2. On Task 2: Reference based generation with pose priors:
    ```bash
    python inference/generate_omnigen_images.py --output_path=OUTPUT_PATH --pose_prior
    ```
*(Note: We run these experiments on an Nvidia Tesla A100 GPU).*

---

## Benchmark ▶️
To compute scores on generated images, you can use compute_scores.py:

```bash
python benchmarking/compute_scores.py --mllm_metrics --stored_result_path=OUTPUT_PATH
```

*(Note: For MLLM QA metrics, we use Gemini 2.0 Flash. You can choose to run without ``` --mllm_metrics ``` if you don't want to prompt it. To correctly setup Gemini, make sure you paste your API key in ``` utils/mllm_metrics.py ```).*

---
## License

This repository contains both code and data, each released under different licenses: the code is released under the BSD 3-Clause Clear license and the data is released under the MultiHuman-Testbench Dataset Research License. 
Please refer to the respective license files for details. 

---

## Citation 📚


If you use Multi-Human-Testbench in your research, please cite our paper (forthcoming):

```bibtex
@article{borse2025multihuman,
  title={MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans},
  author={Borse, Shubhankar and Choi, Seokeon and Park, Sunghyun and Kim, Jeongho and Kadambi, Shreya and Garrepalli, Risheek and Yun, Sungrack and Hayat, Munawar and Porikli, Fatih},
  journal={arXiv preprint arXiv:2506.20879},
  year={2025}
}
```

---

Let's evaluate multi-identity generation models!

