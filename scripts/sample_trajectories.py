import asyncio
import os
import json
import random
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from multidocqa.llm_client import VllmEndpointGenerator
from multidocqa.utils import compute_reward, load_data, create_simple_prompt

# Paths
CIVIL_CODE_FILE = (
    "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
)
DATASET_FILE = "data/processed/train.json"
TRAJECTORY_SAVE_PATH = "data/trajectories.jsonl"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Hyperparameters
N_SAMPLES_PER_INPUT = 100


def main():
    dataset = load_data(DATASET_FILE)

    create_prompt = create_simple_prompt
    prompts = [create_prompt(item["articles"], item["question"]) for item in dataset][
        :10
    ]
    asyncio.run(save_trajectories(prompts, dataset))


async def save_trajectories(prompts, dataset):
    # Open output file
    with open(TRAJECTORY_SAVE_PATH, "w", encoding="utf-8") as fout:
        generator = VllmEndpointGenerator(model=MODEL_NAME)
        print("Sending prompts to model...")
        outputs = generator.generate(
            prompts, n=N_SAMPLES_PER_INPUT, temperature=0.3, top_p=1.0
        )

        idx = 0
        async for outputs in tqdm_asyncio(outputs, total=len(prompts)):
            dataset_item = dataset[idx]
            prompt = prompts[idx]
            idx += 1

            samples = []
            for prediction, reasoning in outputs:
                reward = compute_reward(prediction, dataset_item["label"])
                samples.append(
                    {"reasoning": reasoning, "prediction": prediction, "reward": reward}
                )

            fout.write(json.dumps({"prompt": prompt, "samples": samples}) + "\n")

    print(f"Saved all trajectories to {TRAJECTORY_SAVE_PATH}")


if __name__ == "__main__":
    main()
