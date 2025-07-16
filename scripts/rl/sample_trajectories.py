import asyncio
import json
from typing import Any, List, Tuple

import numpy as np
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from multidocqa.llm_client import VllmEndpointGenerator
from multidocqa.utils import compute_reward, create_simple_prompt, load_data

# Paths
CIVIL_CODE_FILE = (
    "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
)
DATASET_FILE = "data/processed/train.json"
TRAJECTORY_SAVE_PATH = "data/trajectories.jsonl"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Hyperparameters
N_SAMPLES_PER_INPUT = 500


class DataPoint(BaseModel):
    idx: int
    prompt: str
    label: str


async def process_outputs(
    outputs: Any, data_point: DataPoint
) -> Tuple[DataPoint, List[Any]]:
    async for output in outputs:
        samples = []
        for prediction, reasoning in output:
            reward = compute_reward(prediction, data_point.label)
            samples.append(
                {"reasoning": reasoning, "prediction": prediction, "reward": reward}
            )

    return data_point, samples


async def process_chunk(
    generator: VllmEndpointGenerator, chunk: List[DataPoint]
) -> List[Any]:
    """Process a single chunk of prompts."""
    outputs = []
    for seed in range(0, int(N_SAMPLES_PER_INPUT / 10)):
        for prompt in chunk:
            output = generator.generate(
                [prompt.prompt],
                n=int(N_SAMPLES_PER_INPUT / 10),
                temperature=0.3,
                top_p=1.0,
                seed=seed,
                timeout=1000,
            )
            outputs.append((prompt, output))

    tasks = [process_outputs(output, prompt) for prompt, output in outputs]

    # Wait for all outputs to complete
    results = []
    for coroutine in tqdm_asyncio.as_completed(tasks):
        try:
            result = await coroutine
            results.append(result)
        except Exception as e:
            print("Got an exception:", e)

    grouped_results = dict()
    for data_point, output_samples in results:
        if data_point.idx not in grouped_results:
            grouped_results[data_point.idx] = {
                "idx": data_point.idx,
                "prompt": data_point.prompt,
                "label": data_point.label,
                "samples": [],
            }
        grouped_results[data_point.idx]["samples"].extend(output_samples)

    return list(grouped_results.values())


async def save_trajectories(prompts: List[DataPoint]) -> None:
    # Open output file
    with open(TRAJECTORY_SAVE_PATH, "w", encoding="utf-8") as fout:
        generators = [
            VllmEndpointGenerator(
                model=MODEL_NAME,
                openai_endpoint_url=f"http://{host}:8000/v1",
                max_concurrent_requests=30,
            )
            for host in ["ecdpl01", "ecdpl02", "ecdpl03"]
        ]

        print("Sending prompts to model...")
        # Create tasks for processing chunks concurrently
        tasks = []
        chunks = np.array_split(np.array(prompts), 3)
        for idx, chunk in enumerate(chunks):
            tasks.append(process_chunk(generators[idx % 3], chunk.tolist()))

        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks)

        # Write results to file
        for results in results_list:
            for result in results:
                fout.write(json.dumps(result) + "\n")

    print(f"Saved all trajectories to {TRAJECTORY_SAVE_PATH}")


def main() -> None:
    dataset = load_data(DATASET_FILE)

    create_prompt = create_simple_prompt
    prompts = [
        DataPoint(
            idx=idx,
            prompt=create_prompt(item["articles"], item["question"]),
            label=item["label"],
        )
        for idx, item in enumerate(dataset)
    ]
    asyncio.run(save_trajectories(prompts))


if __name__ == "__main__":
    main()
