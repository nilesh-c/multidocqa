import os
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_civil_code, load_data, create_prompt, compute_reward

# Paths
CIVIL_CODE_FILE = (
    "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
)
DATASET_FILE = "data/processed/train.json"
TRAJECTORY_SAVE_PATH = "data/trajectories.jsonl"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Hyperparameters
N_SAMPLES_PER_INPUT = 4
MAX_SEQ_LEN = 2048


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    civil_code = load_civil_code(CIVIL_CODE_FILE)
    dataset = load_data(DATASET_FILE)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Open output file
    with open(TRAJECTORY_SAVE_PATH, "w", encoding="utf-8") as fout:

        for item in tqdm(dataset):
            prompt = create_prompt(item["articles"], item["question"])
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
            ).input_ids.to(device)

            samples = []
            for _ in range(N_SAMPLES_PER_INPUT):
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=5,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                    )
                decoded = tokenizer.decode(
                    output[0][input_ids.shape[-1] :], skip_special_tokens=True
                )
                reward = compute_reward(decoded, item["label"])

                samples.append({"output": decoded, "reward": reward})

            fout.write(json.dumps({"prompt": prompt, "samples": samples}) + "\n")

    print(f"Saved all trajectories to {TRAJECTORY_SAVE_PATH}")


if __name__ == "__main__":
    main()
