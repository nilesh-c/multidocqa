from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# Paths
TRAJECTORY_FILE = "data/trajectories.jsonl"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def reward_fn(completions, references, **kwargs):
    rewards = []
    for pred, label in zip(completions, references):
        pred = pred.strip().upper()
        label = label.strip().upper()
        rewards.append(1.0 if pred == label else 0.0)
    return rewards


def preprocess_data(trajectory_path):
    import json
    from datasets import Dataset

    data = []
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            traj = json.loads(line)
            prompt = traj["prompt"]
            for sample in traj["samples"]:
                data.append(
                    {
                        "prompt": prompt,
                        "completion": sample["output"],
                        "reference": None,  # We will use label indirectly, could embed it later if needed
                    }
                )
    return Dataset.from_list(data)


def main():
    dataset = preprocess_data(TRAJECTORY_FILE)

    training_args = GRPOConfig(
        model_name_or_path=MODEL_NAME,
        output_dir="./checkpoints/grpo/",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        max_steps=1000,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=training_args,
        reward_funcs=lambda completions, **kwargs: reward_fn(
            completions, kwargs["references"]
        ),
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
