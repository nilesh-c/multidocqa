import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict
from multidocqa.utils import load_data, create_simple_prompt
from multidocqa.llm_client import VllmEndpointGenerator

# Script for zero-shot or few-shot in-context learning (ICL) evaluation of a legal reasoning model using OpenAI API

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Initialize AsyncOpenAI client
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# File paths
CIVIL_CODE_FILE = (
    "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
)
DATASET_FILE = "data/processed/train.json"


async def evaluate(dataset: List[Dict]) -> None:
    generator = VllmEndpointGenerator(model=MODEL_NAME)
    correct = 0
    total = len(dataset)

    create_prompt = create_simple_prompt

    prompts = [create_prompt(item["articles"], item["question"]) for item in dataset]

    print("Sending prompts to model...")
    outputs = generator.generate(prompts)

    idx = 0
    async for output in tqdm_asyncio(outputs, total=total):
        item = dataset[idx]
        idx += 1

        prediction, reasoning = output
        predicted_label = prediction.strip()

        gold_label = item["label"]
        print(f"Prediction: {predicted_label}, Gold: {gold_label}")

        if predicted_label == gold_label:
            correct += 1
        else:
            print(
                f"\nIncorrect prediction for ID {item['id']}\nQuestion: {item['question']}\nPrediction: {predicted_label}, Gold: {gold_label}\n"
            )
            print(reasoning)

    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    print("Loading data...")
    dataset = load_data(DATASET_FILE)
    print("Evaluating model...")
    asyncio.run(evaluate(dataset))
