import json
import asyncio
from openai import AsyncOpenAI
import os
from tqdm import tqdm
from typing import List, Dict
from multidocqa.llm_client import VllmEndpointGenerator

# Script for zero-shot or few-shot in-context learning (ICL) evaluation of a legal reasoning model using OpenAI API

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Initialize AsyncOpenAI client
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# File paths
CIVIL_CODE_FILE = (
    "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
)
DATASET_FILE = "data/processed/train.json"


# Load Civil Code Articles
def load_civil_code(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load train/eval dataset
def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:100]


# Prepare Prompt
def create_prompt(civil_code: List[Dict], question: str) -> str:
    code_str = "\n".join(
        [f"Article {art['number']}: {art['content']}" for art in civil_code]
    )
    prompt = f"""
You are a legal reasoning AI. Given a list of Civil Code articles and a legal question or statement, your task is to:
1. Retrieve the articles most relevant to the question or statement.
2. Based on the retrieved articles, determine whether the articles entail the statement as true (Y) or not (N).
3. Repeatedly check and refine your reasoning and conclusions until you reach a final conclusion.

Respond strictly with 'Y' or 'N'.

Civil Code:
{code_str}

Respond to the following question or statement strictly with 'Y' for yes or 'N' for no. Do not use any other words or phrases. Only Y or N is allowed.

Statement: {question}
Answer:
"""
    return prompt


# Evaluate on Dataset
async def evaluate(civil_code: List[Dict], dataset: List[Dict]) -> None:
    generator = VllmEndpointGenerator(model=MODEL_NAME)
    correct = 0
    total = len(dataset)

    prompts = [create_prompt(civil_code, item["question"]) for item in dataset]

    print("Sending prompts to model...")
    outputs = await generator.generate(prompts)

    correct = 0
    for item, output in tqdm(zip(dataset, outputs)):
        gold = item["label"]
        prediction, reasoning = output
        if prediction == gold:
            correct += 1
        else:
            print(
                f"\nIncorrect prediction for ID {item['id']}\nQuestion: {item['question']}\nPrediction: {prediction}, Gold: {gold}\n"
            )
            print(reasoning)

    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    print("Loading data...")
    civil_code = load_civil_code(CIVIL_CODE_FILE)
    dataset = load_data(DATASET_FILE)
    print("Evaluating model...")
    asyncio.run(evaluate(civil_code, dataset))
