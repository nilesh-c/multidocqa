import asyncio
from typing import Dict, List, Literal

from openai import AsyncOpenAI
from pydantic import ValidationError
from tqdm.asyncio import tqdm_asyncio

from multidocqa.llm_client import VllmEndpointGenerator
from multidocqa.utils import ReasoningOutput, load_civil_code, load_data

# Script for zero-shot or few-shot in-context learning (ICL) evaluation
# of a legal reasoning model using OpenAI API

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Initialize AsyncOpenAI client
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# File paths
CIVIL_CODE_FILE = (
    "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
)
DATASET_FILE = "data/processed/train.json"


# Prepare Prompt
def create_simple_prompt(civil_code: List[Dict], question: str) -> str:
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
"""  # noqa: E501
    return prompt


# Prepare Prompt with JSON Output
def create_json_prompt(civil_code: List[Dict], question: str) -> str:
    code_str = "\n".join(
        [f"Article {art['number']}: {art['content']}" for art in civil_code]
    )
    prompt = f"""
You are a legal reasoning AI. Given a list of Civil Code articles and a legal question or statement, your task is to:
1. Retrieve a maximum of 5 articles most relevant to the question or statement.
2. Based on the retrieved articles, determine whether the articles entail the statement as true (Y) or not (N).
3. Repeatedly check and refine your reasoning and conclusions until you reach a final conclusion.

Respond strictly with 'Y' or 'N'.

Civil Code:
{code_str}

After thinking and reasoning, respond to the following question or statement with valid JSON (schema given below) containing and answer ('Y' for yes or 'N' for no)
along with a list of relevant article numbers. Choose a maximum of 5 articles.

Statement: {question}
JSON Schema: {ReasoningOutput.model_json_schema()}
Answer:
"""  # noqa: E501
    return prompt


async def evaluate(
    civil_code: List[Dict], dataset: List[Dict], prompt_type: Literal["json", "simple"]
) -> None:
    generator = VllmEndpointGenerator(model=MODEL_NAME)
    correct = 0
    total = len(dataset)

    create_prompt = (
        create_json_prompt if prompt_type == "json" else create_simple_prompt
    )

    prompts = [create_prompt(civil_code, item["question"]) for item in dataset]

    print("Sending prompts to model...")
    outputs = generator.generate(
        prompts,
        guided_json=(
            ReasoningOutput.model_json_schema() if prompt_type == "json" else None
        ),
    )

    idx = 0
    async for output in tqdm_asyncio(outputs, total=total):
        item = dataset[idx]
        idx += 1

        prediction, reasoning = output
        if prompt_type == "json":
            try:
                prediction = ReasoningOutput.model_validate_json(prediction)
                print("Gold label:", item["label"])
                print(
                    "Gold relevant articles:",
                    [a["number"] for a in item["articles"]],
                )
                print(prediction.model_dump_json(indent=2))
                predicted_label = prediction.answer
            except ValidationError:
                print(f"Invalid JSON response: {prediction}")
                predicted_label = None
        else:
            predicted_label = prediction.strip()

        gold_label = item["label"]
        print(f"Prediction: {predicted_label}, Gold: {gold_label}")

        if predicted_label == gold_label:
            correct += 1
        else:
            print(f"\nIncorrect prediction for ID {item['id']}")
            print(f"Question: {item['question']}")
            print(f"Prediction: {predicted_label}, Gold: {gold_label}\n")
            print(reasoning)

    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    print("Loading data...")
    civil_code = load_civil_code(CIVIL_CODE_FILE)
    dataset = load_data(DATASET_FILE)
    print("Evaluating model...")
    asyncio.run(evaluate(civil_code, dataset, prompt_type="json"))
