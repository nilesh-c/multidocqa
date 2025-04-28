import os
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Literal
from pydantic import BaseModel, Field


def extract_articles(text):
    extracted_articles = []
    lines = text.split("\n")
    current_article = None
    for line in lines:
        line = line.strip()
        if line.startswith("Article"):
            if current_article:
                extracted_articles.append(current_article)
            parts = line.split(" ", 2)
            article_number = parts[1]
            article_content = parts[2] if len(parts) > 2 else ""
            current_article = {"number": article_number, "content": article_content}
        elif current_article and line:
            current_article["content"] += " " + line

    if current_article:
        extracted_articles.append(current_article)

    return extracted_articles


def parse_civil_code(file_path, output_json_path=None):
    """
    Parses a civil code text file and extracts articles, saving them to a JSON file.

    Args:
        file_path (str): Path to the civil code text file.
        output_json_path (str): Path to save the extracted articles as a JSON file.

    Returns:
        list: A list of extracted articles.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        articles = extract_articles(text)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        print(f"Extracted {len(articles)} articles, saved to {output_json_path}")

    return articles


def parse_single_coliee_xml(xml_file_path):
    """
    Parses a single XML file and extracts question-article pairs and articles.

    Args:
        xml_file_path (str): Path to the XML file.

    Returns:
        tuple: A tuple containing a list of question-article pairs and a list of articles.
    """
    dataset = []
    articles_data = []

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for pair in root.findall("pair"):
        pair_id = pair.get("id")
        label = pair.get("label")
        question = pair.find("t2").text.strip()
        articles_text = pair.find("t1").text.strip()
        structured_articles = extract_articles(articles_text)
        dataset.append(
            {
                "id": pair_id,
                "question": question,
                "articles": structured_articles,
                "label": label,
            }
        )
        articles_data.extend(structured_articles)

    return dataset, articles_data


def parse_coliee_xml_folder(folder_path, output_json_path=None):
    """
    Parses all XML files in a folder and creates a combined dataset.

    Args:
        folder_path (str): Path to the folder containing XML files.
        output_json (str): Path to save the combined dataset as a JSON file.

    Returns:
        list: A list of question-article pairs
    """
    combined_dataset = []
    combined_articles_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            dataset, articles_data = parse_single_coliee_xml(file_path)
            combined_dataset.extend(dataset)
            combined_articles_data.extend(articles_data)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_dataset, f, indent=4, ensure_ascii=False)

        print(
            f"Parsed {len(combined_dataset)} question-article pairs and {len(combined_articles_data)} articles, saved to {output_json_path}"
        )

    return combined_dataset


# Load Civil Code Articles
def load_civil_code(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load train/eval dataset
def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)[:50]


class ReasoningOutput(BaseModel):
    answer: Literal["Y", "N"] = Field(
        description="Whether the statement is true or not"
    )
    relevant_articles: List[int] = Field(description="List of relevant article numbers")


# Prepare Prompt
def create_simple_prompt(articles: List[Dict], question: str) -> str:
    code_str = "\n".join(
        [f"Article {art['number']}: {art['content']}" for art in articles]
    )

    prompt = f"""
You are a legal reasoning AI. Given a list of Civil Code articles and a legal question or statement,
your task is to determine whether the articles entail the statement as true (Y) or not (N).
Repeatedly check and refine your reasoning and conclusions until you reach a final conclusion.

Respond strictly with 'Y' or 'N'.

Civil Code articles:
{code_str}

Respond to the following question or statement strictly with 'Y' for yes or 'N' for no. Do not use any other words or phrases. Only Y or N is allowed.

Statement: {question}
Answer:
"""
    return prompt
