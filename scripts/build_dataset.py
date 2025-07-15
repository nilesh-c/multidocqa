#!/usr/bin/env python3
"""
Script to build a comprehensive dataset using COLIEE civil code articles and XML entailment data.
Creates a JSON file with 'articles' and 'entailment_pairs' keys.
"""

import json
import argparse
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import multidocqa
sys.path.insert(0, str(Path(__file__).parent.parent))

from multidocqa.utils import parse_civil_code, parse_coliee_xml_folder


def build_dataset(civil_code_path: str, coliee_xml_folder: str, output_path: str):
    """
    Args:
        civil_code_path (str): Path to the civil code text file
        coliee_xml_folder (str): Path to folder containing COLIEE XML files
        output_path (str): Path to save the output JSON file
    """
    print(f"Parsing COLIEE civil code from: {civil_code_path}")
    articles = parse_civil_code(civil_code_path)

    print(f"Parsing COLIEE XML files from: {coliee_xml_folder}")
    entailment_pairs = parse_coliee_xml_folder(coliee_xml_folder)

    # Build the final dataset structure
    dataset = {"articles": articles, "entailment_pairs": entailment_pairs}

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset built successfully!")
    print(f"- {len(articles)} civil code articles")
    print(f"- {len(entailment_pairs)} entailment pairs")
    print(f"- Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset from civil code and COLIEE XML files"
    )
    parser.add_argument(
        "--civil-code", required=True, help="Path to civil code text file"
    )
    parser.add_argument(
        "--coliee-xml", required=True, help="Path to folder containing COLIEE XML files"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.civil_code):
        print(f"Error: Civil code file not found: {args.civil_code}")
        sys.exit(1)

    if not os.path.exists(args.coliee_xml):
        print(f"Error: COLIEE XML folder not found: {args.coliee_xml}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    build_dataset(args.civil_code, args.coliee_xml, args.output)


if __name__ == "__main__":
    main()
