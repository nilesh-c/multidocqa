import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from multidocqa.dataset import LegalDataset
from multidocqa.model import LegalEntailmentModel


# Load data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Training
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for question_tokens, article_tokens, labels in dataloader:
        question_tokens = {
            key: val.squeeze(1).to(device) for key, val in question_tokens.items()
        }
        article_tokens = [
            {key: val.squeeze(1).to(device) for key, val in tokens.items()}
            for tokens in article_tokens
        ]
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(question_tokens, article_tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Batch Loss: {loss.item():.4f}")
    print(f"Epoch Loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)


# Evaluation
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for question_tokens, article_tokens, labels in dataloader:
            question_tokens = {
                key: val.squeeze(1).to(device) for key, val in question_tokens.items()
            }
            article_tokens = [
                {key: val.squeeze(1).to(device) for key, val in tokens.items()}
                for tokens in article_tokens
            ]
            labels = labels.to(device)

            logits, _ = model(question_tokens, article_tokens)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


# Main
def main():
    # Hyperparameters
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 3
    max_len = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_model_name = "nlpaueb/legal-bert-base-uncased"
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)

    # Dataset and DataLoader
    civil_code = load_json(
        "data/coliee2025/COLIEE2025statute_data-English/text/civil_code_en.json"
    )
    dataset = load_json("data/processed/train.json")

    # Split train data into train/test
    random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train_split = dataset[:split_idx]
    test_split = dataset[split_idx:]

    train_dataset = LegalDataset(train_split, civil_code, tokenizer, max_len)
    test_dataset = LegalDataset(test_split, civil_code, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss
    encoder = AutoModel.from_pretrained(encoder_model_name)
    model = LegalEntailmentModel(encoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Evaluation
    test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save model
    torch.save(model.state_dict(), "legal_entailment_model.pth")


if __name__ == "__main__":
    main()
