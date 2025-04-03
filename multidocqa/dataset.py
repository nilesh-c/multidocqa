from torch.utils.data import Dataset


# Dataset
class LegalDataset(Dataset):
    def __init__(self, data, civil_code, tokenizer, max_len=512):
        self.data = data
        self.civil_code = civil_code
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        label = 1 if item["label"] == "Y" else 0

        # Tokenize question
        question_tokens = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Pre-tokenize all articles in the civil code once
        if not hasattr(self, "pretokenized_articles"):
            self.pretokenized_articles = [
                self.tokenizer(
                    article["content"],
                    truncation=True,
                    max_length=self.max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                for article in self.civil_code
            ]

        return question_tokens, self.pretokenized_articles, label
