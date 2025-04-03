import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class LegalEntailmentModel(nn.Module):
    def __init__(self, encoder, hidden_size=768, use_gumbel=False, temperature=1.0):
        super(LegalEntailmentModel, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size * 2, 2)
        self.use_gumbel = use_gumbel
        self.temperature = temperature

    def forward(self, question_tokens, article_tokens, inference=False):
        # Encode question
        question_embedding = self.encoder(
            **question_tokens
        ).last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Encode all articles in a batch
        batch_size, num_articles = len(article_tokens), len(article_tokens[0])
        flat_article_tokens = {
            key: torch.cat([tokens[key] for tokens in article_tokens], dim=0)
            for key in article_tokens[0]
        }
        article_embeddings = self.encoder(**flat_article_tokens).last_hidden_state
        article_embeddings = article_embeddings.view(
            batch_size, num_articles, -1, self.hidden_size
        )  # (batch_size, num_articles, seq_len, hidden_size)

        # Flatten articles for cross-attention
        article_embeddings_flat = article_embeddings.view(
            batch_size, -1, self.hidden_size
        )  # (batch_size, num_articles * seq_len, hidden_size)

        # Cross-attention between question and articles
        question_context, _ = self.cross_attention(
            query=question_embedding,
            key=article_embeddings_flat,
            value=article_embeddings_flat,
        )  # (batch_size, seq_len, hidden_size)

        # Pool the question context (e.g., take the CLS token representation)
        pooled_question_context = question_context[:, 0, :]  # (batch_size, hidden_size)

        combined_representation = torch.cat(
            [question_embedding, pooled_question_context], dim=-1
        )  # (batch_size, hidden_size * 2)
        # Classification
        logits = self.classifier(combined_representation)
        return logits
