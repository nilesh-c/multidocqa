from typing import Any, Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm
from xformers.ops import memory_efficient_attention


class LegalEntailmentModel(nn.Module):
    def __init__(
        self,
        encoder: Any,
        hidden_size: int = 768,
        use_gumbel: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super(LegalEntailmentModel, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 2)
        self.use_gumbel = use_gumbel
        self.temperature = temperature

        # Add learnable projection matrices for attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        question_tokens: Dict[str, Any],
        article_tokens: List[Dict[str, Any]],
        inference: bool = False,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        with torch.no_grad():
            question_embedding = self.encoder(**question_tokens).last_hidden_state

        batch_size, seq_len_q, hidden_size = question_embedding.shape
        num_articles = len(article_tokens)

        aggregated_context_sum = torch.zeros(
            batch_size, seq_len_q, hidden_size, device=question_embedding.device
        )
        total_articles_processed = 0

        for i in tqdm(range(0, num_articles, chunk_size)):
            chunk = article_tokens[i : i + chunk_size]

            with torch.no_grad():
                article_embeddings_list = [
                    self.encoder(**inputs).last_hidden_state for inputs in chunk
                ]
                article_embeddings = nn.utils.rnn.pad_sequence(
                    article_embeddings_list, batch_first=True
                )

            chunk_size_actual = len(chunk)
            max_seq_len_a = article_embeddings.shape[2]

            article_embeddings = article_embeddings.view(
                batch_size * chunk_size_actual, max_seq_len_a, hidden_size
            ).contiguous()

            question_embedding_expanded = (
                question_embedding.unsqueeze(1)
                .expand(-1, chunk_size_actual, -1, -1)
                .reshape(batch_size * chunk_size_actual, seq_len_q, hidden_size)
                .contiguous()
            )

            # Apply learnable projections
            query = self.q_proj(question_embedding_expanded)
            key = self.k_proj(article_embeddings)
            value = self.v_proj(article_embeddings)

            attended_chunk = memory_efficient_attention(
                query=query,
                key=key,
                value=value,
            )

            attended_chunk = attended_chunk.view(
                batch_size, chunk_size_actual, seq_len_q, hidden_size
            )
            aggregated_context_sum += attended_chunk.sum(dim=1)
            total_articles_processed += chunk_size_actual

            del article_embeddings, question_embedding_expanded, attended_chunk
            torch.cuda.empty_cache()

        aggregated_context = aggregated_context_sum / total_articles_processed
        pooled_question_context = aggregated_context[:, 0, :]

        combined_representation = torch.cat(
            [
                question_embedding[:, 0, :].view(batch_size, hidden_size),
                pooled_question_context,
            ],
            dim=-1,
        )

        logits = self.classifier(combined_representation)
        return logits
