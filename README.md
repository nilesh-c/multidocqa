# multidocqa

Researchy work-in-progress code for an LLM-based system that, given a legal statement or question, it will retrieve relevant paragraphs from a list of civil code articles, and predict if the set of relevant articles entails the legal statement positively or negatively.

Currently focusing on the [COLIEE dataset](https://coliee.org/overview) (tasks 3 and 4)

Here is an example of a training sample, where each article is taken from a set of :

```json
{
    "id": "H27-3-U",
    "question": "In the case where B, who was granted authority of agency to buy a land as an agent of A, concluded a contract for sale of a land \"X\" with C representing that the same is made on behalf of A by the fraud of C to B, A may not rescind the contract for sale.",
    "articles": [
        {
            "number": "101",
            "content": " (1) If the validity of a manifestation of intention that an agent has made to the other party is to be influenced by the absence of intention; by mistake, fraud, or duress; or by the knowledge of or negligence in not knowing of a particular circumstance; whether or not any such fact was present is decided as it concerns the agent. (2) If the validity of a manifestation of intention that the other party has made to the agent is to be influenced by the recipient's knowledge of or negligence in not knowing of a particular circumstance, whether or not any such fact was present is decided as it concerns the agent. (3) If an agent who has been entrusted with performing a specific juridical act performs that act, the principal may not assert that the agent did not know of any particular circumstance of which the principal knew. The same applies to any circumstance of which the principal did not know due to the principal's own negligence."
        },
        {
            "number": "96",
            "content": " (1) A manifestation of intention based on fraud or duress is voidable. (2) If a third party commits a fraud inducing a first party to make a manifestation of intention to a second party, that manifestation of intention is voidable only if the second party knew or could have known that fact. (3) The rescission of a manifestation of intention induced by fraud under the provisions of the preceding two paragraphs may not be duly asserted against a third party in good faith acting without negligence."
        }
    ],
    "label": "N"
}
```

## Tasks

We can try to frame three machine learning tasks with this dataset, depending upon what part of the data is used for supervision:
1. Given a list of 782 civil code articles, given a question/statement, retrieve the relevant articles.
2. Given a question/statement and a set of relevant articles, predict whether the articles positively or negatively entail the statement.
3. Solve 1 & 2 end-to-end: given a list of 782 civil code articles, retrieve the relevant articles and predict the entailment. Note that we do not assume there is supervision for the relevant articles here.

## Approaches to compare for task 3

1. Baseline 1: do we need RAG if we have long-context LLMs? Craft a prompt containing the entire civil code along with the statement, output should be Y/N.
2. Baseline 2: RAG, zero-shot, without any fine-tuning
3. RAG, end-to-end trained with reinforcement learning (how can we optimize the pareto front of computation vs performance?)
4. During the entailment stage - use a reasoning model, sample K reasoning outputs, compute rewards based on entailment labels, use these to train the model using RL.