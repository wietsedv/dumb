from transformers import AutoModel
from transformers.utils.logging import set_verbosity_error

from constants import MODEL_IDS, MODEL_PRETTY

set_verbosity_error()

for model_name, model_id in MODEL_IDS.items():
    model = AutoModel.from_pretrained(model_id)
    # n = round(sum(p.numel() for p in model.parameters()) / 1_000_000)

    n_emb = sum(p.numel() for p in model.embeddings.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())

    n = sum(p.numel() for p in model.parameters())
    n_remainder = n - n_emb - n_enc

    # print(f"{model_name:>20} {round(n / 1_000_000):,}M {round(n_emb / 1_000_000):,}M {round(n_enc / 1_000_000):,}M {round(n_remainder / 1_000_000):,}M")

    print(f"{MODEL_PRETTY[model_name]} & {round(n / 1_000_000):,}M & {round(n_emb / n * 100)}%")