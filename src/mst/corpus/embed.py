"""Search vectors of the corpus: multilingual-e5-small on the CPU through ONNX Runtime.

The model is read from the Hugging Face cache and downloaded there on first use (about 0.5 GB).
A vector is the mean of the token states over the attention mask, normalized to length one, so
the dot product of two vectors is their cosine. e5 expects ``passage: `` before an indexed text
and ``query: `` before a query.
"""

import functools

import numpy as np

MODEL = "intfloat/multilingual-e5-small"
BATCH = 16
MAX_TOKENS = 512


@functools.cache
def _model():
    import onnxruntime
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    def cached(name: str) -> str:
        try:
            return hf_hub_download(MODEL, name, local_files_only=True)
        except Exception:
            return hf_hub_download(MODEL, name)

    tokenizer = Tokenizer.from_file(cached("tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>") or 0, pad_token="<pad>")
    session = onnxruntime.InferenceSession(
        cached("onnx/model.onnx"), providers=["CPUExecutionProvider"]
    )
    return tokenizer, session


def embed(texts: list[str], prefix: str) -> np.ndarray:
    """Unit vectors of ``prefix + text``, one row per text, float32."""
    tokenizer, session = _model()
    inputs = {item.name for item in session.get_inputs()}
    out: np.ndarray | None = None
    # Texts of close length go in one batch, so less padding is computed.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), BATCH):
        batch = order[start : start + BATCH]
        encoded = tokenizer.encode_batch([prefix + texts[i] for i in batch])
        mask = np.array([item.attention_mask for item in encoded], dtype="int64")
        feed = {"input_ids": np.array([item.ids for item in encoded], dtype="int64")}
        feed["attention_mask"] = mask
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(mask)
        hidden = session.run(None, feed)[0]
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        if out is None:
            out = np.zeros((len(texts), pooled.shape[1]), dtype="float32")
        out[batch] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    return out if out is not None else np.zeros((0, 0), dtype="float32")
