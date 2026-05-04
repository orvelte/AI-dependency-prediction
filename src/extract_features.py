"""Standalone feature extraction script.

Reads data/interim/wildchat_longitudinal.parquet in chunks, extracts linguistic
features per conversation, and saves data/processed/feature_matrix.parquet.

Run from the project root:
    python src/extract_features.py

Takes ~8-12 minutes on a standard laptop. No notebook timeout applies.
"""

import re
import sys
import time
import orjson
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

PARQUET_PATH = "data/interim/wildchat_longitudinal.parquet"
META_PATH    = "data/interim/conv_metadata.parquet"
OUT_PATH     = "data/processed/feature_matrix.parquet"
BATCH_SIZE   = 5000

# ── Word sets and phrase lists (faster than alternation regex) ─────────────────
FPP_SET          = {'we', 'us', 'our', 'ours', 'ourselves'}
HEDGING_SET      = {'maybe', 'perhaps', 'probably', 'possibly', 'might', 'could'}
HEDGING_PHRASES  = ['i think', 'i guess', 'i suppose', 'i believe']
SELF_DISC_SET    = {'honestly', 'personally', 'confess', 'admit'}
SELF_DISC_PHRASES= ['i feel', 'i felt', 'to be honest']
FILLER_SET       = {'um', 'uh', 'like', 'actually', 'just', 'okay', 'ok'}
FILLER_PHRASES   = ['you know']
GRATITUDE_SET    = {'thanks', 'appreciate', 'grateful', 'thx', 'cheers'}
GRATITUDE_PHRASES= ['thank you']
DEFERENCE_SET    = {'please', 'kindly'}
DEFERENCE_PHRASES= ['could you', 'would you', 'sorry to bother']
POSITIVE_W       = {'good', 'great', 'thanks', 'happy', 'love', 'excellent', 'wonderful', 'amazing'}
NEGATIVE_W       = {'bad', 'hate', 'terrible', 'awful', 'horrible', 'angry', 'sad', 'frustrated'}

REFUSAL_RE = re.compile(
    r"i(?:'m| am) (?:sorry|unable|not able)|i (?:can't|cannot)|"
    r"i don't think i (?:can|should)|i(?:'m| am) afraid|i must decline|"
    r"i(?:'m| am) not comfortable|i have to decline",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r'[.!?]+')


def _refusal_resistance(turns: list) -> float:
    for i in range(len(turns) - 2):
        if turns[i].get('role') == 'user' and turns[i + 1].get('role') == 'assistant':
            if REFUSAL_RE.search(turns[i + 1].get('content', '')):
                nxt = turns[i + 2] if i + 2 < len(turns) else None
                if nxt and nxt.get('role') == 'user' and len(nxt.get('content', '').split()) >= 4:
                    return 1.0
    return 0.0


def _sentiment_slope(user_texts: list[str]) -> float:
    if len(user_texts) < 2:
        return 0.0
    scores = []
    for text in user_texts:
        words = set(text.lower().split())
        pos, neg = len(words & POSITIVE_W), len(words & NEGATIVE_W)
        scores.append((pos - neg) / max(pos + neg, 1))
    x = np.arange(len(scores), dtype=float)
    return float(np.polyfit(x, scores, 1)[0])


def _extract_row(turns: list) -> dict:
    """All features for one conversation — single token-pass for word-set counts."""
    user_texts = [t.get('content', '') for t in turns if t.get('role') == 'user']
    n_user = max(len(user_texts), 1)
    combined = ' '.join(user_texts)
    tl = combined.lower()
    tokens = tl.split()
    n_tok = max(len(tokens), 1)

    # Single pass over tokens for all word-set counts
    c_fpp = c_hed = c_sd = c_fil = c_grat = c_def = 0
    for w in tokens:
        if w in FPP_SET:       c_fpp  += 1
        if w in HEDGING_SET:   c_hed  += 1
        if w in SELF_DISC_SET: c_sd   += 1
        if w in FILLER_SET:    c_fil  += 1
        if w in GRATITUDE_SET: c_grat += 1
        if w in DEFERENCE_SET: c_def  += 1

    for p in HEDGING_PHRASES:    c_hed  += tl.count(p)
    for p in SELF_DISC_PHRASES:  c_sd   += tl.count(p)
    for p in FILLER_PHRASES:     c_fil  += tl.count(p)
    for p in GRATITUDE_PHRASES:  c_grat += tl.count(p)
    for p in DEFERENCE_PHRASES:  c_def  += tl.count(p)

    n_questions = combined.count('?')
    n_sentences = max(len(SENTENCE_RE.split(combined)), 1)

    if len(user_texts) >= 2:
        scores = []
        for text in user_texts:
            words = set(text.lower().split())
            pos, neg = len(words & POSITIVE_W), len(words & NEGATIVE_W)
            scores.append((pos - neg) / max(pos + neg, 1))
        ss = float(np.polyfit(np.arange(len(scores), dtype=float), scores, 1)[0])
    else:
        ss = 0.0

    return {
        'fpp_rate':                    c_fpp  / n_tok,
        'self_disclosure_rate':        c_sd   / n_tok,
        'hedging_rate':                c_hed  / n_tok,
        'filler_rate':                 c_fil  / n_tok,
        'gratitude_rate':              (c_grat + c_def) / n_tok,
        'refusal_resistance_rate':     _refusal_resistance(turns),
        'question_to_statement_ratio': min(n_questions / n_sentences, 1.0),
        'mean_message_length':         n_tok / n_user,
        'type_token_ratio':            len(set(tokens)) / n_tok,
        'sentiment_slope':             ss,
    }


def process_batch(df_batch: pd.DataFrame, conv_meta_idx: pd.DataFrame, valid_conv_ids: set) -> pd.DataFrame:
    df = df_batch[df_batch['conversation_id'].isin(valid_conv_ids)]
    if df.empty:
        return pd.DataFrame()

    conv_ids = df['conversation_id'].tolist()
    raw_list = df['conversation'].tolist()
    turns_list = [orjson.loads(r) if isinstance(r, (str, bytes)) else r for r in raw_list]
    feat_rows = [_extract_row(turns) for turns in turns_list]

    meta = conv_meta_idx.loc[conv_ids]
    result = pd.DataFrame(feat_rows)
    result.insert(0, 'hashed_ip',                 meta['hashed_ip'].values)
    result.insert(1, 'conversation_id',            conv_ids)
    result.insert(2, 'session_number',             meta['session_number'].values)
    result.insert(3, 'n_user_turns',               meta['n_user_turns'].values)
    result.insert(4, 'return_interval_hours',      meta['return_interval_hours'].values)
    result.insert(5, 'log_n_user_turns',           meta['log_n_user_turns'].values)
    result.insert(6, 'log_return_interval_hours',  meta['log_return_interval_hours'].values)
    return result


def main():
    t_start = time.time()
    print(f"Loading metadata from {META_PATH} ...")
    conv_meta = pd.read_parquet(META_PATH)
    conv_meta['conversation_id'] = conv_meta['conversation_id'].astype(str)
    conv_meta_idx = conv_meta.set_index('conversation_id')
    valid_conv_ids = set(conv_meta_idx.index)
    print(f"  {len(valid_conv_ids):,} valid conversations")

    pf = pq.ParquetFile(PARQUET_PATH)
    n_rows = pf.metadata.num_rows
    n_batches = -(-n_rows // BATCH_SIZE)

    batch_dfs = []
    for batch in tqdm(pf.iter_batches(batch_size=BATCH_SIZE), total=n_batches, desc="Extracting"):
        df_batch = batch.to_pandas()
        df_batch['conversation_id'] = df_batch['conversation_id'].astype(str)
        result = process_batch(df_batch, conv_meta_idx, valid_conv_ids)
        if not result.empty:
            batch_dfs.append(result)

    feature_matrix = pd.concat(batch_dfs, ignore_index=True)
    feature_matrix.to_parquet(OUT_PATH, index=False)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved {len(feature_matrix):,} rows × {feature_matrix.shape[1]} columns → {OUT_PATH}")


if __name__ == "__main__":
    main()
