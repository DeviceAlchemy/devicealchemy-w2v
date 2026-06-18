"""
Scientific Word2Vec Model Trainer v1.0
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""

import os
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec, phrases
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Paths
CORPUS_FILE  = Path("data/dataset.txt")
PHRASES_FILE = Path("data/dataset_phrases.txt")
MODEL_DIR    = Path("models")

# Hyperparameters
WORD2VEC_CONFIG = dict(
    vector_size  = 200,      # Size of the vector
    window       = 7,        # Window to understand the context
    min_count    = 5,        # Min. occurance threshold
    sg           = 1,        # skip-gram — better for smaller datasets
    hs           = 0,        # negative sampling (faster + better quality)
    negative     = 10,       # sweet spot for medium corpus size
    epochs       = 15,       # extra passes compensate for smaller data
    workers      = os.cpu_count(),
    sample       = 1e-4,     # downsampling rate
    seed         = 42,
)

# Phrase detection (bigram + trigram)
PHRASES_CONFIG = dict(
    min_count   = 8,     
    threshold   = 10,      
    delimiter   = "_",
)

# Early stopping criteria
EARLY_STOP_REL_THRESHOLD = 0.005  
EARLY_STOP_PATIENCE      = 2      


class LossLogger(CallbackAny2Vec):
    """Log per-epoch loss and enforce early stopping."""

    def __init__(self):
        self.epoch          = 0
        self.prev_loss      = None
        self.stagnant_count = 0
        self.t_start        = None

    def on_epoch_begin(self, model):
        self.t_start = time.time()

    def on_epoch_end(self, model):
        loss    = model.get_latest_training_loss()
        elapsed = time.time() - self.t_start

        if self.prev_loss is not None:
            delta    = loss - self.prev_loss
            rel      = abs(delta) / max(abs(self.prev_loss), 1e-9)
            delta_str = f"  delta={delta:+.0f}  rel={rel:.4f}"

            if rel < EARLY_STOP_REL_THRESHOLD:
                self.stagnant_count += 1
                log.info(
                    f"Epoch {self.epoch + 1:02d} — loss={loss:.0f}{delta_str}  "
                    f"({elapsed:.0f}s)  [stagnant {self.stagnant_count}/{EARLY_STOP_PATIENCE}]"
                )
                if self.stagnant_count >= EARLY_STOP_PATIENCE:
                    log.info("Early stopping: loss improvement below 0.5% threshold.")
            else:
                self.stagnant_count = 0
                log.info(
                    f"Epoch {self.epoch + 1:02d} — loss={loss:.0f}{delta_str}  "
                    f"({elapsed:.0f}s)"
                )
        else:
            log.info(
                f"Epoch {self.epoch + 1:02d} — loss={loss:.0f}  ({elapsed:.0f}s)"
            )

        self.prev_loss = loss
        self.epoch    += 1


def build_phrases(corpus_path: Path, out_path: Path) -> None:
    log.info("Building bigram phrase model ...")
    sentences    = LineSentence(str(corpus_path))
    bigram_model = phrases.Phrases(sentences, **PHRASES_CONFIG)
    bigram       = phrases.Phraser(bigram_model)

    log.info("Building trigram phrase model ...")
    trigram_model = phrases.Phrases(bigram[sentences], **PHRASES_CONFIG)
    trigram       = phrases.Phraser(trigram_model)

    log.info(f"Writing phrase-merged corpus to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sent in trigram[bigram[sentences]]:
            f.write(" ".join(sent) + "\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bigram_model.save(str(MODEL_DIR / "bigram_phrases.pkl"))
    trigram_model.save(str(MODEL_DIR / "trigram_phrases.pkl"))
    log.info("Phrase models saved.")


def train(use_phrases: bool = True) -> Word2Vec:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = CORPUS_FILE
    if use_phrases:
        if not PHRASES_FILE.exists():
            build_phrases(CORPUS_FILE, PHRASES_FILE)
        corpus_path = PHRASES_FILE

    log.info(f"Training Word2Vec on {corpus_path}")
    log.info(f"Config: {WORD2VEC_CONFIG}")
    log.info(f"CPUs available: {os.cpu_count()}")

    loss_logger = LossLogger()

    model = Word2Vec(
        corpus_file  = str(corpus_path),
        compute_loss = True,
        callbacks    = [loss_logger],
        **WORD2VEC_CONFIG,
    )

    model_path = MODEL_DIR / "materials_w2v.model"
    vec_path   = MODEL_DIR / "materials_w2v.kv"

    model.save(str(model_path))
    model.wv.save(str(vec_path))

    log.info(f"Vocabulary size : {len(model.wv):,} tokens")
    log.info(f"Model saved     : {model_path}")
    log.info(f"Vectors saved   : {vec_path}")

    return model


def sanity_check(model: Word2Vec) -> None:
    wv = model.wv
    test_terms = [
        # Battery / solid-state
        "LiFePO4", "cathode", "electrolyte", "solid_state_electrolyte",
        # Semiconductor / thin-film
        "perovskite", "bandgap", "hafnium_oxide",
        # Spintronics (relevant to your domain)
        "CoFeB", "spin_torque", "TMR",
        # Thermoelectrics
        "thermoelectric", "Seebeck",
    ]

    log.info("\n── Sanity check: most similar terms ──")
    found = 0
    for term in test_terms:
        t = term.lower()
        if t in wv:
            similar = wv.most_similar(t, topn=5)
            pairs   = ", ".join(f"{w} ({s:.3f})" for w, s in similar)
            log.info(f"  {term:35s} -> {pairs}")
            found += 1
        else:
            log.info(f"  {term:35s} -> NOT IN VOCABULARY")

    coverage = found / len(test_terms) * 100
    log.info(f"\n  Vocabulary coverage of test terms: {found}/{len(test_terms)} ({coverage:.0f}%)")
    if coverage < 60:
        log.warning(
            "  Low coverage — check min_count setting or corpus preprocessing. "
            "Rare spintronic terms (TMR, CoFeB) may need min_count ≤ 3."
        )


if __name__ == "__main__":
    model = train(use_phrases=True)
    sanity_check(model)
