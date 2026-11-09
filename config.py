from pathlib import Path

# paths to data
DATASET_DIR = Path("Dataset")
BOOKS_DIR   = DATASET_DIR / "Books"
TRAIN_CSV   = DATASET_DIR / "combined_train_v2.csv"   
TEST_CSV    = DATASET_DIR / "test.csv"

# where the model gets saved after training
MODEL_SAVE_DIR  = Path("model")
PREDICTIONS_PATH = Path("predictions.csv")

# models — the classifier is trained from scratch; only the tokenizer vocab is pretrained
TOKENIZER_NAME        = "bert-base-uncased"
EMBEDDING_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# retrieval
TOP_K = 3  # how many passages to grab per claim

# chunking settings

CHUNK_SIZE    = 200   # words per passage (matches the cached index)
CHUNK_OVERLAP = 50   

# training stuff
BATCH_SIZE    = 16
NUM_EPOCHS    = 14
LEARNING_RATE = 2.5e-4  
WEIGHT_DECAY  = 0.10    
WARMUP_RATIO  = 0.10    
MAX_SEQ_LEN   = 256
SEED          = 42
PATIENCE      = 5

# leakage-free split: paraphrased copies of a claim never cross train/val/test
VAL_FRAC      = 0.15
TEST_FRAC     = 0.15

# label mapping — in the csv it's string, we convert to int
LABEL2ID = {"consistent": 1, "contradict": 0}
ID2LABEL = {0: "contradict", 1: "consistent"}

# ── from-scratch pretraining (book_tokenizer.py / pretrain_mlm.py) ──────────
TOKENIZER_DIR    = Path("tokenizer_books")      # WordPiece vocab learned from the novels
BOOK_VOCAB_SIZE  = 8000
PRETRAIN_DIR     = Path("model_pretrained")     # MLM-pretrained encoder
# encoder size used by both pretrain_mlm.py and train_scratch.py
MODEL_ARCH       = dict(embed_dim=256, num_heads=4, num_layers=4, ffn_dim=1024, dropout=0.1)

# ── training recipe ─────────────────────────────────────────────────────────
WITH_CHAR      = True    # prefix the `char` column to retrieval query + claim
PRETRAINED     = True    # book tokenizer + MLM-pretrained encoder (False = original recipe)
GROUP_SAMPLING = True    # one copy of each paraphrased claim per epoch
FINETUNE = (dict(lr=1e-4, finetune_dropout=0.1, focal_gamma=0.0,
                 label_smoothing=0.1, weight_decay=0.01, rdrop_alpha=0.0)
            if PRETRAINED else      # original from-random recipe
            dict(lr=2.5e-4, finetune_dropout=0.40, focal_gamma=2.0,
                 label_smoothing=0.15, weight_decay=0.10, rdrop_alpha=0.35))
USE_SYNTH      = True    # add synth_data.py pairs (name/number/negation swaps) to training
SYNTH_PATH     = "cache/synth_samples_final.pkl"
SYNTH_N        = 2400

# Sentence-level evidence: pack the most relevant sentences from the top
# passages into MAX_SEQ_LEN instead of truncating 3 x 200-word passages.
SENTENCE_EVIDENCE = True
