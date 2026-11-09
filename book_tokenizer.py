"""
book_tokenizer.py — train a WordPiece vocabulary on the novels themselves.

The original pipeline borrowed bert-base-uncased's 30,522-token vocabulary.
With a 256-d model that is a 7.8M-parameter embedding table, most of which
the tiny labelled set never touches. A vocabulary learned from the books is
~4x smaller, covers the character and place names as whole tokens, and keeps
the project free of anything pretrained on the classifier side.

    python book_tokenizer.py          # writes tokenizer_books/
"""

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers, decoders
from transformers import PreTrainedTokenizerFast

import config

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]   # [PAD] must be id 0


def book_files():
    return sorted(str(p) for p in Path(config.BOOKS_DIR).glob("*.txt"))


def train_tokenizer(out_dir=config.TOKENIZER_DIR, vocab_size=config.BOOK_VOCAB_SIZE):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, min_frequency=2,
                                        special_tokens=SPECIALS)
    tok.train(book_files(), trainer)
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"],
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    return fast


def load_tokenizer(path=config.TOKENIZER_DIR):
    if not (Path(path) / "tokenizer.json").exists():
        return train_tokenizer(path)
    return PreTrainedTokenizerFast.from_pretrained(str(path))


if __name__ == "__main__":
    t = train_tokenizer()
    print(f"vocab: {t.vocab_size} -> {config.TOKENIZER_DIR}")
    print(t.tokenize("Noirtier handed the conspiracy dossier to Villefort at Marseilles."))
