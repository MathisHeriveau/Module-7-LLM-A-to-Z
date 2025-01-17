
# ****************************************************** #
#               TRAIN A TOKENIZER                        #
# ****************************************************** #

import sentencepiece as spm
import os

vocab_size = 4096
path = "LLM-Mastery-hand/"

spm.SentencePieceTrainer.train(
    input=path+"wiki.txt",
    model_prefix=path+"test_wiki_tokenizer",
    model_type="bpe",
    vocab_size=vocab_size,
    self_test_sample_size=0,
    input_format="text",
    character_coverage=0.995,
    num_threads = os.cpu_count(),
    split_digits=True,
    allow_whitespace_only_pieces=True,
    byte_fallback=True,
    unk_surface="\\243\201\207\\",
    normalization_rule_name="identity"
)

print("Tokenizer training")