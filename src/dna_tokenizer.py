from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional, Union, Tuple

class DNACharacterTokenizer(PreTrainedTokenizer):
    """
    Simple character-level tokenizer for DNA sequences.
    """
    def __init__(
        self,
        max_len=1024,
        **kwargs
    ):
        # Define vocabulary
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "A": 5,
            "C": 6,
            "G": 7,
            "T": 8,
            "N": 9,
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len
        
        # Set special token attributes explicitly
        # These attributes are what the HF data collator looks for
        self._pad_token = "[PAD]"
        self._unk_token = "[UNK]"
        self._cls_token = "[CLS]"
        self._sep_token = "[SEP]"
        self._mask_token = "[MASK]"
        
        # Call parent init
        super().__init__(
            pad_token=self._pad_token,
            unk_token=self._unk_token,
            cls_token=self._cls_token,
            sep_token=self._sep_token,
            mask_token=self._mask_token,
            **kwargs
        )
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        return list(text.upper())
    
    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["[UNK]"])
    
    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, "[UNK]")
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.vocab["[CLS]"]] + token_ids_0 + [self.vocab["[SEP]"]]
        cls = [self.vocab["[CLS]"]]
        sep = [self.vocab["[SEP]"]]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if x in [self.vocab["[CLS]"], self.vocab["[SEP]"]] else 0 for x in token_ids_0]
        
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
    
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.vocab["[SEP]"]]
        cls = [self.vocab["[CLS]"]]
        
        if token_ids_1 is None:
            return [0] * len(cls + token_ids_0 + sep)
        
        return [0] * len(cls + token_ids_0 + sep) + [1] * len(token_ids_1 + sep)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the vocabulary dictionary to a file."""
        import os
        import json

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(out_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, ensure_ascii=False))

        return (out_vocab_file,)
