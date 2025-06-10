from typing import List, MutableMapping, Tuple, Union

import chess
import tokenizers
from tokenizers import models, pre_tokenizers, processors
from torch import Tensor as TT
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_fast import BatchEncoding

OffsetsList = List[Tuple[int, int]]
DictLike = MutableMapping

# import tiktoken
# def getTiktokenizer() -> tiktoken.Encoding:
#     """
#     Defines a tiktoken-based BPE encoder for UCI chess moves. This
#     tokenizer effectively tokenizes UCI moves by the square names.
#     One notable variation is that promotions must be in upper-case.

#     Vocabulary:
#     Special Tokens (4): "\<|pad|\>", "\<|startoftext|\>", "\<|endoftext|\>", "\<|unknown|\>"
#     Square Tokens (64): a1 through h8
#     Promote Tokens (4): Q, B, R, N
#     UNUSED (8120): Need 8192-4-64-4=8120 unused tokens of the form <|unused####|>
#     """
#     special_tokens = ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|unknown|>"]
#     unused_tokens = [f"<|unused{i:04d}" for i in range(8120)]
#     chess_vocab = special_tokens + chess.SQUARE_NAMES + list("QBRN") + unused_tokens
#     mergeable_ranks = {k.encode():v for (v,k) in enumerate(chess_vocab)}
#     chess_pat_str = r'[a-h][1-8]|[QBRN]'

#     enc = tiktoken.Encoding(
#         name="chess_enc",
#         pat_str=chess_pat_str, # or \d|\s
#         mergeable_ranks=mergeable_ranks,
#         special_tokens={k:v for (v,k) in enumerate(special_tokens)},
#     )

#     return enc


class UciTokenizer(PreTrainedTokenizerFast):
    _PAD_TOKEN: str
    _UNK_TOKEN: str
    _EOS_TOKEN: str
    _BOS_TOKEN: str

    stoi: dict[str, int]
    """Integer to String mapping"""

    itos: dict[int, str]
    """String to Integer Mapping. This is the vocab"""

    def __init__(
        self,
        stoi,
        itos,
        pad_token,
        unk_token,
        bos_token,
        eos_token,
        bos_token_id,
        name_or_path,
        **kwargs,
    ):
        self.stoi = stoi
        self.itos = itos

        self._PAD_TOKEN = pad_token
        self._UNK_TOKEN = unk_token
        self._EOS_TOKEN = eos_token
        self._BOS_TOKEN = bos_token

        # Define the model
        tok_model = models.WordLevel(vocab=self.stoi, unk_token=self._UNK_TOKEN)

        slow_tokenizer = tokenizers.Tokenizer(tok_model)
        slow_tokenizer.pre_tokenizer = self._init_pretokenizer()

        # post processing adds special tokens unless explicitly ignored
        post_proc = processors.TemplateProcessing(
            single=f"{bos_token} $0",
            pair=None,
            special_tokens=[(bos_token, bos_token_id)],
        )
        slow_tokenizer.post_processor = post_proc

        super().__init__(
            tokenizer_object=slow_tokenizer,
            unk_token=self._UNK_TOKEN,
            bos_token=self._BOS_TOKEN,
            eos_token=self._EOS_TOKEN,
            pad_token=self._PAD_TOKEN,
            name_or_path=name_or_path,
            **kwargs,
        )

        # Override the decode behavior to ensure spaces are correctly handled
        def _decode(
            token_ids: int | List[int] | dict | TT,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) -> int | List[int]:

            if isinstance(token_ids, int):
                return self.itos.get(token_ids, self._UNK_TOKEN)

            if isinstance(token_ids, dict):
                token_ids = token_ids["input_ids"]

            if isinstance(token_ids, TT):
                token_ids = token_ids.tolist()

            if isinstance(token_ids, list):
                tokens_str = [self.itos.get(xi, self._UNK_TOKEN) for xi in token_ids]
                processed_tokens = self._process_str_tokens(tokens_str)

                return processed_tokens

            raise ValueError(
                f"Unknown input type to decode() for argument 'token_ids'. Received: {type(token_ids)} "
            )

        self._decode = _decode

    def _init_pretokenizer(self) -> pre_tokenizers.PreTokenizer:
        raise NotImplementedError

    def _process_str_tokens(
        self, tokens_str: list[str], return_player_ids: bool
    ) -> list[str]:
        raise NotImplementedError

    def get_id2square_list() -> list[int]:
        raise NotImplementedError


class UciTileTokenizer(UciTokenizer):
    """Uci tokenizer converting start/end tiles and promotion types each into individual tokens"""

    PLAYER_BY_INDICES = [True, False] * 1024

    SPECIAL_TOKENS = (_PAD_TOKEN, _BOS_TOKEN, _EOS_TOKEN, _UNK_TOKEN) = [
        "<|pad|>",
        "<|startoftext|>",
        "<|endoftext|>",
        "<|unknown|>",
    ]

    stoi: dict[str, int]
    itos: dict[int, str]

    _split_regex: str
    _promote_chars: str

    id2square: List[int] = list(range(4, 68))
    """
    List mapping token IDs to squares on the chess board. Order is file then rank, i.e.: 
    `A1, B1, C1, ..., F8, G8, H8`    
    """

    def get_id2square_list(self) -> List[int]:
        return self.id2square

    def __init__(self, *, upper_promotions: bool, **kwargs):
        # Remove conflicting arguments from kwargs if they exist
        kwargs.pop("pad_token", None)
        kwargs.pop("unk_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("name_or_path", None)

        self.upper_promotions = upper_promotions

        if upper_promotions:
            self._promote_chars = "QRBN"
            self._split_regex = r"[a-h][1-8]|[QRBN]"
        else:
            self._promote_chars = "qrbn"
            self._split_regex = r"[a-h][1-8]|[qrnb]"

        self.stoi = {
            tok: idx
            for tok, idx in list(
                zip(
                    self.SPECIAL_TOKENS
                    + chess.SQUARE_NAMES
                    + list(self._promote_chars),
                    range(72),
                )
            )
        }

        self.itos = {
            idx: tok
            for tok, idx in list(
                zip(
                    self.SPECIAL_TOKENS
                    + chess.SQUARE_NAMES
                    + list(self._promote_chars),
                    range(72),
                )
            )
        }

        super().__init__(
            self.stoi,
            self.itos,
            pad_token=self._PAD_TOKEN,
            unk_token=self._UNK_TOKEN,
            bos_token=self._BOS_TOKEN,
            eos_token=self._EOS_TOKEN,
            bos_token_id=self.stoi[self._BOS_TOKEN],
            name_or_path="austindavis/uci_tile_tokenizer",
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

    def _init_pretokenizer(self):
        # Pre-tokenizer to split input into UCI moves
        pattern = tokenizers.Regex(self._split_regex)
        pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Split(pattern=pattern, behavior="merged_with_previous"),
            ]
        )
        return pre_tokenizer

    def _process_str_tokens(self, token_str: list[str]):
        moves = []
        next_move = ""
        for token in token_str:

            # skip special tokens
            if token in self.all_special_tokens:
                continue

            # handle promotions
            if len(token) == 1:
                next_move += token
                continue

            # handle regular tokens if there's room
            if len(next_move) < 4:
                next_move += token
                continue

            moves.append(next_move)
            next_move = token

        moves.append(next_move)
        return " ".join(moves)

    def compute_ply_end_indices(
        self,
        inputs: Union[DictLike, TT, OffsetsList, List[OffsetsList]],
    ) -> List[Union[int, List[int]]]:
        """
        Computes the indices of the last token in each ply of a chess
        move sequence based on the offset_mapping from a tokenizer. If
        the input sequences were batched, this function will method will
        call itself recursively and return the results as a List.

        NOTE: This method does not respect the rules of chess. It only
        considers the offset mappings of the tokens.

        # Parameters:
        ----------
        **`inputs`** : Union[DictLike, TT, OffsetsList, List[OffsetsList]]
            The batch encoding that includes the offset_mapping key or else
            the offset mapping itself.

        # Returns:
        -------
        `List[Union[int, List[int]]]`
            A list of indices corresponding to the last token of each ply.
            NOTE: If the inputs are batched, the output will be a list of
            lists, with each sublist corresponding to an entry in the batch.
            However, do not assume these sublists will have equal lengths,
            as the number of moves in a tokenized sequence depends on game
            structure rather than token count, even when padding is applied.
        """
        # handle various input types
        if isinstance(inputs, DictLike):
            offset_mapping = inputs.get("offset_mapping", None)
            assert (
                offset_mapping is not None
            ), "Encoding missing offset_mapping. Re-encode using `return_offsets_mapping=True`"
        else:  # assume the inputs are the offset_mapping
            offset_mapping = inputs

        # reduce to a 2D tensor or 1D list
        if isinstance(offset_mapping[0], List):
            return [
                self.compute_ply_end_indices(offset_mapping[i])
                for i in range(len(offset_mapping))
            ]

        if isinstance(offset_mapping, TT):
            if len(offset_mapping.shape) > 2:
                return [
                    self.compute_ply_end_indices(offset_mapping[i])
                    for i in range(len(offset_mapping))
                ]
            else:
                offset_mapping = offset_mapping.tolist()

        last_token_indices = []
        seen_starts = set()

        # Iterate in reverse to ensure we catch the final token
        for i in range(len(offset_mapping) - 1, -1, -1):
            start, end = offset_mapping[i]
            # skip special tokens
            if end > 0 and end not in seen_starts:
                last_token_indices.append(i)
            seen_starts.add(start)
        last_token_indices.reverse()
        return last_token_indices

    @staticmethod
    def compute_players_from_indices(indices: Union[list[list[int]], list[int]]):
        """Computes the player responsible for each token in the input sequence."""
        assert indices, "No indices provided"

        if isinstance(indices[0], list):
            return [
                UciTileTokenizer.compute_players_from_indices(indices[i])
                for i in range(len(indices))
            ]

        # initialize all tokens as white's move
        players = [True] * (1 + indices[-1])

        # Falsify indices of black moves
        for start, end in zip(indices[::2], indices[1::2]):
            players[start + 1 : end + 1] = [False] * (end - start)

        return players

    @staticmethod
    def compute_players(encoding: BatchEncoding, according_to="output"):
        """
        Determines which player (white=True, black=False) is associated with each token in the sequence.
        This method works based on chess move sequences tokenized using the UciTileTokenizer.

        # Parameters:
        ----------
        **`encoding`** : BatchEncoding
            Tokenized input of a chess game, where each token represents a move or special token.

        **`according_to`** : str (optional, default='output')
            Specifies the perspective for associating players:
            - 'output': Returns the player whose next move is predicted by the sequence (the output move).
            - Otherwise: Returns the player associated with the input tokens (i.e., which player made each move).

        # Returns:
        -------
        List[bool]
            A list of boolean values indicating the player for each token:
            - True for white (player 1),
            - False for black (player 2).

            The list length corresponds to the number of tokens in the sequence, including special tokens if any.

        # Example Usage:
        ```
        >>> tok = UciTileTokenizer()
        >>> encoding = tok('e2e4 d7d5 e4d5 e7e6 d5e6 d8g5 e6e7 g5f6 e7f8Q')
        >>> print(encoding['input_ids'])
        [1, 16, 32, 55, 39, 32, 39, 56, 48, 39, 48, 63, 42, 48, 56, 42, 49, 56, 65, 68]
        >>> tok.compute_players(encoding)
        [True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, True, False]
        >>> tok.compute_players(encoding, according_to='input')
        [True, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, True]
        ```

        # Notes:
        -------
        This method does not rely on board position calculations. Therefore, when
        using `according_to='output'`, it cannot reliably predict which player is
        responsible for selecting the final token of the sequence. For instance,
        if a pawn is moved to the back rank (e.g., 'e7e8'), then white must select
        the promotion class on the next token; however, this algorithm will predict
        that black is responsible for selecting the next token instead of white.
        """

        if not isinstance(encoding["input_ids"][0], int):
            return [
                UciTileTokenizer.compute_players(encoding[i].ids, according_to)
                for i in range(len(encoding["input_ids"]))
            ]

        input_ids = encoding["input_ids"]

        players = [] if according_to == "output" else [True]
        current_player = False
        num_tokens_in_ply = 0
        has_specials = False

        for i, token_id in enumerate(input_ids):
            if token_id == 1:
                has_specials = True
                continue

            if num_tokens_in_ply == 0:
                # check if promotion OR unknown token ID
                if token_id > 67 or token_id == 3:
                    players.append(current_player)
                    num_tokens_in_ply = 0
                else:
                    num_tokens_in_ply += 1
                    current_player = not current_player
                    players.append(current_player)
            elif num_tokens_in_ply == 1:
                num_tokens_in_ply = 0
                players.append(current_player)
            else:
                raise ValueError("Illegal move sequence")

        if according_to == "output":
            # anticipate what output should be based on the final input token
            # see notes for more detail
            if num_tokens_in_ply == 0:
                if token_id > 67:
                    players.append(not current_player)
                else:
                    players.append(current_player)
            else:
                players.append(current_player)

        return players if has_specials else players[1:]


class PgnCharTokenizer(UciTokenizer):
    itos = {
        0: " ",
        1: "#",
        2: "+",
        3: "-",
        4: ".",
        5: "0",
        6: "1",
        7: "2",
        8: "3",
        9: "4",
        10: "5",
        11: "6",
        12: "7",
        13: "8",
        14: "9",
        15: ";",
        16: "=",
        17: "B",
        18: "K",
        19: "N",
        20: "O",
        21: "Q",
        22: "R",
        23: "a",
        24: "b",
        25: "c",
        26: "d",
        27: "e",
        28: "f",
        29: "g",
        30: "h",
        31: "x",
    }

    stoi = {
        " ": 0,
        "#": 1,
        "+": 2,
        "-": 3,
        ".": 4,
        "0": 5,
        "1": 6,
        "2": 7,
        "3": 8,
        "4": 9,
        "5": 10,
        "6": 11,
        "7": 12,
        "8": 13,
        "9": 14,
        ";": 15,
        "=": 16,
        "B": 17,
        "K": 18,
        "N": 19,
        "O": 20,
        "Q": 21,
        "R": 22,
        "a": 23,
        "b": 24,
        "c": 25,
        "d": 26,
        "e": 27,
        "f": 28,
        "g": 29,
        "h": 30,
        "x": 31,
    }

    def __init__(self):
        super().__init__(
            self.stoi,
            self.itos,
            pad_token="",
            unk_token="",
            bos_token=";",
            eos_token="",
            bos_token_id=self.stoi[";"],
            name_or_path="austindavis/pgn_char_tokenizer",
        )

    def _init_pretokenizer(self):
        # Pre-tokenizer to split input into UCI moves
        pattern = tokenizers.Regex(r".")
        pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=pattern, behavior="merged_with_previous"),
            ]
        )
        return pre_tokenizer

    def _process_str_tokens(self, token_str):
        print(token_str)
        return "".join(token_str)


class UciCharTokenizer(UciTokenizer):
    """Uci tokenizer converting every character into a token"""

    itos = {
        0: " ",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: ";",
        10: "#",
        11: "a",
        12: "b",
        13: "c",
        14: "d",
        15: "e",
        16: "f",
        17: "g",
        18: "h",
        19: "n",
        20: "r",
        21: "q",
        22: "k",
    }
    """Integer to String mapping"""

    stoi = {
        " ": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        ";": 9,
        "#": 10,
        "a": 11,
        "b": 12,
        "c": 13,
        "d": 14,
        "e": 15,
        "f": 16,
        "g": 17,
        "h": 18,
        "n": 19,
        "r": 20,
        "q": 21,
        "k": 22,
    }
    """String to Integer Mapping. This is the vocab"""

    def __init__(self):
        super().__init__(
            self.stoi,
            self.itos,
            pad_token="<pad>",
            unk_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            name_or_path="austindavis/uci_char_tokenizer",
        )

    def _init_pretokenizer(self):
        # Pre-tokenizer to split input into UCI moves
        pattern = tokenizers.Regex(r".")
        pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Split(pattern=pattern, behavior="merged_with_previous"),
            ]
        )
        return pre_tokenizer

    def _process_str_tokens(self, token_str):
        moves = []
        next_move = ""
        for token in token_str:

            # skip special tokens
            if token in self.all_special_tokens:
                continue

            if len(next_move) <= 4:
                next_move += token
                continue

            # move length is now 5

            # handle easy promotes
            if next_move[-1] in "nrqk":
                moves.append(next_move)
                next_move = token
                continue

            # handle bishop promotes
            if next_move[-1] == "b" and token in "abcdefgh":
                moves.append(next_move)
                next_move = token
                continue

            # non-promotion, clear next move
            moves.append(next_move[:-1])
            next_move = next_move[-1] + token

        moves.append(next_move)
        return moves


if __name__ == "__main__":
    tok = UciTileTokenizer(upper_promotions=True)

    # fmt: off
    
    assert tok("e2e4Q b7b8N e2e7 a1", add_special_tokens=True) == {
        "input_ids": [1, 16, 32, 68, 53, 61, 71, 16, 56, 4],
        "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }

    assert tok("e2e4Q b7b8N e2e7 a1")["input_ids"] == [1,16,32,68,53,61,71,16,56,4]

    try:
        tok.compute_ply_end_indices(tok("e2"))
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError for missing offsets")

    assert tok.compute_ply_end_indices(tok("h8", return_offsets_mapping=True)) == [1]
    assert tok.compute_ply_end_indices(tok("e2", return_offsets_mapping=True)) == [1]
    assert tok.compute_ply_end_indices(tok("", return_offsets_mapping=True)) == []

    assert tok.compute_ply_end_indices(tok("b1c3", return_offsets_mapping=True)) == [2]
    assert tok.compute_ply_end_indices(tok("b1c3Q", return_offsets_mapping=True)) == [3]
    assert tok.compute_ply_end_indices(tok("QBNR", return_offsets_mapping=True)) == [4]
    assert tok.compute_ply_end_indices(tok("QQ", return_offsets_mapping=True)) == [2]
    assert tok.compute_ply_end_indices(tok("b1b2 Q", return_offsets_mapping=True)) == [2, 3]
    assert tok.compute_ply_end_indices(tok("b1b2 QQ", return_offsets_mapping=True)) == [2, 4]
    assert tok.compute_ply_end_indices(tok("b1b2 QQQ", return_offsets_mapping=True)) == [2, 5]

    # the following shows the inconsistency in the final token handling
    assert tok.compute_ply_end_indices(tok("b1 b2 b3 b4 b5", return_offsets_mapping=True)) == [1,2,3,4,5]
    assert tok.compute_ply_end_indices(tok("b1 b2 b3 b4 b5b8", return_offsets_mapping=True)) == [1,2,3,4,6]

    assert tok.compute_ply_end_indices(tok([""], return_offsets_mapping=True)) == [[]]
    assert tok.compute_ply_end_indices(tok(["e2"], return_offsets_mapping=True)) == [[1]]
    assert tok.compute_ply_end_indices(tok(["e2", "Q"], return_offsets_mapping=True)) == [[1],[1],]
    
    assert tok.compute_ply_end_indices(tok("e2e4 d7d5 e4d5 e7e6 d5e6 d8g5 e6e7 g5f6 e7f8Q", return_offsets_mapping=True, return_tensors='pt')) == [2, 4, 6, 8, 10, 12, 14, 16, 19]

    encoding = tok("e2e4Q b7b8N e2e7 a1", return_offsets_mapping=True) # special tokens are added by default
    indices = tok.compute_ply_end_indices(encoding)
    assert tok.compute_players(encoding,  according_to='output') == [True, True, True, False, False, False, True, True, False, False]
    assert tok.compute_players(encoding, according_to='input') == [True, True, True, True, False, False, False, True, True, False]
    assert tok.compute_players_from_indices(indices) == [True,True,True,True, False, False, False, True, True, False]
    encoding = tok("e2e4Q b7b8N e2e7 a1", add_special_tokens=False, return_offsets_mapping=True)
    assert tok.compute_players(encoding,  according_to='output') == [True, True, False, False, False, True, True, False, False]
    assert tok.compute_players(encoding, according_to='input') == [True, True, True, False, False, False, True, True, False]

    # fmt: on
