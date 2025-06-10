from typing import Optional, Union, overload

import torch
from torch import Tensor as TT
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from src.modeling.probe import Probe
from src.modeling.uci_tokenizers import UciTileTokenizer

UCI_SCORED_RECORD = dict
"""
Single lichess-uci-scored record with columns:
    ["Site", "Transcript", "Scores"]
"""


class DataWrapper:

    def __init__(self, data: dict[str, list[dict[tuple[int, str], TT]]], metadata={}):
        self.metadata = metadata
        self.raw = data
        self.categories = list(data.keys())
        cat1 = data[self.categories[0]]
        self.num_games = len(cat1)
        game0 = cat1[0]
        self._raw_keys = game0.keys()

    @overload
    def get(self, category: str, game: int, layer: int, mode: str) -> TT: ...
    @overload
    def get(self, category: str, game: int, layer: int) -> dict[str, TT]: ...
    @overload
    def get(self, category: str, game: int) -> dict[tuple[int, str], TT]: ...
    @overload
    def get(self, category: str) -> list[dict[tuple[int, str], TT]]: ...
    @overload
    def get(self, game: int) -> tuple[dict[tuple[int, str], TT]]: ...

    def get(
        self,
        category: Optional[str] = None,
        game: Optional[int] = None,
        layer: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> TT:
        r"""Access a tensor from the raw data.

        Args:
            category (str): select either 'inputs' or 'labels'
            game: (int): index of game in batch
            layer: (int): probe layer
            mode: (str): probe mode
        """
        if category is None:
            return tuple(self.get(cat, game, layer, mode) for cat in self.raw.keys())

        if game is None:
            return self.raw[category]

        if layer is None:
            return self.raw[category][game]

        if mode is None:
            return {
                m: v for (l, m), v in self.raw[category][game].items() if l == layer
            }

        return self.raw[category][game].get((layer, mode))

    def keys(self):
        """Maintain backward compatibility with direct dictionary access."""
        return self.raw.keys()

    def __getitem__(self, key):
        """Maintain backward compatibility with direct dictionary access."""
        return self.raw[key]

    def __setitem__(self, key, value):
        """Maintain backward compatibility with direct dictionary assignment."""
        self.raw[key] = value

    def __repr__(self):
        return "\n  - ".join(
            [
                "DataWrapper",
                f"categories={self.categories}",
                f"num_games={self.num_games}",
                f"probe_keys={list(self._raw_keys)}",
                f"metadata={self.metadata}",
            ]
        )

    def __iter__(self):
        """Iterate over the games"""
        for game_idx in range(self.num_games):
            yield self.get(game=game_idx)


class ThreePhaseCollator:

    tokenizer: PreTrainedTokenizerFast
    llm: GPT2LMHeadModel
    probe: Probe
    label_column_name: str

    def __init__(self, tokenizer, llm, probe, label_column_name="label"):
        self.tokenizer = tokenizer
        self.llm = llm
        self.probe = probe
        self.label_column_name = label_column_name

    def __call__(
        self,
        transcript_or_batch: Union[str, UCI_SCORED_RECORD, list[UCI_SCORED_RECORD]],
    ):
        return ThreePhaseCollator.collate(
            transcript_or_batch,
            tokenizer=self.tokenizer,
            llm=self.llm,
            probe=self.probe,
            label_column_name=self.label_column_name,
        )

    @staticmethod
    def collate(
        transcript_or_batch: Union[str, UCI_SCORED_RECORD, list[UCI_SCORED_RECORD]],
        tokenizer: PreTrainedTokenizerFast,
        llm: GPT2LMHeadModel,
        probe: Probe,
        label_column_name: str,
        ply_token_offset: int = 0,
    ) -> DataWrapper:

        assert ply_token_offset in range(3)

        if isinstance(transcript_or_batch, str):
            transcript_or_batch = [{"Transcript": transcript_or_batch}]

        if isinstance(transcript_or_batch, dict):
            transcript_or_batch = [transcript_or_batch]

        batch = transcript_or_batch

        has_labels = label_column_name in transcript_or_batch[0]

        num_bins = probe.config.out_features
        batch_size = len(batch)
        transcripts = [e["Transcript"] for e in batch]
        raw_labels = [e[label_column_name] for e in batch] if has_labels else None

        encoding = tokenizer.batch_encode_plus(
            transcripts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=llm.config.n_ctx,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        plies_per_game = [len(t.split()) for t in transcripts]

        ply_indices = [
            list(
                range(
                    1 + ply_token_offset,  # add 1 for <|startoftext|> token
                    min(
                        3 * n + 1, llm.config.n_ctx
                    ),  # three tokens in each ply, plus <|startoftext|> token
                    3,  # skip by 3
                )
            )
            for n in plies_per_game
        ]

        # hidden_state.shape = [n_layer, [batch, n_pos, n_embed]]
        hidden_states: list[TT] = llm.forward(
            encoding["input_ids"].to(llm.device),
            output_hidden_states=True,
            return_dict=True,
        )["hidden_states"]

        inputs_by_game = [{} for _ in range(batch_size)]
        labels_by_game = [{} for _ in range(batch_size)] if has_labels else None
        boundaries = torch.linspace(0.0, 1.0, num_bins - 1)

        # must handle one sample at a time since
        # plies don't always align across games
        for game_idx in range(batch_size):

            # need to create a mask we which selects
            # token positions specific to each player
            game_plies = torch.tensor(ply_indices[game_idx])

            # We use tensors to facilitate indexing here & below
            if has_labels:
                # slice here since llm's context limits may force us to use fewer game_labels
                # than are actually available in the dataset
                game_labels = torch.tensor([raw_labels[game_idx]] * game_plies.shape[0])
                # torch.tensor(raw_labels[game_idx])[: game_plies.shape[0]]
                # if len(game_labels) + 1 != len(transcripts[game_idx].split()):
                if len(game_labels) != len(transcripts[game_idx].split()):
                    continue  # TODO: figure out why these label arrays don't have the same number of elements as the transcripts

                binned_labels = torch.bucketize(game_labels, boundaries).long()
                # CrossEntropyLoss requires long 😕

            for layer in range(probe.config.n_layers):
                hs_subset = hidden_states[layer][game_idx, game_plies]
                token_count = hs_subset.shape[0]
                if token_count == 0:
                    continue

                # Define mask dynamically based on the number of modes
                num_modes = len(probe.config.modes)
                mode_masks = {
                    mode: (torch.arange(token_count) % num_modes == i)
                    for i, mode in enumerate(probe.config.modes)
                }

                for mode in probe.config.modes:
                    mode_mask = mode_masks[mode]
                    if mode_mask.sum() == 0:
                        continue  # Skip if no tokens match this mode
                    try:
                        inputs = hs_subset[mode_mask]
                    except:
                        print("a")
                    inputs_by_game[game_idx][(layer, mode)] = inputs.to(probe.device)

                    if has_labels:
                        labels = binned_labels[mode_mask].to(probe.device)
                        labels_by_game[game_idx][(layer, mode)] = labels

        return DataWrapper(
            {
                "inputs": inputs_by_game,
                **({"labels": labels_by_game} if has_labels else {}),
            },
            metadata={
                "transcripts": transcripts,
                "raw_labels": raw_labels,
                "bin_edges": boundaries.tolist() if has_labels else None,
            },
        )


class VariableLenUciCollator:

    def __init__(self, tokenizer, llm, probe):
        self.tokenizer = tokenizer
        self.llm = llm
        self.probe = probe

    def __call__(
        self,
        transcript_or_batch: Union[str, UCI_SCORED_RECORD, list[UCI_SCORED_RECORD]],
    ):
        return self.collate(transcript_or_batch, self.tokenizer, self.llm, self.probe)

    @staticmethod
    def collate(
        transcript_or_batch: Union[str, UCI_SCORED_RECORD, list[UCI_SCORED_RECORD]],
        tokenizer: UciTileTokenizer,
        llm: GPT2LMHeadModel,
        probe: Probe,
    ):
        match transcript_or_batch:
            case str():
                return VariableLenUciCollator.collate(
                    {"Transcript": transcript_or_batch}, tokenizer, llm, probe
                )
            case dict():
                return VariableLenUciCollator.collate(
                    [transcript_or_batch], tokenizer, llm, probe
                )

        scores_present = "Scores" in transcript_or_batch[0]
        return VariableLenUciCollator._collate(
            transcript_or_batch, tokenizer, llm, probe, scores_present
        )

    @staticmethod
    def _collate(
        batch: list[UCI_SCORED_RECORD],
        tok: UciTileTokenizer,
        llm: GPT2LMHeadModel,
        probe: Probe,
        has_scores: bool,
    ) -> DataWrapper:

        B = probe.config.out_features
        batch_size = len(batch)
        transcripts = [e["Transcript"] for e in batch]
        scores = [e["Scores"] for e in batch] if has_scores else None

        encoding = tok.batch_encode_plus(
            transcripts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=llm.config.n_ctx,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        ply_indices = [
            # [t + 1 for t in game if t + 1 < llm.config.n_ctx][:-1]
            [t for t in game if t < llm.config.n_ctx]
            for game in tok.compute_ply_end_indices(encoding)
        ]

        # hidden_state.shape = [n_layer, [batch, n_pos, n_embed]]
        hidden_states: list[TT] = llm.forward(
            encoding["input_ids"].to(llm.device),
            output_hidden_states=True,
            return_dict=True,
        )["hidden_states"]

        inputs_by_game = [{} for _ in range(batch_size)]
        labels_by_game = [{} for _ in range(batch_size)] if has_scores else None
        boundaries = torch.linspace(0.0, 1.0, B - 1)

        # must handle one sample at a time since
        # plies don't always align across games
        for game_idx in range(batch_size):

            # need to create a mask we which selects
            # token positions specific to each player
            game_plies = torch.tensor(ply_indices[game_idx])

            # We use tensors to facilitate indexing here & below
            if has_scores:
                # slice here since llm's context limits may force us to use fewer game_scores
                # than are actually available in the dataset
                game_scores = torch.tensor(scores[game_idx])[: game_plies.shape[0]]
                # if len(game_scores) + 1 != len(transcripts[game_idx].split()):
                if len(game_scores) != len(transcripts[game_idx].split()):
                    continue  # TODO: figure out why these score arrays don't have the same number of elements as the transcripts

                binned_scores = torch.bucketize(game_scores, boundaries).long()
                # CrossEntropyLoss requires long 😕

            for layer in range(probe.config.n_layers):
                hs_subset = hidden_states[layer][game_idx, game_plies]
                token_count = hs_subset.shape[0]
                if token_count == 0:
                    continue

                # Define mask dynamically based on the number of modes
                num_modes = len(probe.config.modes)
                mode_masks = {
                    mode: (torch.arange(token_count) % num_modes == i)
                    for i, mode in enumerate(probe.config.modes)
                }

                for mode in probe.config.modes:
                    mode_mask = mode_masks[mode]
                    if mode_mask.sum() == 0:
                        continue  # Skip if no tokens match this mode

                    inputs = hs_subset[mode_mask]
                    inputs_by_game[game_idx][(layer, mode)] = inputs.to(probe.device)

                    if has_scores:
                        labels = binned_scores[mode_mask].to(probe.device)
                        labels_by_game[game_idx][(layer, mode)] = labels

        return DataWrapper(
            {
                "inputs": inputs_by_game,
                **({"labels": labels_by_game} if has_scores else {}),
            },
            metadata={
                "transcripts": transcripts,
                "raw_scores": scores,
                "bin_edges": boundaries.tolist() if has_scores else None,
            },
        )


class HiddenStateCollator:
    """
    A utility class for collating text inputs into hidden states using a pre-trained language model (LLM).
    This class is designed to process text inputs (either as a single string, a dictionary, or a batch of dictionaries),
    tokenize them using a specified tokenizer, and return the hidden states of the LLM rather than the logits.
    Attributes:
        tokenizer (PreTrainedTokenizerFast): The tokenizer used to preprocess the text inputs.
        llm (GPT2LMHeadModel): The pre-trained language model used to compute hidden states.
        text_column_name (str): The key in the input dictionary that contains the text data.
    Methods:
        __call__(transcript_or_batch):
            Processes the input text or batch of text and returns the hidden states of the LLM.
        collate(transcript_or_batch, tokenizer, llm, text_column_name, ply_token_offset=0):
            A static method that performs the collation of text inputs into hidden states.
    """

    tokenizer: PreTrainedTokenizerFast
    llm: GPT2LMHeadModel
    text_column_name: str

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        llm: GPT2LMHeadModel,
        text_column_name: str,
    ):
        self.tokenizer = tokenizer
        self.llm = llm.train(False).requires_grad_(False)
        self.text_column_name = text_column_name

    def __call__(
        self,
        transcript_or_batch: Union[str, dict, list[dict]],
    ):
        return HiddenStateCollator.collate(
            transcript_or_batch,
            tokenizer=self.tokenizer,
            llm=self.llm,
            text_column_name=self.text_column_name,
        )

    @staticmethod
    def collate(
        transcript_or_batch: Union[str, dict, list[dict]],
        tokenizer: PreTrainedTokenizerFast,
        llm: GPT2LMHeadModel,
        text_column_name: str,
        ply_token_offset: int = 0,
    ) -> tuple[torch.Tensor]:

        assert ply_token_offset in range(3)

        if isinstance(transcript_or_batch, str):
            transcript_or_batch = [{text_column_name: transcript_or_batch}]

        if isinstance(transcript_or_batch, dict):
            transcript_or_batch = [transcript_or_batch]

        batch = transcript_or_batch

        transcripts = [e[text_column_name] for e in batch]

        encoding = tokenizer.batch_encode_plus(
            transcripts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=llm.config.n_ctx,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        # hidden_state.shape = [n_layer, [batch, n_pos, n_embed]]
        hidden_states: tuple[torch.Tensor] = llm.forward(
            encoding["input_ids"].to(llm.device),
            output_hidden_states=True,
            return_dict=True,
        )["hidden_states"]

        return hidden_states
