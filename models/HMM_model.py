from operator import itemgetter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pyhmmer
from nptyping import NDArray
from pyhmmer.plan7 import HMM


class ProfileHMMPredictor:
    def __init__(
        self,
        training_data: pd.DataFrame,
        drop_duplicates_for_training: bool = True,
        pbp_seqs: List[str] = ["a1_seq", "b2_seq", "x2_seq"],
    ):
        self.pbp_seqs = pbp_seqs
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.background = pyhmmer.plan7.Background(self.alphabet, uniform=True)
        self.builder = pyhmmer.plan7.Builder(self.alphabet)
        self.pipeline = pyhmmer.plan7.Pipeline(
            self.alphabet, background=self.background
        )
        self.hmm_mic_dict = self._phenotype_representative_HMMs(
            training_data, drop_duplicates_for_training
        )

    def _build_hmm(self, sequences: Iterable[str], mic: float) -> HMM:
        seqs = [
            pyhmmer.easel.TextSequence(name=str(i).encode(), sequence=j)
            for i, j in enumerate(sequences)
        ]
        msa = pyhmmer.easel.TextMSA(
            name=str(mic).encode(), sequences=seqs
        ).digitize(self.alphabet)
        return self.builder.build_msa(msa, self.background)[0]

    def _phenotype_representative_HMMs(
        self,
        data: pd.DataFrame,
        unique_sequences: bool = True,
    ) -> Dict:
        hmm_mic_dict = {}  # type: ignore
        for mic, sequences in data.groupby("log2_mic"):
            sequences = sequences[self.pbp_seqs].sum(axis=1)
            if unique_sequences:
                sequences.drop_duplicates(inplace=True)
            hmm_mic_dict[mic] = self._build_hmm(sequences, mic)
        return hmm_mic_dict

    def predict_phenotype(self, seqs: Iterable[str]) -> NDArray:
        digital_sequences = [
            pyhmmer.easel.TextSequence(sequence=i).digitize(self.alphabet)
            for i in seqs
        ]

        def get_prediction(seq):
            hits = self.pipeline.scan_seq(seq, self.hmm_mic_dict.values())
            hits = [[hit.score, float(hit.name)] for hit in hits]
            hits.sort(key=itemgetter(0))
            return hits[-1][1]

        return np.array([get_prediction(seq) for seq in digital_sequences])
