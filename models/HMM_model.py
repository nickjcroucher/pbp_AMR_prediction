from typing import List

import pyhmmer
from pyhmmer.plan7 import HMM


def build_hmm(sequences: List[str]) -> HMM:
    seqs = [
        pyhmmer.easel.TextSequence(name=str(i).encode(), sequence=j)
        for i, j in enumerate(sequences)
    ]
    alphabet = pyhmmer.easel.Alphabet.amino()
    msa = pyhmmer.easel.TextMSA(name=b"msa", sequences=seqs).digitize(alphabet)
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    return builder.build_msa(msa, background)[0]
