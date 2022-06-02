#!/usr/bin/env python
import pandas as pd
import os
from pathlib import Path
import subprocess
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO

from utils import read_queries

SCORE_PROTEIN = "score_protein.py"
BASE_PATH = "work/training_data/"
DATASET_IDS = "work/input_lists/train_test_161"


def split_chains(full_model_path):
    idx = Path(full_model_path).stem.strip(".pdb")[:4]
    parser = PDBParser(QUIET=1)
    io = PDBIO()
    struct = parser.get_structure(idx, full_model_path)
    pdb_chains = struct.get_chains()
    for chain in pdb_chains:
        io.set_structure(chain)
        io.save(struct.get_id() + chain.get_id() + ".pdb")


def main():
    data = []
    for idx in read_queries(DATASET_IDS):
        os.chdir(Path(BASE_PATH) / idx)
        split_chains(f"{idx}.pdb")
        rec = f"{idx[:4] + idx[-2]}.pdb"
        lig = f"{idx[:4] + idx[-1]}.pdb"

        for protein in Path(f"{idx[:4] + idx[-2]}.{idx[:4] + idx[-1]}").glob(
            "model.000.*.pdb"
        ):
            try:
                proc = subprocess.run(
                    [SCORE_PROTEIN, "-i", str(protein), "-r", rec, "-l", lig],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except Exception as err:
                print(err)
            else:
                data.append(proc.stdout.strip().split("\t"))

    df_data = pd.DataFrame(data, columns=["model_id", "a_score", "irmsd"])
    df_data.to_csv("/u21/aybars/scratch/torch_a_score_out")
    print("Finished")


if __name__ == "__main__":
    main()
