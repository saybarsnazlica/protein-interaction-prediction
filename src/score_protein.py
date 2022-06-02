#!/usr/bin/env python
import argparse
import math
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from torch.nn import (
    Sequential,
    Flatten,
    Linear,
    Module,
    ReLU,
    Sigmoid,
    Dropout,
)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--protein_complex_file")
parser.add_argument("-r", "--reference_receptor")
parser.add_argument("-l", "--reference_ligand")
args = parser.parse_args()


NACCESS = "/opt/protein/bin/naccess"
IFACE = "../lib/iface/iface"
TMP_WORK_DIR = "./tmp-work-dir/"
PROJECT_DIR = Path.cwd().resolve()
ATOM_NAMES_PATH = PROJECT_DIR / "work" / "atom_names"
SVD_MODEL_PATH = PROJECT_DIR / "models" / "svd_model_500"
ML_MODEL_PATH = PROJECT_DIR / "models" / "pytorch" / "model_20220505_183633_4.pth"
NORMALIZER_PATH = (
    PROJECT_DIR / "models" / "pytorch" / "normalizer161_20220505_114320.pkl"
)


class ProteinModel:
    def __init__(self, model_id, path):
        self.idx = model_id
        self.path = path

    def get_full_id(self):
        return "_".join([self.idx] + Path(self.path).parts[-1].split(".")[1:3])

    def extract_chain(self, chain, out_file):
        parser = PDBParser(QUIET=1)
        io = PDBIO()
        struct = parser.get_structure(self.idx, self.path)
        pdb_chains = struct.get_chains()

        for struct_chain in pdb_chains:
            if struct_chain.id == chain:
                io.set_structure(struct_chain)
                io.save(out_file)

    def read_coord(self, key_list):
        coords = []
        skip_ligand = None

        with open(self.path) as fh:
            for line in fh:
                if "HEADER" in line:
                    if "lig" in line.split()[1]:
                        skip_ligand = False
                    elif "rec" in line.split()[1]:
                        skip_ligand = True
                elif skip_ligand:
                    continue

                if len(line) > 50 and line[0:4] == "ATOM":
                    line = line.split()
                    lig_index = line[5]
                    a_name2, r_name2 = line[2], line[3]
                    key2 = "_".join([lig_index, r_name2, a_name2])
                    if key2 in key_list:
                        x, y, z = float(line[6]), float(line[7]), float(line[8])
                        coords.append([x, y, z])

        return coords


class Net(Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        self.layers = Sequential(
            Flatten(),
            Linear(n_inputs, 64),
            Dropout(),
            ReLU(),
            Linear(64, 32),
            Dropout(),
            ReLU(),
            Linear(32, 1),
            Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def get_ligand_contacts(iface_out_path):
    contacts = []
    with open(iface_out_path) as iface_out:
        for line in iface_out:
            if line[:7] == "CONTACT":
                line = line.split()
                lig_index = line[7]
                r_name2, a_name2 = line[9], line[10]
                key2 = "_".join([lig_index, r_name2, a_name2])
                contacts.append(key2)

    return contacts


def make_output_dir(output_dir: str):
    output_directory = Path(output_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)


def sum_asa(asa_file):
    asa_sum = {}
    with open(asa_file) as asa_file:
        for line in asa_file:
            line = line.split()
            r_name = line[3]
            a_name = line[2]
            asa = float(line[9])
            key = r_name + "_" + a_name
            if key not in asa_sum:
                asa_sum[key] = 0.0
            asa_sum[key] += asa

    return asa_sum


def sum_contacts(iface_out):
    sum_contact_dict = {}

    with open(iface_out) as iface_out:
        for line in iface_out:
            if line[:7] == "CONTACT":
                line = line.split()
                r_name1, a_name1 = line[4], line[5]
                r_name2, a_name2 = line[9], line[10]
                key1 = f"{r_name1}_{a_name1}"
                key2 = f"{r_name2}_{a_name2}"
                if key1 not in sum_contact_dict:
                    sum_contact_dict[key1] = {}
                if key2 not in sum_contact_dict[key1]:
                    sum_contact_dict[key1][key2] = 0
                sum_contact_dict[key1][key2] += 1
        return sum_contact_dict


def calc_contact_ratio(inputs):
    asa_sum1, asa_sum2, contacts, rec_total_area, lig_total_area = inputs
    eps = 0.0000001
    result = {}

    for key1 in asa_sum1:
        denominator_1 = float(asa_sum1[key1])
        for key2 in asa_sum2:
            denominator_2 = float(asa_sum2[key2])
            number_of_observations = 0
            if key1 in contacts:
                if key2 in contacts[key1]:
                    number_of_observations = contacts[key1][key2]
            if not number_of_observations:
                continue
            denom_normalized1 = denominator_1 / rec_total_area
            denom_normalized2 = denominator_2 / lig_total_area
            f_expected_12 = denom_normalized1 * denom_normalized2
            ratio = number_of_observations / (f_expected_12 + eps)
            ratio = np.log(ratio + 1.0)
            result[f"{key1}_{key2}"] = ratio

    return result


def run_naccess(temp_rec, temp_lig):
    subprocess.run([NACCESS, temp_rec], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([NACCESS, temp_lig], check=True, stdout=subprocess.DEVNULL)


def create_iface_command(inputs):
    rec_path, lig_path, receptor_chain, ligand_chain = inputs
    iface_cmd = [
        IFACE,
        "--rf",
        str(rec_path),
        "--lf",
        str(lig_path),
        "--d",
        "5",
        "--v",
        "1",
        "--rc",
        receptor_chain,
        "--lc",
        ligand_chain,
    ]

    return iface_cmd


def run_iface(iface_cmd, out_path):
    with open(out_path, "w") as out:
        subprocess.run(iface_cmd, stdout=out, stderr=out, text=True, check=True)


def write_normalized_contacts(contacts_result, norm_out_file):
    df_cont = pd.DataFrame.from_dict(
        contacts_result, orient="index", columns=["contact_ratio"]
    )
    df_cont.reset_index(inplace=True)
    df_split = df_cont["index"].str.split("_", expand=True)
    r1, r2, r3, r4 = df_split[0], df_split[1], df_split[2], df_split[3]
    df_cont["res1"] = r1 + "_" + r2
    df_cont["res2"] = r3 + "_" + r4
    df_cont.drop("index", axis="columns", inplace=True)
    df_cont = df_cont[["res1", "res2", "contact_ratio"]]
    df_cont.to_csv(norm_out_file, sep="\t", header=False, index=False)


def run_contact_commands(inputs):
    output_folder, model_idx, rec_file, lig_file, receptor_chain, ligand_chain = inputs

    os.chdir(output_folder)
    iface_out = f"{model_idx}.iface"
    arg_list = [rec_file, lig_file, receptor_chain, ligand_chain]
    iface_cmd = create_iface_command(arg_list)
    run_iface(iface_cmd, iface_out)
    run_naccess(rec_file, lig_file)
    a_asa, b_asa = f"{model_idx}_rec.asa", f"{model_idx}_lig.asa"
    asa_sum1, asa_sum2 = sum_asa(a_asa), sum_asa(b_asa)
    contacts = sum_contacts(iface_out)
    rec_total_area = sum([value for value in asa_sum1.values()])
    lig_total_area = sum([value for value in asa_sum2.values()])
    contact_inputs = [asa_sum1, asa_sum2, contacts, rec_total_area, lig_total_area]
    contacts_result = calc_contact_ratio(contact_inputs)
    norm_out_file = Path(f"{model_idx}-norm.tsv")
    write_normalized_contacts(contacts_result, norm_out_file)
    os.chdir(CWD)


def run_contact_piper_native(receptor_chain, ligand_chain, output_path):
    native_rec_path = Path(args.reference_receptor).resolve()
    native_lig_path = Path(args.reference_ligand).resolve()
    native_rec = ProteinModel("native_rec", str(native_rec_path))
    native_lig = ProteinModel("native_lig", str(native_lig_path))
    native_rec_out = output_path / "native_rec.pdb"
    native_lig_out = output_path / "native_lig.pdb"

    with open(native_rec_out, "w") as rec_out, open(native_lig_out, "w") as lig_out:
        native_rec.extract_chain(receptor_chain, rec_out)
        native_lig.extract_chain(ligand_chain, lig_out)

    contacts_inputs = [
        output_path,
        "native",
        str(native_rec_out),
        str(native_lig_out),
        receptor_chain,
        ligand_chain,
    ]

    run_contact_commands(contacts_inputs)


def read_atomnames(fnames):
    namelist = []
    namedict = {}
    with open(fnames) as fh2:
        for line in fh2:
            w = line.strip().split()
            if len(w) != 2:
                continue
            i, s = w[0], w[1]
            namedict[s] = int(i)
            namelist.append(s)

    return namelist, namedict


def parse_norm_file(namedict, pairindex, nfeat, f):
    mat = np.zeros(nfeat)
    num = {}
    denom = {}

    with open(f, "r") as fh:
        for line in fh:
            w = line.strip().split("\t")
            if len(w) < 3:
                continue
            key1, key2 = w[0], w[1]
            val = float(w[-1])

            if key1 not in namedict or key2 not in namedict:
                continue
            index1, index2 = namedict[key1], namedict[key2]
            pindex = pairindex[index1][index2]

            if pindex not in num:
                num[pindex] = 0.0
                denom[pindex] = 0.0
            num[pindex] += val
            denom[pindex] += 1.0

    for pindex in num:
        ratio = 0.0
        if int(denom[pindex]) > 0:
            ratio = float(num[pindex]) / float(denom[pindex])
        mat[pindex] = ratio

    return mat


def reduce_dimensions(norm_file_path):
    namelist, namedict = read_atomnames(ATOM_NAMES_PATH)
    nat = len(namelist)
    nfeat = int(float(nat * (nat - 1)) / 2.0) + nat
    pairindex = np.zeros((nat, nat), dtype=int)

    count = 0
    for i in range(0, nat):
        for j in range(0, nat):
            if j >= i:
                pairindex[i, j] = count
                pairindex[j, i] = count
                count += 1


    if not os.path.isfile(norm_file_path):
        print("Can't find ", norm_file_path)

    with open(SVD_MODEL_PATH, "rb") as model_in:
        svd = pickle.load(model_in)
        big_mat = parse_norm_file(namedict, pairindex, nfeat, norm_file_path)
        reduced_mat = svd.transform([big_mat])
    
    return reduced_mat


def calculate_a_score(reduced_mat):
    pytorch_model = Net(500)
    checkpoint = torch.load(ML_MODEL_PATH)
    pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    norm = pickle.load(open(NORMALIZER_PATH, "rb"))
    reduced_mat = norm.transform(reduced_mat)
    reduced_mat = torch.from_numpy(reduced_mat.astype(np.float32))
    a_score = pytorch_model(reduced_mat)
    a_score = a_score[0][0]
    return a_score


def square_distances(r1, r2):
    dist = 0.0
    for k in range(0, 3):
        del12 = r1[k] - r2[k]
        dist += del12 * del12
    return dist


def calculate_rmsd(coords1, coords2):
    n = len(coords1)
    if not n or n != len(coords2):
        raise Exception("Error reading coords")
    rmsd = 0.0
    for k in range(0, n):
        d2 = square_distances(coords1[k], coords2[k])
        rmsd += d2
    rmsd = math.sqrt(rmsd / float(n))
    return rmsd


def calculate_interface_rmsd(protein, output_path):
    native_lig = ProteinModel("native_lig", args.reference_ligand)
    native_ligand_contacts = get_ligand_contacts(output_path / "native.iface")
    init_lig_coords = native_lig.read_coord(native_ligand_contacts)
    docked_lig_coords = protein.read_coord(native_ligand_contacts)
    rmsd = calculate_rmsd(init_lig_coords, docked_lig_coords)
    return rmsd


def run_feature_pipeline():
    output_path = Path(TMP_WORK_DIR).resolve()
    model_id_stem = Path(args.reference_receptor).stem.strip(".pdb")[:4]
    receptor_chain = Path(args.reference_receptor).stem.strip(".pdb")[-1]
    ligand_chain = Path(args.reference_ligand).stem.strip(".pdb")[-1]
    model_id = model_id_stem + receptor_chain + ligand_chain
    protein_model = ProteinModel(model_id, args.protein_complex_file)

    with open(
        Path(TMP_WORK_DIR) / f"{protein_model.idx}_rec.pdb", "w"
    ) as rec_file, open(
        Path(TMP_WORK_DIR) / f"{protein_model.idx}_lig.pdb", "w"
    ) as lig_file:
        protein_model.extract_chain(receptor_chain, rec_file)
        protein_model.extract_chain(ligand_chain, lig_file)

    contacts_inputs = [
        output_path,
        protein_model.idx,
        Path(rec_file.name).resolve(),
        Path(lig_file.name).resolve(),
        receptor_chain,
        ligand_chain,
    ]

    run_contact_commands(contacts_inputs)
    run_contact_piper_native(receptor_chain, ligand_chain, output_path)

    return protein_model


def main():
    make_output_dir(TMP_WORK_DIR)
    prot = run_feature_pipeline()
    feat_vec = reduce_dimensions(Path(TMP_WORK_DIR) / f"{prot.idx}-norm.tsv")
    a_score = calculate_a_score(feat_vec)
    irmsd = calculate_interface_rmsd(prot, Path(TMP_WORK_DIR))
    print(f"ID: {prot.get_full_id()}\tA-Score: {a_score:.3f}\tIRMSD: {irmsd:.3f}")
    shutil.rmtree(TMP_WORK_DIR)


if __name__ == "__main__":
    main()
