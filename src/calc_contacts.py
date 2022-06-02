#!/usr/bin/env python
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import prody

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_id", "-i")
parser.add_argument("--piper_complexes_dir", "-p")
parser.add_argument("--out_dir", "-o")
args = parser.parse_args()

NACCES = "/opt/protein/bin/naccess"
IFACE = "lib/iface/iface"

project_root = Path().cwd().resolve()


class ProteinModel:
    def __init__(self, model_id, path):
        self.model_id = model_id
        self.path = path

    def read_ca_coord(self):
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
                    a_name = line[12:16]

                    if "CA" not in a_name:
                        continue

                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append([x, y, z])

        return coords

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
                    r_name2 = line[3]
                    a_name2 = line[2]

                    key2 = "_".join([lig_index, r_name2, a_name2])

                    if key2 in key_list:
                        x, y, z = float(line[6]), float(line[7]), float(line[8])
                        coords.append([x, y, z])

        return coords

    def extract_chain(self, chain, out_file):
        selected_chain = prody.parsePDB(self.path, chain=chain)
        prody.writePDBStream(out_file, selected_chain)


def find_file_with_pattern(pattern: str):
    file_path = Path.cwd().resolve().glob(pattern)

    return file_path


def make_output_dir(output_dir: str):
    query_output_dir = Path(output_dir) / args.pdb_id
    query_output_dir.mkdir(parents=True, exist_ok=True)


def piper_docked_complexes(idx, complex_dir):
    idx_a, idx_b = (idx[:4] + idx[-2]), (idx[:4] + idx[-1])

    if (complex_dir / idx / f"{idx_a}.{idx_b}").is_dir():
        model_dir = complex_dir / idx / f"{idx_a}.{idx_b}"
    else:
        model_dir = complex_dir / idx / f"{idx_b}.{idx_a}"

    for model_path in model_dir.glob("*000.00.pdb"):
        model_name = f"{idx}.{model_path.name.strip('.pdb').strip('.model')}"
        model_id = "-".join(model_name.split("."))

        yield ProteinModel(model_id, model_path)


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

                r_name1 = line[4]
                a_name1 = line[5]
                r_name2 = line[9]
                a_name2 = line[10]

                key1 = f"{r_name1}_{a_name1}"
                key2 = f"{r_name2}_{a_name2}"

                if key1 not in sum_contact_dict:
                    sum_contact_dict[key1] = {}

                if key2 not in sum_contact_dict[key1]:
                    sum_contact_dict[key1][key2] = 0

                sum_contact_dict[key1][key2] += 1

        return sum_contact_dict


def calc_contact_ratio(inputs, type):
    asa_sum1, asa_sum2, contacts, rec_total_area, lig_total_area = inputs

    tiny = 0.0000001
    result = {}

    for key1 in asa_sum1:
        denominator_1 = float(asa_sum1[key1])

        for key2 in asa_sum2:
            denominator_2 = float(asa_sum2[key2])

            number_of_observations = 0.0

            if key1 in contacts:
                if key2 in contacts[key1]:
                    number_of_observations = contacts[key1][key2]

            denom_normalized1 = denominator_1 / rec_total_area
            denom_normalized2 = denominator_2 / lig_total_area
            f_expected_12 = denom_normalized1 * denom_normalized2

            ratio = number_of_observations / (f_expected_12 + tiny)
            ratio = np.log(ratio + 1.0)
            result[f"{key1}_{key2}"] = ratio

    if type == "receptor":
        for key1 in asa_sum1:
            denominator_1 = float(asa_sum1[key1])
            result[key1] = denominator_1

    elif type == "ligand":
        for key2 in asa_sum2:
            denominator_2 = float(asa_sum2[key2])
            result[key2] = denominator_2

    return result


def run_n_access(temp_rec, temp_lig):
    n_access_cmd_a = [NACCES, temp_lig]
    n_access_cmd_b = [NACCES, temp_rec]
    subprocess.run(n_access_cmd_a, check=True)
    subprocess.run(n_access_cmd_b, check=True)


def create_iface_command(arguments: list):
    rec_path, lig_path, receptor_chain, ligand_chain = arguments

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
        proc = subprocess.run(iface_cmd, stdout=out, stderr=out, text=True, check=True)
        print(f"{' '.join(proc.args)}")


def write_normalized_contacts(contacts_result, norm_out_file):
    df_cont = pd.DataFrame.from_dict(
        contacts_result, orient="index", columns=["contact_ratio"]
    )
    df_cont.reset_index(inplace=True)

    df_cont["res1"] = (
        df_cont["index"].str.split("_", expand=True)[0]
        + "_"
        + df_cont["index"].str.split("_", expand=True)[1]
    )
    df_cont.drop("index", axis="columns", inplace=True)
    df_cont = df_cont[["res1", "contact_ratio"]]
    df_cont.to_csv(norm_out_file, sep="\t", header=False, index=False)


def run_commands(contacts_inputs: list):
    (
        pdb_path,
        protein_complex,
        rec_path,
        lig_path,
        receptor_chain,
        ligand_chain,
    ) = contacts_inputs

    os.chdir(pdb_path)

    iface_out = Path(f"{protein_complex.model_id}.iface")

    arguments = [rec_path, lig_path, receptor_chain, ligand_chain]
    iface_cmd = create_iface_command(arguments)

    run_iface(iface_cmd, iface_out)

    run_n_access(rec_path, lig_path)

    a_asa_pattern = f"{protein_complex.model_id}-{receptor_chain}*.asa"
    b_asa_pattern = f"{protein_complex.model_id}-{ligand_chain}*.asa"

    a_asa, b_asa = (
        find_file_with_pattern(a_asa_pattern),
        find_file_with_pattern(b_asa_pattern),
    )

    asa_sum1, asa_sum2 = sum_asa(str(next(a_asa))), sum_asa(str(next(b_asa)))
    contacts = sum_contacts(str(iface_out))

    rec_total_area = sum([value for value in asa_sum1.values()])
    lig_total_area = sum([value for value in asa_sum2.values()])

    inputs = [asa_sum1, asa_sum2, contacts, rec_total_area, lig_total_area]
    contacts_result_rec = calc_contact_ratio(inputs, "receptor")
    norm_out_file_rec = Path(f"{protein_complex.model_id}_rec-norm.tsv")
    write_normalized_contacts(contacts_result_rec, norm_out_file_rec)

    inputs = [asa_sum1, asa_sum2, contacts, rec_total_area, lig_total_area]
    contacts_result_lig = calc_contact_ratio(inputs, "ligand")
    norm_out_file_lig = Path(f"{protein_complex.model_id}_lig-norm.tsv")
    write_normalized_contacts(contacts_result_lig, norm_out_file_lig)


def run_contact_piper_native(contact_output_path):
    receptor_chain, ligand_chain = args.pdb_id[-2], args.pdb_id[-1]
    native_stem = Path(args.piper_complexes_dir) / args.pdb_id / f"{args.pdb_id[:4]}"

    native_rec = ProteinModel(
        f"{args.pdb_id}-native", f"{str(native_stem)}{receptor_chain}.pdb"
    )
    native_lig = ProteinModel(
        f"{args.pdb_id}-native", f"{str(native_stem)}{ligand_chain}.pdb"
    )

    native_rec_path = (
        contact_output_path / f"{native_rec.model_id}-{receptor_chain}.pdb"
    )
    native_lig_path = contact_output_path / f"{native_lig.model_id}-{ligand_chain}.pdb"

    with open(native_rec_path, "w") as temp_rec, open(native_lig_path, "w") as temp_lig:
        native_rec.extract_chain(receptor_chain, temp_rec)
        native_lig.extract_chain(ligand_chain, temp_lig)

    native = ProteinModel(f"{args.pdb_id}-native", None)

    contacts_inputs = [
        contact_output_path,
        native,
        native_rec_path,
        native_lig_path,
        receptor_chain,
        ligand_chain,
    ]

    run_commands(contacts_inputs)

    os.chdir(project_root)


def run_contact_calc_piper():
    contact_output_path = Path(args.out_dir).resolve() / args.pdb_id
    receptor_chain, ligand_chain = args.pdb_id[-2], args.pdb_id[-1]

    for piper_complex in piper_docked_complexes(
        args.pdb_id, Path(args.piper_complexes_dir)
    ):
        with tempfile.NamedTemporaryFile(
            mode="w+t",
            prefix=f"{piper_complex.model_id}-{receptor_chain}-",
            suffix=".pdb",
        ) as temp_rec, tempfile.NamedTemporaryFile(
            mode="w+t",
            prefix=f"{piper_complex.model_id}-{ligand_chain}-",
            suffix=".pdb",
        ) as temp_lig:
            piper_complex.extract_chain(receptor_chain, temp_rec)
            piper_complex.extract_chain(ligand_chain, temp_lig)

            contacts_inputs = [
                contact_output_path,
                piper_complex,
                temp_rec.name,
                temp_lig.name,
                receptor_chain,
                ligand_chain,
            ]

            run_commands(contacts_inputs)
            clean_files()
            os.chdir(project_root)


def clean_files():

    all_files = [
        Path().cwd().glob(extension)
        for extension in ("*.asa", "*.rsa", "*.log", "*.iface", "*.pdb")
    ]

    for file_group in all_files:
        for file in file_group:
            if os.path.exists(file):
                os.remove(file)


if __name__ == "__main__":
    make_output_dir(args.out_dir)
    run_contact_calc_piper()
