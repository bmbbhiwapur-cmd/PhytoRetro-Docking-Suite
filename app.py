import time
import streamlit as st
import subprocess
import os
import shutil
import urllib.request
import urllib.parse
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64
import io

# --- CRITICAL FIX 1: FORCE MATPLOTLIB TO HEADLESS BACKEND ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

# =====================================================================
# 1. INITIALIZATION & CLOUD BACKEND BOOTSTRAPPING
# =====================================================================
st.set_page_config(page_title="PhytoRetro Docking Suite", layout="wide", page_icon="🌿")

@st.cache_resource
def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")
                return False
    return True

vina_ready = ensure_linux_vina_exists()

@st.cache_data
def load_database():
    try:
        with open("phyto_retro_db.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Database file 'phyto_retro_db.json' not found. Please ensure it is in your GitHub repository.")
        return {}

PHYTO_RETRO_DB = load_database()

def initialize_session_states():
    defaults = {
        "protein_name": "Unknown Protein", "cx": 0.0, "cy": 0.0, "cz": 0.0,
        "sx": 20, "sy": 20, "sz": 20, "exhaustiveness": 8,
        "target_ready": False, "ligand_ready": False, "local_target_path": None,
        "pdb_id_display": "Custom", "docking_results_raw": None, "redesign_docking_results_raw": None,
        "serialized_ligand_block": None, "ligand_summary_text": "", "smiles_cache": "",
        "baseline_affinity": None, "baseline_pre_uff": "N/A", "baseline_post_uff": "N/A",
        "baseline_delta_uff": "N/A", "redesign_baseline_affinity": None,
        "selected_analog_smiles": None, "selected_analog_name": None, "active_synthesis_protocol": None,
        "uff_cache": {}, "detected_pockets": [], "selected_tree_data": None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_session_states()

def safe_rerun():
    try: st.rerun()
    except AttributeError: st.experimental_rerun()

# =====================================================================
# 2. NAMED REACTION SYNTHESIS ENGINE (Kürti & Czakó Mapping)
# =====================================================================
SYNTHESIS_PROTOCOLS = {
    "methyl": {
        "reaction": "Williamson Ether Synthesis / O-Methylation",
        "reference": "Kürti & Czakó, Strategic Applications of Named Reactions, p. 486",
        "reagents": "NaH (Sodium Hydride), MeI (Methyl Iodide) or Dimethyl Sulfate",
        "conditions": "Anhydrous DMF or THF, 0 °C warming to Room Temperature, 2-4 hours",
        "mechanism": "<strong>1. Deprotonation:</strong> The strong base (NaH) abstracts the acidic proton from the native phytochemical's hydroxyl/phenol group, generating a highly nucleophilic alkoxide/phenoxide anion and evolving hydrogen gas.<br><strong>2. S<sub>N</sub>2 Attack:</strong> The nucleophilic oxygen attacks the electrophilic carbon of Methyl Iodide in a concerted S<sub>N</sub>2 mechanism, displacing the iodide leaving group to forge the new stable ether (O-methyl) bond."
    },
    "acetyl": {
        "reaction": "Steglich Esterification / O-Acetylation",
        "reference": "Kürti & Czakó, Strategic Applications of Named Reactions, p. 428",
        "reagents": "Acetic Anhydride, DMAP (Catalyst), Pyridine or Et3N",
        "conditions": "DCM (Dichloromethane), 0 °C, 1-3 hours",
        "mechanism": "<strong>1. Activation:</strong> DMAP acts as a hyper-nucleophilic catalyst, attacking acetic anhydride to form a highly reactive, resonance-stabilized acylpyridinium intermediate.<br><strong>2. Acyl Transfer:</strong> The native phytochemical's hydroxyl group attacks the acylpyridinium complex, facilitating rapid acyl transfer to form the acetate ester, regenerating the DMAP catalyst and releasing acetic acid as a byproduct."
    },
    "fluoro": {
        "reaction": "Electrophilic Aromatic Fluorination",
        "reference": "Modern Synthetic Analog to Balz-Schiemann (Kürti & Czakó p. 32)",
        "reagents": "Selectfluor (F-TEDA-BF4)",
        "conditions": "Acetonitrile (MeCN), Room Temperature or mild reflux",
        "mechanism": "<strong>1. S<sub>E</sub>Ar Attack:</strong> The electron-rich aromatic ring of the native phytochemical attacks the highly reactive, positively polarized fluorine atom of the Selectfluor reagent, generating a resonance-stabilized Wheland intermediate (sigma complex).<br><strong>2. Rearomatization:</strong> Rapid loss of a proton restores aromaticity, yielding the sterically conservative, metabolically stable fluoro-bioisostere."
    },
    "glucoside": {
        "reaction": "Koenigs-Knorr Glycosylation",
        "reference": "Kürti & Czakó, Strategic Applications of Named Reactions, p. 244",
        "reagents": "Acetobromoglucose (Glycosyl donor), Ag2CO3 (Silver Carbonate promoter)",
        "conditions": "DCM, strictly anhydrous, protected from light, Room Temp",
        "mechanism": "<strong>1. Halophilic Activation:</strong> The Silver(I) salt acts as a halophilic promoter, coordinating with the bromide of the sugar donor. Precipitation of AgBr generates a highly reactive, transient oxocarbenium ion.<br><strong>2. Nucleophilic Trapping:</strong> The phytochemical's nucleophilic oxygen attacks the anomeric center of the oxocarbenium ion to establish the stereoselective beta-glycosidic linkage."
    },
    "amide": {
        "reaction": "Schotten-Baumann Reaction",
        "reference": "Kürti & Czakó, Strategic Applications of Named Reactions, p. 400",
        "reagents": "Acyl Chloride or Acid Anhydride, Aqueous NaOH or Pyridine",
        "conditions": "Biphasic (Aqueous/Organic) mixture, rapid stirring, 0 °C",
        "mechanism": "<strong>1. Addition:</strong> The native phytochemical's amine nitrogen acts as a nucleophile, attacking the highly electrophilic carbonyl carbon of the acid chloride to form a tetrahedral intermediate.<br><strong>2. Elimination:</strong> The tetrahedral intermediate collapses, expelling the chloride leaving group. The base (NaOH or Pyridine) acts as an acid sponge, neutralizing the generated HCl to drive the stable amide bond formation to completion."
    },
    "epox": {
        "reaction": "Prilezhaev Reaction (Epoxidation)",
        "reference": "Kürti & Czakó, Strategic Applications of Named Reactions, p. 364",
        "reagents": "mCPBA (meta-Chloroperoxybenzoic acid)",
        "conditions": "DCM (Dichloromethane), 0 °C, 2-6 hours",
        "mechanism": "<strong>Concerted Transfer:</strong> The reaction proceeds via a highly ordered 'butterfly' transition state. The electrophilic oxygen of the peroxyacid is transferred synchronously to the electron-rich alkene $\pi$-bond of the native scaffold, yielding the oxirane (epoxide) ring while expelling m-chlorobenzoic acid as the byproduct."
    }
}

def get_synthesis_protocol(analog_name):
    name_lower = analog_name.lower()
    for key, protocol in SYNTHESIS_PROTOCOLS.items():
        if key in name_lower:
            return protocol
    return {
        "reaction": "Standard Late-Stage Functionalization (LFS)",
        "reference": "General Organic Synthesis Transformation",
        "reagents": "Standard functional group transfer reagents",
        "conditions": "Solvent-dependent, optimization required",
        "mechanism": "Direct substitution or addition pathway utilizing the native phytochemical's most reactive steric and electronic nodes. Exact mechanistic pathway depends on target bioisosteric functional group topology."
    }

# =====================================================================
# 3. BIOINFORMATICS STRUCTURAL CONVERTERS & PARSERS (Truncated for brevity, kept essential functions)
# =====================================================================
def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try: urllib.request.urlretrieve(url, local_pdb); return True, local_pdb
    except Exception: return False, f"Could not download {pdb_id.upper()}."

def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown", "mw": "N/A", "formula": "N/A"}
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{urllib.parse.quote(smiles_string)}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode())
            props = res["PropertyTable"]["Properties"][0]
            metadata["name"] = props.get("Title", "Target Derivative")
            metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
    except Exception: pass 
    return metadata

def identify_protein_cavities(pdbqt_file):
    coords = []
    if not os.path.exists(pdbqt_file): return []
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if len(coords) < 10: return []
    arr, min_b, max_b = np.array(coords), np.min(coords, axis=0), np.max(coords, axis=0)
    step = (max_b - min_b) / 4.0
    pockets, idx = [], 1
    for i in range(1,4):
        for j in range(1,4):
            for k in range(1,4):
                pt = min_b + np.array([i*step[0], j*step[1], k*step[2]])
                dists = np.linalg.norm(arr - pt, axis=1)
                score = np.sum((dists > 3.0) & (dists < 12.0))
                if np.sum(dists <= 3.0) < 20 and score > 20:
                    pockets.append({"Pocket_ID": f"Cavity {idx}", "cx": round(pt[0], 2), "cy": round(pt[1], 2), "cz": round(pt[2], 2), "bx": 20.0, "by": 20.0, "bz": 20.0, "Score": score})
                    idx += 1
    return sorted(pockets, key=lambda x: x["Score"], reverse=True)[:3]

def compute_protein_bounding_box(pdbqt_file):
    if not os.path.exists(pdbqt_file): return 0, 0, 0, 20, 20, 20
    coords = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try: coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError: pass
    if not coords: return 0, 0, 0, 20, 20, 20
    coords = np.array(coords)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    center = (min_c + max_c) / 2.0
    size = (max_c - min_c) + 15.0
    return center[0], center[1], center[2], min(126.0, size[0]), min(126.0, size[1]), min(126.0, size[2])

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False):
    temp_out = f"temp_{output_pdbqt}"
    try:
        with open(input_pdb, "r", encoding="utf-8", errors="ignore") as pdb, open(temp_out, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    rec = line[:6].strip()
                    res = line[17:20].strip()
                    try: x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    except: continue
                    element = line[76:78].strip()
                    if not element: element = "C"
                    vina_type = element.title()
                    if element == "C" and "AR" in line[12:16].upper(): vina_type = "A"
                    pdbqt.write(f"{rec:<6}    1  C   {res:>3} A   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00    +0.000 {vina_type:<2}\n")
            if is_ligand: pdbqt.write("ENDROOT\nTORSDOF 4\n")
            else: pdbqt.write("ENDMDL\n")
        shutil.move(temp_out, output_pdbqt)
        return True, output_pdbqt
    except Exception as e: return False, str(e)

def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if not mol: return False, "Invalid SMILES."
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if res != 0: res = AllChem.EmbedMolecule(mol, useRandomCoords=True, ignoreSmoothingFailures=True)
        if res != 0: return False, "3D Embedding Failed."
        try: AllChem.MMFFOptimizeMolecule(mol)
        except: pass
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        ok, msg = convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return ok, msg
    except Exception as e: return False, str(e)

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"): current_mode = int(line.split()[1]); current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode: poses[current_mode] = "".join(current_lines)
            else: current_lines.append(line)
    return poses

def get_pose_affinity(stdout_text, idx):
    if not stdout_text: return "N/A"
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
        if m and int(m.group(1)) == idx: return m.group(2)
    return "N/A"

def calculate_advanced_adme(smiles):
    default_adme = {"MW": 0.0, "LogP": 0.0, "HBD": 0, "HBA": 0, "TPSA": 0.0, "Violations": 0, "Lipinski_Obey": "N/A", "Oral_Bio": "N/A", "Permeability": "N/A"}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return default_adme
        mw, logp, hbd, hba, tpsa = Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol)
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        bbb = (tpsa < 79) and (0.4 < logp < 6.0)
        return {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa, "Violations": violations, "Lipinski_Obey": "Yes" if violations <= 1 else "No", "Oral_Bio": "High" if violations == 0 else "Low", "Permeability": "BBB+" if bbb else "Low"}
    except Exception: return default_adme

# =====================================================================
# 4. VISUALIZATION UTILITIES & HTML REPORTING
# =====================================================================
def generate_clean_2d_image(smiles_str, zoom_level=450):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            img = Draw.MolToImage(Chem.RemoveHs(mol), size=(zoom_level, int(zoom_level * 0.77)))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return f'<img src="data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06);"/>'
    except Exception: pass
    return None

def render_advanced_modeling_blueprint(receptor_data, ligand_data, unique_id="container"):
    html_content = f"""
    <div id="wrapper_{unique_id}" style="width:100%;">
        <div id="{unique_id}" style="height: 480px; width: 100%; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer_{unique_id} = $3Dmol.createViewer(document.getElementById('{unique_id}'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{receptor_data}`, 'pdb');
            viewer_{unique_id}.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}});
        }}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{ligand_data}`, 'pdb');
            viewer_{unique_id}.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        viewer_{unique_id}.zoomTo(); viewer_{unique_id}.render();
    </script>
    """
    components.html(html_content, height=510)

def generate_ayurvedic_card(data):
    if not data: return ""
    return f"""
    <div style="background-color: #f4fcf7; padding: 20px; border-radius: 10px; border-left: 6px solid #2e7d32; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 25px;">
        <h2 style="margin-top: 0; color: #1b5e20;">🌳 {data.get('Herb / Tree Name', 'Ayurvedic Plant')}</h2>
        <h4 style="margin: 5px 0; color: #388e3c;"><i>{data.get('Scientific Name', 'N/A')}</i></h4>
        <h5 style="margin: 5px 0; color: #555;"><b>Target Clinical Pathway:</b> {data.get('Medicinal Activity', 'N/A')}</h5>
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0;"><b>Target Protein PDB:</b> {data.get('Target Protein / Receptor Name', 'N/A')}</p>
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0;"><b>Native Ligand (Phytochemical):</b> {data.get('Phytochemical', 'N/A')}</p>
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0; word-break: break-all;"><b>Canonical SMILES:</b> {data.get('Canonical SMILES', 'N/A')}</p>
    </div>
    """

def build_comprehensive_html_report(meta, adme_p, adme_v, analog_name, analog_smiles, p_2d, v_2d, orig_aff, new_aff, syn_protocol):
    
    synthesis_html_block = f"""
    <div style="background-color:#f8fafc; border-left: 5px solid #0369a1; padding:20px; border-radius:6px; margin:20px 0;">
        <h3 style="color:#0369a1; margin-top:0;">Laboratory Synthesis Blueprint (Kürti & Czakó Mapping)</h3>
        <p><strong>Identified Transformation / Named Reaction:</strong> {syn_protocol['reaction']}</p>
        <p><strong>Literature Reference:</strong> <i>{syn_protocol['reference']}</i></p>
        <p><strong>Standard Laboratory Reagents:</strong> {syn_protocol['reagents']}</p>
        <p><strong>Reaction Conditions:</strong> {syn_protocol['conditions']}</p>
        <h4 style="margin-bottom:5px; color:#334155;">Reaction Mechanism Protocol:</h4>
        <p style="font-size:14px; color:#475569; background:#fff; padding:15px; border:1px solid #e2e8f0; border-radius:4px;">
            {syn_protocol['mechanism']}
        </p>
    </div>
    """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PhytoRetro Final Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 40px; background-color: #f9f9fb; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h1, h2, h3 {{ color: #1e3c72; }}
            .structure-box {{ display: flex; gap: 30px; margin: 20px 0; background: #fafafa; padding: 20px; border-radius: 8px; border: 1px solid #eef2f7; align-items: center; justify-content: center; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
            th {{ background-color: #f8fafc; color: #1e3c72; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="border-bottom: 2px solid #eef2f7; padding-bottom: 10px;">🔬 PhytoRetro Complete Execution Report</h1>
            
            <h2>1. Target Receptor & Ligand Profile</h2>
            <p><strong>Protein PDB ID:</strong> {meta['id']}</p>
            <p><strong>Native Phytochemical SMILES:</strong> {meta['smiles']}</p>

            <h2>2. Generative Scaffold Optimization (Phase 2)</h2>
            <p><strong>Isolated Bioisostere Name:</strong> {analog_name}</p>
            <p><strong>Redesigned Target SMILES:</strong> {analog_smiles}</p>
            
            <div class="structure-box">
                <div style="flex:1; text-align: center;"><h4>Original Lead</h4>{p_2d}</div>
                <div style="flex:1; text-align: center;"><h4>Optimized Derivative</h4>{v_2d}</div>
            </div>

            {synthesis_html_block}

            <h2>3. ADMET 3.0 Pharmacokinetics Comparison</h2>
            <table>
                <tr><th>Physiochemical Property</th><th>Original Lead</th><th>Redesigned Analog</th></tr>
                <tr><td>Lipinski Compliance?</td><td>{adme_p['Lipinski_Obey']}</td><td>{adme_v['Lipinski_Obey']}</td></tr>
                <tr><td>Oral Route Usability</td><td>{adme_p['Oral_Bio']}</td><td>{adme_v['Oral_Bio']}</td></tr>
                <tr><td>Permeability Barrier</td><td>{adme_p['Permeability']}</td><td>{adme_v['Permeability']}</td></tr>
                <tr><td>TPSA (Å²)</td><td>{adme_p['TPSA']}</td><td>{adme_v['TPSA']}</td></tr>
                <tr><td>LogP</td><td>{adme_p['LogP']}</td><td>{adme_v['LogP']}</td></tr>
            </table>

            <h2>4. Master Docking Verdict</h2>
            <p><strong>Original Baseline Affinity:</strong> {orig_aff} kcal/mol</p>
            <p><strong>Optimized Derivative Affinity:</strong> {new_aff} kcal/mol</p>
        </div>
    </body>
    </html>
    """

# =====================================================================
# 5. APPLICATION UI
# =====================================================================

st.title("🌿 PhytoRetro Docking Suite")
st.markdown("**Database-Driven Phytochemical Structural Redesign & Molecular Docking Pipeline**")

if not PHYTO_RETRO_DB: st.stop()

# --- PHASE 1: TARGET SELECTION & BASELINE DOCKING ---
st.write("---")
st.header("🔒 Phase 1: Target Matrix Selection & Baseline Docking")

col_params, col_visual = st.columns([1, 1])
trigger_rerun = False

with col_params:
    activity_profile = st.radio("Select Therapeutic Indication Protocol:", options=list(PHYTO_RETRO_DB.keys()), horizontal=True)
    available_plants = list(PHYTO_RETRO_DB.get(activity_profile, {}).keys())
    selected_plant = st.selectbox("Select Target Ayurvedic Monograph Matrix:", options=available_plants)

    if st.button("📥 Load Target Structure & Phytochemical", type="primary"):
        plant_data = PHYTO_RETRO_DB[activity_profile][selected_plant]
        st.session_state.pdb_id_display = plant_data["receptor_id"]
        st.session_state.smiles_cache = plant_data["native_ligand_smiles"]
        st.session_state.protein_name = plant_data['phytochemical'] + " Target"
        
        st.session_state.selected_tree_data = {
            'Herb / Tree Name': selected_plant, 'Scientific Name': plant_data['scientific_name'],
            'Medicinal Activity': activity_profile, 'Target Protein / Receptor Name': st.session_state.pdb_id_display,
            'Phytochemical': plant_data['phytochemical'], 'Canonical SMILES': st.session_state.smiles_cache
        }

        with st.spinner("Downloading RCSB Receptor..."):
            succ, path = fetch_pdb_from_rcsb(st.session_state.pdb_id_display)
            if succ:
                st.session_state.local_target_path = path
                st.session_state.target_ready, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")

        with st.spinner("Embedding Ligand 3D Topology..."):
            st.session_state.ligand_ready, _ = convert_smiles_to_pdbqt(st.session_state.smiles_cache, "ligand.pdbqt")
            if st.session_state.ligand_ready:
                with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()

        if st.session_state.target_ready and st.session_state.ligand_ready:
            st.success("Pipeline initialized!")
            trigger_rerun = True

    if st.session_state.get('selected_tree_data'):
        st.markdown(generate_ayurvedic_card(st.session_state.selected_tree_data), unsafe_allow_html=True)

    st.subheader("Smart Cavity Configurator")
    if st.button("🌐 Enable Blind Docking (Full Protein Surface)", use_container_width=True):
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            c_x, c_y, c_z, s_x, s_y, s_z = compute_protein_bounding_box("protein.pdbqt")
            st.session_state.cx, st.session_state.cy, st.session_state.cz = round(c_x,1), round(c_y,1), round(c_z,1)
            st.session_state.sx, st.session_state.sy, st.session_state.sz = int(s_x), int(s_y), int(s_z)
            trigger_rerun = True

    grid_cx = st.number_input("Center X", value=float(st.session_state.cx), step=1.0)
    grid_cy = st.number_input("Center Y", value=float(st.session_state.cy), step=1.0)
    grid_cz = st.number_input("Center Z", value=float(st.session_state.cz), step=1.0)
    grid_sx = st.slider("Size X", 10, 126, int(st.session_state.sx))
    grid_sy = st.slider("Size Y", 10, 126, int(st.session_state.sy))
    grid_sz = st.slider("Size Z", 10, 126, int(st.session_state.sz))
    
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Initialize Baseline Docking", type="primary", disabled=not can_dock)

with col_visual:
    st.header("Active Viewport Canvas")
    if st.session_state.docking_results_raw is None:
        rec_view = ""
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            with open("protein.pdbqt", "r") as f: rec_view = f.read()
        render_advanced_modeling_blueprint(rec_view, st.session_state.serialized_ligand_block, unique_id="v_phase1")
    else:
        st.subheader("Baseline Docking Results")
        if os.path.exists("docking_poses.pdbqt"):
            poses = split_docking_poses("docking_poses.pdbqt")
            if poses:
                aff = get_pose_affinity(st.session_state.docking_results_raw, 1)
                st.session_state.baseline_affinity = aff
                
                try:
                    if float(aff) > 0: st.error("🚨 **WARNING:** Positive binding energy detected. Molecule is experiencing steric clashes. Expand the Grid Box.")
                except: pass

                st.html(f"""
                <div style="background-color:#f0f7f4; border-left:6px solid #2e7d32; padding:16px; border-radius:8px; margin-bottom:15px;">
                    <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold;">Baseline Pose Affinity</span><br>
                    <span style="font-size:36px; font-weight:900; color:#1b5e20;">{aff} <span style="font-size:18px; font-weight:normal;">kcal/mol</span></span>
                </div>
                """)
                with open("protein.pdbqt", "r") as f: p_data = f.read()
                render_advanced_modeling_blueprint(p_data, poses[1], unique_id="p1_res")

if run_btn and can_dock:
    vina_path = os.path.abspath("vina")
    cmd = [vina_path, "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
           "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
           "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
           "--exhaustiveness", "8", "--out", "docking_poses.pdbqt"]
    p_bar = st.progress(0, text="Docking...")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out_log, p_count = [], 0
        while True:
            char = proc.stdout.read(1).decode("utf-8", errors="ignore")
            if not char: break
            out_log.append(char)
            if char == '*': p_count += 1; p_bar.progress(min(100, int((p_count/50)*100)))
        proc.wait()
        if proc.returncode == 0:
            st.session_state.docking_results_raw = "".join(out_log)
            trigger_rerun = True
        else: st.error("Docking failed.")
    except Exception as e: st.error(str(e))

# --- PHASE 2: REDESIGN STUDIO ---
st.write("---")
st.header("🧬 Phase 2: Database-Driven Scaffold Customization Studio")

if st.session_state.get('selected_tree_data'):
    act = st.session_state.selected_tree_data['Medicinal Activity']
    plnt = st.session_state.selected_tree_data['Herb / Tree Name']
    analogs = PHYTO_RETRO_DB[act][plnt].get("analogs", {})
    
    col_rd_p, col_rd_v = st.columns([1, 1])
    
    with col_rd_p:
        sel_analog = st.selectbox("Select Pre-Validated Synthetic Bioisostere / Analog:", options=list(analogs.keys()))
        if sel_analog:
            st.session_state.selected_analog_name = sel_analog
            st.session_state.selected_analog_smiles = analogs[sel_analog]
            st.success(f"Analog Locked: {sel_analog}")
            
            # --- NEW: NAMED REACTION INJECTION (Kürti & Czakó) ---
            syn_protocol = get_synthesis_protocol(sel_analog)
            st.session_state.active_synthesis_protocol = syn_protocol
            
            st.markdown(f"""
            <div style="background-color:#f8fafc; border-left: 5px solid #0369a1; padding:20px; border-radius:6px; margin-top:20px;">
                <h4 style="color:#0369a1; margin-top:0;">🧪 Laboratory Synthesis Protocol</h4>
                <p><strong>Identified Named Reaction:</strong> {syn_protocol['reaction']}</p>
                <p><strong>Textbook Ref:</strong> <i>{syn_protocol['reference']}</i></p>
                <p><strong>Reagents:</strong> {syn_protocol['reagents']}</p>
                <p><strong>Conditions:</strong> {syn_protocol['conditions']}</p>
                <hr style="border:0; border-top:1px solid #cbd5e1; margin:10px 0;">
                <p style="font-size:14px; color:#475569;"><strong>Reaction Mechanism:</strong><br>{syn_protocol['mechanism']}</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_rd_v:
        if st.session_state.get('selected_analog_smiles'):
            img_html = generate_clean_2d_image(st.session_state.selected_analog_smiles, zoom_level=500)
            if img_html: st.markdown(img_html, unsafe_allow_html=True)
else:
    st.warning("Select a target in Phase 1.")

# --- PHASE 3 & 4: ADME & VALIDATION ---
st.write("---")
st.header("🎯 Phase 3 & 4: Pharmacokinetics & Validation Docking")

if st.session_state.get('selected_analog_smiles') and st.session_state.docking_results_raw:
    col_p4_1, col_p4_2 = st.columns([1, 1])
    
    with col_p4_1:
        if st.button("🔄 Embed Analog to 3D Coordinates", type="secondary"):
            with st.spinner("Converting..."):
                ok, msg = convert_smiles_to_pdbqt(st.session_state.selected_analog_smiles, "redesign_ligand.pdbqt")
                if ok: st.success("3D Structure Built!")
                else: st.error(msg)
                    
    with col_p4_2:
        can_run_p4 = os.path.exists("protein.pdbqt") and os.path.exists("redesign_ligand.pdbqt")
        if st.button("🚀 Execute Validation Docking", type="primary", disabled=not can_run_p4):
            vina_path = os.path.abspath("vina")
            cmd = [vina_path, "--receptor", "protein.pdbqt", "--ligand", "redesign_ligand.pdbqt", 
                   "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
                   "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
                   "--exhaustiveness", "8", "--out", "redesign_docking_poses.pdbqt"]
            p_bar = st.progress(0, text="Validating...")
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out_log, p_count = [], 0
                while True:
                    char = proc.stdout.read(1).decode("utf-8", errors="ignore")
                    if not char: break
                    out_log.append(char)
                    if char == '*': p_count += 1; p_bar.progress(min(100, int((p_count/50)*100)))
                proc.wait()
                if proc.returncode == 0:
                    st.session_state.redesign_docking_results_raw = "".join(out_log)
                    trigger_rerun = True
            except Exception as e: st.error(str(e))

    if st.session_state.redesign_docking_results_raw and os.path.exists("redesign_docking_poses.pdbqt"):
        st.write("---")
        
        p4_poses = split_docking_poses("redesign_docking_poses.pdbqt")
        orig_pose = split_docking_poses("docking_poses.pdbqt").get(1, "")
        new_aff_str = get_pose_affinity(st.session_state.redesign_docking_results_raw, 1)
        orig_aff_str = st.session_state.baseline_affinity

        with open("protein.pdbqt", "r") as f: p_data = f.read()
        
        col_3d_1, col_3d_2 = st.columns(2)
        with col_3d_1:
            st.markdown(f"#### Original Lead ({orig_aff_str} kcal/mol)")
            render_advanced_modeling_blueprint(p_data, orig_pose, unique_id="p4_o")
        with col_3d_2:
            st.markdown(f"#### Redesigned Derivative ({new_aff_str} kcal/mol)")
            render_advanced_modeling_blueprint(p_data, p4_poses[1], unique_id="p4_n")

        try: delta_aff = round(float(new_aff_str) - float(orig_aff_str), 2)
        except: delta_aff = 0.0
        
        master_verdict = f"🟢 Improved by **{delta_aff} kcal/mol**." if delta_aff < 0 else f"🔴 Worsened by **+{delta_aff} kcal/mol**."
        st.info(f"**Synthesis Verdict:** {master_verdict}")
        
        # --- HTML EXPORT WITH SYNTHESIS DATA ---
        adme_p = calculate_advanced_adme(st.session_state.smiles_cache)
        adme_v = calculate_advanced_adme(st.session_state.selected_analog_smiles)
        p_2d = generate_clean_2d_image(st.session_state.smiles_cache, zoom_level=300)
        v_2d = generate_clean_2d_image(st.session_state.selected_analog_smiles, zoom_level=300)
        
        meta = {'id': st.session_state.pdb_id_display, 'smiles': st.session_state.smiles_cache}
        
        html_report = build_comprehensive_html_report(
            meta, adme_p, adme_v, st.session_state.selected_analog_name, 
            st.session_state.selected_analog_smiles, p_2d, v_2d, orig_aff_str, new_aff_str, st.session_state.active_synthesis_protocol
        )
        
        st.download_button("📥 Download Master HTML Report (with Synthesis Data)", data=html_report, file_name="PhytoRetro_Synthesis_Report.html", mime="text/html", use_container_width=True)

if trigger_rerun: safe_rerun()
