import os
import re

# --- CONFIGURATION ---
XENIUM_DIR = '/DATA1/Roxana/Hierarchy/data/xenium/' 

def get_xenium_metadata(xenium_folder_path):
    metadata_summary = []
    items = os.listdir(xenium_folder_path)
    xenium_folders = [item for item in items if os.path.isdir(os.path.join(xenium_folder_path, item))]
    
    print(f"Found {len(xenium_folders)} potential Xenium folders in {XENIUM_DIR}\n")
    
    for folder_name in xenium_folders:
        # Clean up and normalize the name
        sample_name = folder_name.replace('_outs', '').replace('_out', '').lower()
        
        # --- INITIALIZATION ---
        species = 'Unknown'
        disease = 'N/A'
        tissue = 'N/A'
        
        # --- 1. SPECIES PARSING (Retained for sanity check) ---
        human_indicators = ['human', 'hcolon', 'hlung', 'hpancreas', 'hkidney', 'htonsil', 'hskin', 'hbonemarrow', 'hbone', 'hheart', 'hlymphnode', 'hliver']
        mouse_indicators = ['mouse', 'mfemur', 'mbrain', 'tgcrnd8', 'pup', 'mouse_colon']

        if any(p in sample_name for p in human_indicators):
            species = 'Human'
        elif any(p in sample_name for p in mouse_indicators):
            species = 'Mouse'

        # --- 3. TISSUE PARSING (Retained for display) ---
        if 'brain' in sample_name or 'glioblastoma' in sample_name or 'alzheimers' in sample_name:
            tissue = 'Brain'
        elif 'lung' in sample_name:
            tissue = 'Lung'
        elif 'colon' in sample_name or 'colorectal' in sample_name or 'crc' in sample_name:
            tissue = 'Colon'
        elif 'kidney' in sample_name:
            tissue = 'Kidney'
        elif 'pancreas' in sample_name:
            tissue = 'Pancreas'
        elif 'liver' in sample_name:
            tissue = 'Liver'
        elif 'breast' in sample_name or 'idc' in sample_name or 'ilc' in sample_name or 'ductal_adenocarcinoma' in sample_name:
            tissue = 'Breast'
        elif 'femur' in sample_name or 'bonemarrow' in sample_name or 'bone' in sample_name:
            tissue = 'Bone/Marrow'
        elif 'skin' in sample_name or 'melanoma' in sample_name:
            tissue = 'Skin'
        elif 'tonsil' in sample_name or 'lymphnode' in sample_name or 'lymph_node' in sample_name:
            tissue = 'Lymphoid/Immune'
        elif 'heart' in sample_name:
            tissue = 'Heart'
        elif 'prostate' in sample_name:
            tissue = 'Prostate'
        elif 'ovarian' in sample_name:
            tissue = 'Ovary'

        # --- 2. SIMPLIFIED DISEASE/NORMAL CLASSIFICATION (The requested change) ---
        
        # Keywords indicating disease, cancer model, or active pathology
        diseased_keywords = ['cancer', 'diseased', 'melanoma', 'gbm', 'alzheimers', 'leukemia', 'tgcrnd8', 'idc', 'ilc', 'crc', 'ductal_adenocarcinoma', 'hyperplasia', 'reactive']
        
        # If any major disease keyword is present, classify as Diseased
        if any(p in sample_name for p in diseased_keywords):
            disease = 'Diseased'
        
        # If the sample is not explicitly Diseased, classify it as Normal/Control
        else:
            disease = 'Normal/Control'
        
        # --- Add Summary ---
        metadata_summary.append({
            'Folder_Name': folder_name,
            'Species': species,
            'Tissue': tissue,
            'Disease_Status': disease # This will be either 'Diseased' or 'Normal/Control'
        })

    return metadata_summary

if __name__ == "__main__":
    summary_data = get_xenium_metadata(XENIUM_DIR)
    
    if summary_data:
        print("--- XENIUM DATA METADATA SUMMARY (Diseased/Normal Only) ---")
        for item in summary_data:
            print(f"\n📁 Folder: {item['Folder_Name']}")
            print(f"  🔬 Species: {item['Species']}")
            print(f"  🧬 Tissue: {item['Tissue']}")
            print(f"  🦠 Status: {item['Disease_Status']}")
        print("\n------------------------------------")
    else:
        print(f"No Xenium folders found in {XENIUM_DIR} or path is incorrect.")