import os
import re

# --- CONFIGURATION ---
# MERSCOPE_DIR = '/path/to/your/merscope/data/' 
# Using a placeholder variable here as the user's path is already in the runfile command

def get_merscope_metadata(merscope_folder_path):
    """
    Scans a directory and infers metadata (species, disease, tissue) from 
    the folder name using enhanced parsing rules for common Merfish cancer datasets.
    """
    metadata_summary = []
    
    if not os.path.exists(merscope_folder_path):
        print(f"Error: Directory not found at {merscope_folder_path}")
        return []

    items = os.listdir(merscope_folder_path)
    merscope_folders = [item for item in items if os.path.isdir(os.path.join(merscope_folder_path, item))]
    
    print(f"Found {len(merscope_folders)} potential Merscope folders in {merscope_folder_path}\n")
    
    for folder_name in merscope_folders:
        # Normalize the name for parsing
        sample_name = folder_name.lower().replace('_', '').replace('-', '')
        
        # --- INITIALIZATION ---
        species = 'Unknown'
        disease = 'N/A'
        tissue = 'N/A'
        
        # --- 1. SPECIES PARSING (Simplifed as all are 'Human') ---
        if 'human' in sample_name or re.search(r'\bhuman[a-z]{2,10}cancer', sample_name):
            species = 'Human'
        elif 'mouse' in sample_name or re.search(r'\bm[a-z]{2,5}\b', sample_name):
            species = 'Mouse'

        # --- 2. TISSUE PARSING (ENHANCED for Cancer Names) ---
        if 'brain' in sample_name or 'cortex' in sample_name or 'pfc' in sample_name:
            tissue = 'Brain/CNS'
        elif 'breast' in sample_name:
            tissue = 'Breast'
        elif 'colon' in sample_name:
            tissue = 'Gastrointestinal'
        elif 'liver' in sample_name:
            tissue = 'Liver'
        elif 'lung' in sample_name:
            tissue = 'Lung'
            
        # --- FIX: Specific Cancer Tissue Names ---
        elif 'melanoma' in sample_name:
            tissue = 'Skin'
        elif 'uterine' in sample_name:
            tissue = 'Uterus'
        elif 'ovarian' in sample_name:
            tissue = 'Ovary'
        elif 'prostate' in sample_name:
            tissue = 'Prostate'
        elif 'kidney' in sample_name:
            tissue = 'Kidney'
        # --- End FIX ---

        # --- 3. DISEASE/STATUS PARSING (Mostly 'Diseased/Cancer' for this set) ---
        if any(p in sample_name for p in ['cancer', 'tumor', 'melanoma', 'diseased']):
            disease = 'Diseased'
        elif any(p in sample_name for p in ['normal', 'control', 'wildtype', 'wt']):
            disease = 'Normal/Control'
        
        # --- Add Summary ---
        metadata_summary.append({
            'Folder_Name': folder_name,
            'Species': species,
            'Tissue': tissue,
            'Disease_Status': disease
        })

    return metadata_summary

if __name__ == "__main__":
    # Note: Replace this with the user's actual path if running outside the runfile context
    MERSCOPE_DIR = '/DATA1/Roxana/Hierarchy/data/merscope/' 
    summary_data = get_merscope_metadata(MERSCOPE_DIR)
    
    if summary_data:
        print("--- MERSCOPE DATA METADATA SUMMARY (Finalized) ---")
        for item in summary_data:
            print(f"\n📁 Folder: {item['Folder_Name']}")
            print(f"  🔬 Species: {item['Species']}")
            print(f"  🧬 Tissue: {item['Tissue']}")
            print(f"  🦠 Status: {item['Disease_Status']}")
        print("\n------------------------------------")
    else:
        print(f"No Merscope folders found in {MERSCOPE_DIR} or path is incorrect.")