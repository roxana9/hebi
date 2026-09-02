#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:18:23 2025

@author: tower4
"""

import os
import re

# --- CONFIGURATION ---
# *** IMPORTANT: SET YOUR COS-MX BASE DIRECTORY HERE ***
# Using the standard path we've been working with:
COS_MX_BASE_PATH = "/DATA1/Roxana/Hierarchy/data/cosmx/"

def get_cosmx_metadata_from_folder_names(base_path):
    """
    Scans the directory for folders and infers metadata (Species, Tissue, Disease) 
    based on keywords in the folder names.
    """
    metadata_summary = []
    
    if not os.path.exists(base_path):
        print(f"Error: Directory not found at {base_path}")
        return []

    items = os.listdir(base_path)
    # Filter for directories (assuming each sample is in its own folder)
    cosmx_folders = [item for item in items if os.path.isdir(os.path.join(base_path, item))]
    
    print(f"Found {len(cosmx_folders)} potential CosMx folders in {base_path}\n")
    
    for folder_name in cosmx_folders:
        # Normalize the name for robust parsing
        sample_name = folder_name.lower().replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # --- INITIALIZATION ---
        species = 'Unknown'
        tissue = 'N/A'
        disease = 'N/A'
        
        # --- 1. SPECIES PARSING ---
        if 'human' in sample_name or 'h ' in sample_name.split():
            species = 'Human'
        elif 'mouse' in sample_name or 'm ' in sample_name.split():
            species = 'Mouse'
            
        # --- 2. TISSUE PARSING ---
        if 'liver' in sample_name:
            tissue = 'Liver'
        elif 'pancreas' in sample_name:
            tissue = 'Pancreas'
        elif 'kidney' in sample_name:
            tissue = 'Kidney'
        elif 'brain' in sample_name:
            tissue = 'Brain'

        # --- 3. DISEASE/NORMAL CLASSIFICATION ---
        
        # Keywords indicating disease, cancer model, or active pathology
        diseased_keywords = ['cancer', 'tumor', 'diseased', 'pathology', 'melanoma']
        
        # Check for normal/control FIRST (higher precedence)
        if any(p in sample_name for p in ['normal', 'control', 'wildtype', 'nondiseased']):
            disease = 'Normal'
        
        # If not explicitly normal, check for disease status
        elif any(p in sample_name for p in diseased_keywords):
            disease = 'Diseased'
        
        # If no explicit label, check if species/tissue implies a general control status
        elif species != 'Unknown' and disease == 'N/A':
            disease = 'Normal'


        # --- Add Summary ---
        metadata_summary.append({
            'Folder_Name': folder_name,
            'Species': species,
            'Tissue': tissue,
            'Disease_Status': disease
        })

    return metadata_summary

if __name__ == "__main__":
    summary_data = get_cosmx_metadata_from_folder_names(COS_MX_BASE_PATH)
    
    if summary_data:
        print("\n--- COS-MX DATA METADATA SUMMARY (Dynamic Scan) ---")
        for item in summary_data:
            print(f"\n📁 Folder: {item['Folder_Name']}")
            print(f"  🔬 Species: {item['Species']}")
            print(f"  🧬 Tissue: {item['Tissue']}")
            print(f"  🦠 Status: {item['Disease_Status']}")
        print("\n------------------------------------")
    else:
        print(f"No CosMx folders found at {COS_MX_BASE_PATH}.")