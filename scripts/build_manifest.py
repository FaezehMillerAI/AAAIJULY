import json
import argparse
from pathlib import Path
import sys
import re
import os
import pandas as pd
import random

# Ensure workspace is on PATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.manifest import save_manifest, generate_mock_manifest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="indiana", choices=["indiana", "mimic"])
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/rezakurniawan27/iu-xray/iu_xray")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()

def extract_indication(report_text):
    if not report_text:
        return ""
    match = re.search(r"indication:\s*(.*?)\.\s", report_text.lower())
    if match:
        return match.group(1).strip()
    return ""

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mock:
        print("Mock mode specified. Generating mock manifest...")
        manifest_path = generate_mock_manifest(
            output_dir=out_dir,
            image_dir=out_dir / "mock_images",
            num_train=16,
            num_val=4,
            num_test=8
        )
        print(f"Mock manifest created at {manifest_path}")
        return
        
    examples = []
    
    if args.dataset == "indiana":
        # Search for CSV structures first (as in adr-test-iu.ipynb)
        reports_csv = data_dir / "indiana_reports.csv"
        projections_csv = data_dir / "indiana_projections.csv"
        
        # Check alternative locations
        search_dirs = [
            data_dir,
            Path("/kaggle/input/chest-xrays-indiana-university"),
            Path("/kaggle/input/datasets/raddar/chest-xrays-indiana-university")
        ]
        
        for sd in search_dirs:
            if (sd / "indiana_reports.csv").exists():
                reports_csv = sd / "indiana_reports.csv"
                projections_csv = sd / "indiana_projections.csv"
                data_dir = sd
                break
                
        if reports_csv.exists() and projections_csv.exists():
            print(f"Found Indiana reports and projections CSVs in {data_dir}. Building manifest...")
            rep_df = pd.read_csv(reports_csv)
            proj_df = pd.read_csv(projections_csv)
            
            # Merge and filter for frontal projection
            df = pd.merge(proj_df, rep_df, on="uid", how="left")
            df = df[df["projection"].str.lower().eq("frontal")].copy()
            
            for c in ["findings", "impression"]:
                df[c] = df[c].fillna("").astype(str)
                
            records = df.to_dict(orient="records")
            
            # Setup random seed for splitting
            random.seed(1234)
            random.shuffle(records)
            
            total = len(records)
            train_end = int(total * 0.8)
            val_end = int(total * 0.9)
            
            for idx, row in enumerate(records):
                study_id = f"s{row['uid']}"
                filename = row["filename"]
                
                # Check normalized image paths
                img_paths_to_try = [
                    data_dir / "images" / "images_normalized" / filename,
                    data_dir / "images" / filename,
                    data_dir / filename
                ]
                image_path = img_paths_to_try[0]
                for p in img_paths_to_try:
                    if p.exists():
                        image_path = p
                        break
                        
                # Use only Findings — exclude Impression (per user preference)
                findings_text = row["findings"].strip()
                report = findings_text if findings_text else row["impression"].strip()
                indication = extract_indication(row["impression"]) or extract_indication(row["findings"])
                
                if idx < train_end:
                    split = "train"
                elif idx < val_end:
                    split = "val"
                else:
                    split = "test"
                    
                examples.append({
                    "study_id": study_id,
                    "image_path": str(image_path.absolute()),
                    "report": report,
                    "indication": indication if indication else "radiology evaluation",
                    "split": split,
                    "metadata": {"source": "indiana_csv"}
                })
                
        else:
            # Fallback to annotation.json structure if CSVs not found
            annot_path = data_dir / "annotation.json"
            if not annot_path.exists():
                print(f"Dataset path {data_dir} does not contain annotation.json or CSVs. Generating mock fallback...")
                main_mock(out_dir)
                return
                
            print(f"Found annotation.json at {annot_path}. Building manifest...")
            with open(annot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            studies = data if isinstance(data, list) else []
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        studies = v
                        break
            
            for study in studies:
                study_id = str(study.get("id", ""))
                report = study.get("report", "")
                split = study.get("split", "train")
                indication = study.get("indication", "")
                if not indication:
                    indication = extract_indication(report)
                    
                images = study.get("image_path", [])
                if isinstance(images, str):
                    images = [images]
                    
                for img in images:
                    # Robust path resolution search
                    img_paths_to_try = [
                        data_dir / img,
                        data_dir / "images" / img,
                        data_dir / "images" / "images_normalized" / img,
                        data_dir.parent / img,
                        data_dir.parent / "images" / img,
                        data_dir.parent / "images" / "images_normalized" / img
                    ]
                    img_path = img_paths_to_try[0]
                    for p in img_paths_to_try:
                        if p.exists():
                            img_path = p
                            break
                            
                    examples.append({
                        "study_id": study_id,
                        "image_path": str(img_path.absolute()),
                        "report": report,
                        "indication": indication if indication else "radiology evaluation",
                        "split": split,
                        "metadata": {"source": "indiana_json"}
                    })
                    
    elif args.dataset == "mimic":
        print(f"Searching MIMIC-CXR files in {data_dir}...")
        import ast
        
        # Locate files directory
        files_dir = None
        candidate_files_dirs = [
            data_dir / "official_data_iccv_final" / "files",
            data_dir / "official_data_iccv_final" / "official_data_iccv_final" / "files",
            data_dir / "files"
        ]
        for c in candidate_files_dirs:
            if c.exists() and c.is_dir():
                files_dir = c
                break
                
        # Locate CSV files
        train_csv = None
        val_csv = None
        candidate_csv_paths = [
            (data_dir / "mimic_cxr_aug_train.csv", data_dir / "mimic_cxr_aug_validate.csv"),
            (data_dir / "official_data_iccv_final" / "mimic_cxr_aug_train.csv", data_dir / "official_data_iccv_final" / "mimic_cxr_aug_validate.csv"),
            (data_dir / "official_data_iccv_final" / "official_data_iccv_final" / "mimic_cxr_aug_train.csv", data_dir / "official_data_iccv_final" / "official_data_iccv_final" / "mimic_cxr_aug_validate.csv")
        ]
        for tc, vc in candidate_csv_paths:
            if tc.exists() and vc.exists():
                train_csv = tc
                val_csv = vc
                break
                
        if not train_csv or not files_dir:
            raise FileNotFoundError(f"Could not locate MIMIC-CXR CSV files or image directory in {data_dir}")
            
        print(f"Loading MIMIC-CXR data from {train_csv.name} and {val_csv.name}...")
        
        def safe_parse_list(val):
            if not val or pd.isna(val):
                return []
            if isinstance(val, list):
                return val
            try:
                return ast.literal_eval(val)
            except Exception:
                return []
                
        train_examples = []
        val_test_examples = []
        
        # Load and parse training set
        train_df = pd.read_csv(train_csv)
        print(f"Parsing {len(train_df)} training patient rows...")
        for _, row in train_df.iterrows():
            images_list = safe_parse_list(row.get("image", "[]"))
            reports_list = safe_parse_list(row.get("text", "[]"))
            
            studies_in_row = []
            study_images = {}
            for img_path in images_list:
                parts = Path(img_path).parts
                study_id = None
                for part in parts:
                    if part.startswith("s") and part[1:].isdigit():
                        study_id = part
                        break
                if not study_id:
                    study_id = parts[-2] if len(parts) >= 2 else "unknown"
                if study_id not in study_images:
                    study_images[study_id] = []
                    studies_in_row.append(study_id)
                study_images[study_id].append(img_path)
                
            num_matches = min(len(studies_in_row), len(reports_list))
            for i in range(num_matches):
                study_id = studies_in_row[i]
                report_text = reports_list[i]
                img_rel_path = study_images[study_id][0]
                img_abs_path = files_dir.parent / img_rel_path
                indication = extract_indication(report_text)
                
                train_examples.append({
                    "study_id": study_id,
                    "image_path": str(img_abs_path.absolute()),
                    "report": report_text.strip(),
                    "indication": indication if indication else "radiology evaluation",
                    "split": "train",
                    "metadata": {"source": "mimic_csv"}
                })
                
        # Load and parse validation set
        val_df = pd.read_csv(val_csv)
        print(f"Parsing {len(val_df)} validation patient rows...")
        for _, row in val_df.iterrows():
            images_list = safe_parse_list(row.get("image", "[]"))
            reports_list = safe_parse_list(row.get("text", "[]"))
            
            studies_in_row = []
            study_images = {}
            for img_path in images_list:
                parts = Path(img_path).parts
                study_id = None
                for part in parts:
                    if part.startswith("s") and part[1:].isdigit():
                        study_id = part
                        break
                if not study_id:
                    study_id = parts[-2] if len(parts) >= 2 else "unknown"
                if study_id not in study_images:
                    study_images[study_id] = []
                    studies_in_row.append(study_id)
                study_images[study_id].append(img_path)
                
            num_matches = min(len(studies_in_row), len(reports_list))
            for i in range(num_matches):
                study_id = studies_in_row[i]
                report_text = reports_list[i]
                img_rel_path = study_images[study_id][0]
                img_abs_path = files_dir.parent / img_rel_path
                indication = extract_indication(report_text)
                
                val_test_examples.append({
                    "study_id": study_id,
                    "image_path": str(img_abs_path.absolute()),
                    "report": report_text.strip(),
                    "indication": indication if indication else "radiology evaluation",
                    "split": "val",  # placeholder
                    "metadata": {"source": "mimic_csv"}
                })
                
        # Shuffling and Capping
        random.seed(1234)
        random.shuffle(train_examples)
        random.shuffle(val_test_examples)
        
        train_examples = train_examples[:8000]
        
        # Split val_test_examples into 50% val and 50% test
        num_val_test = len(val_test_examples)
        val_size = min(1000, num_val_test // 2)
        test_size = min(1000, num_val_test - val_size)
        
        val_examples = val_test_examples[:val_size]
        test_examples = val_test_examples[val_size : val_size + test_size]
        
        for ex in val_examples:
            ex["split"] = "val"
        for ex in test_examples:
            ex["split"] = "test"
            
        examples.extend(train_examples)
        examples.extend(val_examples)
        examples.extend(test_examples)
        
        print(f"MIMIC Parsing Complete: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test examples selected.")

    if not examples:
        print("No examples found. Creating mock fallback...")
        main_mock(out_dir)
        return
        
    manifest_path = out_dir / "common_manifest.jsonl"
    save_manifest(examples, manifest_path)
    print(f"Built common manifest at {manifest_path} with {len(examples)} entries.")

def main_mock(out_dir):
    manifest_path = generate_mock_manifest(
        output_dir=out_dir,
        image_dir=out_dir / "mock_images",
        num_train=16,
        num_val=4,
        num_test=8
    )
    print(f"Mock manifest created at {manifest_path}")

if __name__ == "__main__":
    main()
