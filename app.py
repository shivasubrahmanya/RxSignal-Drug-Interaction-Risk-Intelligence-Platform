import os
import pandas as pd

# Base path
base_path = r"C:\Users\shiva\Desktop\FAERS"

quarters = [
    "faers_ascii_2025q1",
    "faers_ascii_2025q2",
    "faers_ascii_2025q3",
    "faers_ascii_2025q4"
]

def process_quarter(base_path, quarter_name):
    path = os.path.join(base_path, quarter_name, "ASCII")

    print(f"\nReading files from: {path}")

    demo_file = [f for f in os.listdir(path) if f.startswith("DEMO") and f.endswith(".txt")][0]
    drug_file = [f for f in os.listdir(path) if f.startswith("DRUG") and f.endswith(".txt")][0]
    reac_file = [f for f in os.listdir(path) if f.startswith("REAC") and f.endswith(".txt")][0]

    # Read with correct separator
    demo = pd.read_csv(os.path.join(path, demo_file), sep='$', engine='python', encoding='latin1', on_bad_lines='skip')
    drug = pd.read_csv(os.path.join(path, drug_file), sep='$', engine='python', encoding='latin1', on_bad_lines='skip')
    reac = pd.read_csv(os.path.join(path, reac_file), sep='$', engine='python', encoding='latin1', on_bad_lines='skip')

    print("Files loaded")

    # Normalize column names
    demo.columns = demo.columns.str.upper()
    drug.columns = drug.columns.str.upper()
    reac.columns = reac.columns.str.upper()

    # DEBUG (optional)
    print("Columns OK")

    # ❌ REMOVED ROLE_COD FILTER (IMPORTANT FIX)

    # Keep required columns
    demo = demo[["PRIMARYID", "AGE", "SEX"]]
    drug = drug[["PRIMARYID", "DRUGNAME"]]
    reac = reac[["PRIMARYID", "PT"]]

    # Normalize drug names
    drug["DRUGNAME"] = drug["DRUGNAME"].astype(str).str.upper().str.strip()
    
    # Get patients with at least 2 unique drugs
    drug = drug.drop_duplicates(subset=["PRIMARYID", "DRUGNAME"])
    drug_counts = drug.groupby("PRIMARYID").size()
    valid_pids = drug_counts[drug_counts >= 2].index
    drug = drug[drug["PRIMARYID"].isin(valid_pids)]
    
    if drug.empty:
        print(f"{quarter_name} processed. Rows: 0")
        return pd.DataFrame(columns=["drug_a", "drug_b", "event", "age", "gender"])

    # We will return the prepared dataframes and do chunked processing in the main loop
    return drug, reac, demo, valid_pids

# Output folder
output_path = r"C:\Users\shiva\Desktop\FAERS\processed"
os.makedirs(output_path, exist_ok=True)


# Run processing
for q in quarters:
    print(f"\nProcessing {q}...")
    result = process_quarter(base_path, q)
    out_csv = os.path.join(output_path, f"{q}.csv")
    
    if isinstance(result, pd.DataFrame):
        result.to_csv(out_csv, index=False)
        continue
        
    drug, reac, demo, valid_pids = result
    
    # Process in chunks to avoid MemoryError (280M+ rows possible)
    chunk_size = 1000
    total_rows = 0
    
    # Empty existing file or create with header
    pd.DataFrame(columns=["drug_a", "drug_b", "event", "age", "gender"]).to_csv(out_csv, index=False)
    
    for i in range(0, len(valid_pids), chunk_size):
        chunk_pids = valid_pids[i:i+chunk_size]
        
        d_chunk = drug[drug["PRIMARYID"].isin(chunk_pids)]
        r_chunk = reac[reac["PRIMARYID"].isin(chunk_pids)]
        demo_chunk = demo[demo["PRIMARYID"].isin(chunk_pids)]
        
        # Self-merge to get all combinations, then filter to keep unique pairs (A < B)
        d_pairs = d_chunk.merge(d_chunk, on="PRIMARYID")
        d_pairs = d_pairs[d_pairs["DRUGNAME_x"] < d_pairs["DRUGNAME_y"]]
        
        # Merge with reactions and demographics
        merged = d_pairs.merge(r_chunk, on="PRIMARYID").merge(demo_chunk, on="PRIMARYID")
        
        # Rename columns to match expected output
        df = merged.rename(columns={
            "DRUGNAME_x": "drug_a",
            "DRUGNAME_y": "drug_b",
            "PT": "event",
            "AGE": "age",
            "SEX": "gender"
        })
        
        df = df[["drug_a", "drug_b", "event", "age", "gender"]]
        total_rows += len(df)
        
        # Append to CSV
        df.to_csv(out_csv, mode='a', header=False, index=False)
        print(f"  Processed chunk {i//chunk_size + 1}, cumulative rows: {total_rows}")

    print(f"{q} finished. Total rows: {total_rows}")

print("\nAll quarters processed successfully.")