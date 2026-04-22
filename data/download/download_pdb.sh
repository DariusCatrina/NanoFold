#!/bin/bash

# ==========================================
# Configuration Variables
# ==========================================
PEPTIDES_FILE="./data/download/small_peptides.tsv"
# Using File 1 based on your description (Col 1: entryId, Col 2: repId, Col 3: taxId)
CLUSTER_FILE="./data/download/1-AFDBClusters-entryId_repId_taxId.tsv.gz"
OUTPUT_DIR="/work/dgc26/small_peptides_pdb"
PARALLEL_DOWNLOADS=4 # Number of concurrent downloads

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==========================================
# Step 1: Extract Target Representative IDs
# ==========================================
echo "Extracting representative IDs from ${PEPTIDES_FILE}..."
# Assuming repId is in the first column
cut -f1 "$PEPTIDES_FILE" > target_repIds.tmp

# ==========================================
# Step 2: Map repIds to member entryIds
# ==========================================
echo "Scanning the cluster file to find all member entry IDs..."
echo "(This may take a few minutes depending on file size and disk speed)"

# We use awk to do this efficiently in a single pass:
# 1. NR==FNR reads the first file (target_repIds.tmp) and stores IDs in an array.
# 2. It then streams the .gz file and checks if column 2 (repId) is in our array.
# 3. If yes, it prints column 1 (entryId).
zcat "$CLUSTER_FILE" | awk '
    NR==FNR { targets[$1]; next } 
    $2 in targets { print $1 }
' target_repIds.tmp - > target_entryIds.tmp

MEMBER_COUNT=$(wc -l < target_entryIds.tmp)
echo "Found ${MEMBER_COUNT} total cluster members."

# ==========================================
# Step 3: Download PDB structures in parallel
# ==========================================
echo "Starting downloads to ./${OUTPUT_DIR}/ (Using ${PARALLEL_DOWNLOADS} parallel workers)..."

# The xargs command reads the entry IDs and runs wget concurrently
# -n 1 : Pass one entryId per wget command
# -P   : Run N parallel processes
# wget -q -nc : Quiet mode, No-Clobber (skips already downloaded files)
cat target_entryIds.tmp | xargs -n 1 -P "$PARALLEL_DOWNLOADS" -I {} \
    wget -q -nc "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.pdb" -P "$OUTPUT_DIR"

echo "Download complete! Cleaning up temporary files..."
rm target_repIds.tmp target_entryIds.tmp

echo "All done. Your structures are in the '${OUTPUT_DIR}' directory."