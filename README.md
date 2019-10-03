# clustering_by_hdbscan

method to cluster DNA sequences to identify distinct haplotypes in a background of sequencing error

## installing the environment

### Download miniconda

### install the conda environment
`conda env create -f environment.yaml`

### activate the env before running the script
`conda activate clustering`

## to run the script:
### alignment free version:
`python /path_to_repo/clustering_by_hdbscan/cluster_dna_alignment_free.py -in input_file.fasta -o ./ -n run_name -s 2 -k 6`
### alignment dependent version:
`python /path_to_repo/clustering_by_hdbscan/cluster_dna_alignment_free.py -in input_file.fasta -o ./ -n run_name -s 2 -f 10 -c 20`


## Method outline:

Step 1: 

DNA sequences are encoded using the 1Hot method (requires aligned sequences) or by kmer counts (alignment free method)

Step 2:

dimensionality reduced using PCA (alignment dependent method only)

Step 3:

Sequences are clustered using hdbscan.


