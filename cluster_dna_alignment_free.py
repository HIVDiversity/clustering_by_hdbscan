import os
import pathlib
import argparse
import subprocess
from itertools import groupby
import collections
import math
from sklearn import decomposition
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


__author__ = 'Colin Anthony'


def py3_fasta_iter(fasta_name):
    """
    modified from Brent Pedersen: https://www.biostars.org/p/710/#1412
    given a fasta file. yield tuples of header, sequence
    """
    fh = open(str(fasta_name), 'r')
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        # drop the ">"
        header_str = header.__next__()[1:].strip()
        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (header_str, seq)


def fasta_to_dct(file_name):
    """
    :param file_name: The fasta formatted file to read from.
    :return: a dictionary of the contents of the file name given. Dictionary in the format:
    {sequence_id: sequence_string, id_2: sequence_2, etc.}
    """
    dct = collections.defaultdict(str)
    my_gen = py3_fasta_iter(file_name)
    for k, v in my_gen:
        new_key = k.replace(" ", "_")
        if new_key in dct.keys():
            print("Duplicate sequence ids found. Exiting")
            raise KeyError("Duplicate sequence ids found")
        dct[new_key] = v.upper()

    return dct


def fasta_to_dct_rev(file_name):
    """
    :param file_name: The fasta formatted file to read from.
    :return: a dictionary of the contents of the file name given. Dictionary in the format:
    {sequence_id: sequence_string, id_2: sequence_2, etc.}
    """
    dct = collections.defaultdict(list)
    my_gen = py3_fasta_iter(file_name)
    for k, v in my_gen:
        new_key = k.replace(" ", "_")
        if new_key in dct.keys():
            print("Duplicate sequence ids found. Exiting")
            raise KeyError("Duplicate sequence ids found")
        dct[str(v).upper()].append(new_key)

    return dct


def calc_kmers(seq_dict, name_prefix, ksize=6):
    """
    function to generate kmer counts for each sequence in a dict of key=sequence, value=list of seq names
    :param seq_dict: (str) dict of key=sequence, value=list of seq names
    :param k: (int) size of kmer
    :return: (dict) sorted dict of dicts of kmer counts for each sequence
    """
    kmers = collections.defaultdict(lambda: collections.defaultdict(int))
    sortedKmer = collections.defaultdict(dict)
    all_kmers = collections.defaultdict(int)

    for i, (seq_name, seq) in enumerate(seq_dict.items()):
        for i in range(len(seq) - ksize + 1):
            kmer = seq[i:i + ksize]
            kmers[seq_name][kmer] += 1
            all_kmers[kmer] +=1

    for name, kmer_count_dict in kmers.items():
        for master_kmer in all_kmers.keys():
            if master_kmer not in kmer_count_dict.keys():
                kmer_count_dict[master_kmer] = 0

        sortedKmer[name] = sorted(kmer_count_dict.items(), key=lambda x: x[0])

    return sortedKmer


def cluster_hdbscan(pca_array, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(pca_array)

    return clusterer


def collect_seqs_by_cluster(sequence_d, cluster_obj):

    seqs_by_cluster = collections.defaultdict(lambda: collections.defaultdict(str))
    for entry in cluster_obj:
        seq_name = entry[0]
        seq = sequence_d[seq_name]
        cluster = str(entry[1]).zfill(3)
        if cluster == "-01":
            cluster = "out"
        cluster_prob = str(entry[2])
        new_name = f"{seq_name}_{cluster}_{cluster_prob}"
        seqs_by_cluster[cluster][new_name] = seq

    return seqs_by_cluster


def write_clusters_to_fasta(clustered_dict, prefix_name, outpath):
    target_path = pathlib.Path(outpath, f"{prefix_name}_clusters")
    if target_path.exists():
        search = target_path.glob("*.fasta")
        for old_file in search:
            os.unlink(str(old_file))
    target_path.mkdir(parents=True, exist_ok=True)
    for clust, seq_dict in clustered_dict.items():
        outfile = pathlib.Path(target_path, f"{prefix_name}_{clust}_cluster.fasta")
        with open(outfile, 'w') as fh:
            for seq_name, seq in seq_dict.items():
                fh.write(f">{seq_name}\n{seq}\n")

    return pathlib.Path(target_path).glob("*_cluster.fasta")


def rand_jitter(arr):
    stdev = .015*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def colors(num_colrs):
    ret = collections.defaultdict(float)
    if num_colrs <= 20:
        palette = sns.color_palette("tab20", num_colrs)
    else:
        num_samplings = math.ceil(num_colrs/20)
        palette = []
        for item in range(num_samplings):
            palette.append(sns.color_palette("tab20", 20))

    for i in range(num_colrs):
        ret[str(i)] = palette[i]

    return ret


def compute_pca(array):
    # get max number of PCA components
    max_pca_components = min(len(array), len(array[0]))

    # do PCA on array of kmers for plotting
    pca = decomposition.PCA(n_components=max_pca_components)
    pca.fit(array)

    pca_array = pca.transform(array)

    return pca_array


def plot_clusters(clusterer, array, num_clusts, outfile):
    """

    :param clusterer: (hdbscan object) cluster object from hdbscan
    :param array: (array) array of sorted kmer counts for all sequences
    :param num_clusts: (int) the number of clusters identified
    :param outfile: (str) the name and path of the outfile
    :return:
    """
    # get max number of PCA components
    max_pca_components = min(len(array), len(array[0]))

    # do PCA on array of kmers for plotting
    pca = decomposition.PCA(n_components=max_pca_components)
    pca = pca.fit(array)
    pca_array = pca.transform(array)

    palette = colors(num_clusts)
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    plot_kwds = {'alpha': 0.8, 's': 80, 'linewidths': 0.1}
    clusters = list(clusterer.labels_)

    cluster_colors = [palette[str(col)] if col != -1 else (0, 0, 0) for idx, col in enumerate(clusters)]

    x = rand_jitter(pca_array.T[0])
    y = rand_jitter(pca_array.T[1])
    plt.scatter(x, y, c=cluster_colors, **plot_kwds)
    w = 6.875
    h = 4
    f = plt.gcf()
    f.set_size_inches(w, h)

    plt.savefig(outfile, ext='png', dpi=600, format='png', facecolor='white', bbox_inches='tight')


def d_freq_lists(dna_list):
    """
    function to generate a dict of base frequencies from a list of aligned DNA sequences
    :param dna_list: (list) a list of DNA sequences
    :return: (dict) a dictionary of the frequency for each base, for each site in the alignment
    """

    n = len(dna_list[0])
    dist_dict = {'A': [0]*n, 'C': [0]*n, 'G': [0]*n, 'T': [0]*n, '-': [0]*n}

    total = len(dna_list)
    for seq in dna_list:
        for index, dna in enumerate(seq):
            dist_dict[dna][index] += 1

    for base, freqlist in dist_dict.items():
        for i, cnt in enumerate(freqlist):
            frq = round((cnt/total*100), 4)
            freqlist[i] = frq
        dist_dict[base] = freqlist

    return dist_dict


def consensus_maker(d):
    """
    Create a consensus sequence from an alignment
    :param d: (dict) dictionary of an alignment (key = seq name (str): value = aligned sequence (str))
    :return: (str) the consensus sequence
    """
    seq_list = []
    for names, seq in d.items():
        seq_list.append(seq)

    master_profile = d_freq_lists(seq_list)
    n = len(seq_list[0])
    consensus = ""
    degen = {('A', 'G'): 'R', ('C', 'T'): 'Y', ('A', 'C'): 'M', ('G', 'T'): 'K', ('C', 'G'): 'S', ('A', 'T'): 'W',
             ('A', 'C', 'T'): 'H', ('C', 'G', 'T'): 'B', ('A', 'C', 'G'): 'V', ('A', 'G', 'T'): 'D',
             ('A', 'C', 'G', 'T'): 'N'}

    for i in range(n):
        dct = {N: master_profile[N][i] for N in ['T', 'G', 'C', 'A', '-']}
        m = max(dct.values())
        b = max(dct, key=dct.get)
        l = list(sorted(N for N in ['T', 'G', 'C', 'A', '-'] if dct[N] == m))
        if len(l) == 1:
            consensus += str(b)
        elif '-' in l and len(l) == 2:
            l.remove('-')
            l = tuple(l)
            consensus += str(l)
        elif '-' in l and len(l) > 2:
            l.remove('-')
            l = tuple(l)
            consensus += str(degen[l])
        else:
            l = tuple(l)
            consensus += str(degen[l])

    return consensus


def main(infile, outpath, name, min_cluster_size, kmer_size):
    """
    script to haplotype DNA sequences based on kmer counts and hierarchical clustering
    :param infile: (str) path and name of the input fasta file
    :param outpath: (str) the path to where the outfiles will be written
    :param name: (str) prefix for the outfiles
    :param min_cluster_size: (int) minimum number of sequences needed to make a cluster
    :return: None
    """

    # get absolute paths
    infile = pathlib.Path(infile).absolute()
    outpath = pathlib.Path(outpath).absolute()
    out_name = name + "_clustered_haplotypes.fasta"
    outfile = pathlib.Path(outpath, out_name)
    print("\nOutfile will be: ", outfile)

    in_seqs_d = fasta_to_dct(infile)
    for seq_name, seq in in_seqs_d.items():
        in_seqs_d[seq_name] = seq.replace("-", "")

    total_number_seqs = len(in_seqs_d)

    print("\ncounting kmers for each sequence\n")
    kmer_dict = calc_kmers(in_seqs_d, name, kmer_size)
    sequence_names = list(kmer_dict.keys())
    master_kmer_cnt_array = []
    for seq_name, kmer_counts_tups in kmer_dict.items():
        seq_kmer_array = []
        for (kmer, cnt) in kmer_counts_tups:
            seq_kmer_array.append(cnt)
        master_kmer_cnt_array.append(seq_kmer_array)

    kmer_array = master_kmer_cnt_array

    cluster_array = compute_pca(kmer_array)

    # cluster (HDBSCAN)
    print("\nClustering sequences\n")
    clusterer = cluster_hdbscan(cluster_array, min_cluster_size)

    # get seq names, clusters and cluster probability values
    names_clusts_clustprob = list(zip(sequence_names, clusterer.labels_, clusterer.probabilities_))

    # number of clusters
    num_clusts = sorted(list(set(clusterer.labels_)))[-1] + 1
    print(f"\nThere were {num_clusts} clusters found\n")
    plot_outfile = pathlib.Path(outpath, name + "_clusters_by_2_PCA_components.png")
    plot_clusters(clusterer, cluster_array, num_clusts, plot_outfile)

    # collect sequences in to their clusters
    clustered_seqs_d = collect_seqs_by_cluster(in_seqs_d, names_clusts_clustprob)

    # write each cluster to a file
    clustered_outfiles = write_clusters_to_fasta(clustered_seqs_d, name, outpath)

    # align each cluster's fasta file
    # Todo: only asign if the prob of being in the cluster is > some value (how to adjust feq values?)
    aligned_cluster_files = []
    for file in clustered_outfiles:
        path = file.parent
        new_file = str(file).replace('.fasta', '_aligned.fasta')
        cluster_outfile = pathlib.Path(path, new_file)
        aligned_cluster_files.append(cluster_outfile)
        cmd = f"mafft {file} > {str(cluster_outfile)} 2>/dev/null"
        subprocess.call(cmd, shell=True)
        os.unlink(str(file))

    print("\nGetting consensus/centroid for each cluster\n")
    master_clustered_cons_d = collections.defaultdict(str)

    # read in fasta files
    for file in aligned_cluster_files:
        cluster_name = file.stem
        aligned_cluster_seq_d = fasta_to_dct(file)

        # if cluster file is the outliers, get num of outlies and subtract from total
        # Todo: try to find closest real cluster consensus and assign seq to that cluster for freq adjustment?
        if "out" in cluster_name.split("_"):
            total_number_seqs += len(aligned_cluster_seq_d.keys())
            continue

        consensus = consensus_maker(aligned_cluster_seq_d)
        cons_name = f"{cluster_name}"
        cluster_size = len(aligned_cluster_seq_d.keys())
        seq_to_use = consensus
        if consensus in fasta_to_dct_rev(infile).keys():
            # match_names_list = fasta_to_dct_rev(infile)[consensus]
            cons_name = cons_name + "_orig"
        else:
            print("No input seq identical to consensus, using consensus")
            cons_name = cons_name + "_cons"

        if seq_to_use not in master_clustered_cons_d.keys():
            master_clustered_cons_d[seq_to_use] = f"{cons_name}_{str(cluster_size).zfill(4)}"
        else:
            old_name = master_clustered_cons_d[seq_to_use]
            old_name_parts = old_name.split("_")
            new_count = int(old_name_parts[-1]) + cluster_size
            new_name = f"{'_'.join(old_name_parts[:-1])}_{str(new_count).zfill(4)}"
            master_clustered_cons_d[seq_to_use] = new_name

    # add frequencies and write to file
    with open(outfile, 'w') as fh:
        for cluster_seq, cluster_name in master_clustered_cons_d.items():
            name_parts = cluster_name.split("_")
            cluster_size = int(name_parts[-1])
            cluster_freq = round(cluster_size/total_number_seqs * 100, 4)
            new_clust_name = f"{'_'.join(name_parts)}_{cluster_freq}"
            fh.write(f">{new_clust_name}\n{cluster_seq}\n")

    print(f"\n{num_clusts} clusters were identified\n")

    print("Clustering complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusters unaligned DNA sequences into haplotypes, '
                                                 'using principle component analysis '
                                                 'and density based clustering with HDBSCAN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in', '--infile', default=argparse.SUPPRESS, type=str,
                        help='The input fasta file', required=True)
    parser.add_argument('-o', '--outpath', default=argparse.SUPPRESS, type=str,
                        help='The path to where the output file will be copied', required=True)
    parser.add_argument('-n', '--name', default=argparse.SUPPRESS, type=str,
                        help='The prefix for the outfile', required=True)
    parser.add_argument('-s', '--min_cluster_size', default=2, type=int,
                        help='The minimum number of sequences needed to form a cluster', required=False)
    parser.add_argument('-k', '--kmer_size', default=6, type=int,
                        help='The size of the kmer', required=False)
    args = parser.parse_args()
    infile = args.infile
    outpath = args.outpath
    name = args.name
    min_cluster_size = args.min_cluster_size
    kmer_size = args.kmer_size

    main(infile, outpath, name, min_cluster_size, kmer_size)
