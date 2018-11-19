#!/usr/bin/python3
from __future__ import print_function
from __future__ import division
import os
import argparse
from itertools import groupby
import collections
import pandas as pd
from sklearn import decomposition
from sklearn import manifold
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
from Bio.Align.AlignInfo import SummaryInfo
from Bio.Alphabet.IUPAC import ambiguous_dna
from Bio.Align import MultipleSeqAlignment


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


def recode_dna_ordinal(dna_dict):
    coded_dna_d = collections.defaultdict(list)
    nucl_coding_assignment = {"A": 1,
                              "C": 2,
                              "G": 3,
                              "T": 4,
                              "-": 5,
                              }

    for seq_name, seq in dna_dict.items():
        new_seq = []
        seq = seq.upper()
        for base in seq:
            if base not in list(nucl_coding_assignment.keys()):
                new_base = 6
            else:
                new_base = nucl_coding_assignment[base]

            new_seq.append(new_base)

        coded_dna_d[seq_name] = new_seq

    return coded_dna_d


def recode_dna_onehot(dna_dict):
    """

    :param dna_dict:
    :return:
    """
    coded_dna_d = collections.defaultdict(list)
    bases = ["A", "C", "G", "T", "-", "N"]
    for seq_name, seq in dna_dict.items():
        new_seq = []
        for base in seq:
            new_base_array = [0, 0, 0, 0, 0, 0]
            if base in bases:
                base_index = bases.index(base)
                new_base_array[base_index] = 1
            else:
                base_index = -1
                new_base_array = [0, 0, 0, 0, 0, 0]
                new_base_array[base_index] = 1
            new_seq.extend(new_base_array)

        coded_dna_d[seq_name] = new_seq

    return coded_dna_d


def compute_pca(df_array, pca_components):
    pca = decomposition.PCA(n_components=pca_components)
    pca.fit(df_array)

    pca_array = pca.transform(df_array)

    return pca_array


def compute_tsne(df_array, pca_components):
    """

    :param df_array:
    :param pca_components:
    :return:
    """
    tsne = manifold.TSNE(n_components=pca_components)
    tsne.fit(df_array)
    tsne_array = tsne.fit_transform(df_array.values)

    return tsne_array


def cluster_hdbscan(pca_array, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(pca_array)

    return clusterer


def get_centroids(clustered_seqs_d, total_number_seqs, in_seqs_d_reversed, in_seqs_d, outfile):
    print("getting consensus of cluster")
    collected_cluster_centroids = collections.defaultdict(list)
    outliers = collections.defaultdict(list)

    for cluster, cluster_dict in clustered_seqs_d.items():
        cluster_size = len(cluster_dict)
        freq = round((cluster_size / total_number_seqs) * 100, 3)
        if cluster != "-01":
            name_stem_assigned = False
            name_stem = ''
            align_obj = MultipleSeqAlignment([], ambiguous_dna)
            for seq_name, seq in cluster_dict.items():
                if not name_stem_assigned:
                    name_stem = "_".join(seq_name.split("_")[:6])
                    name_stem_assigned = True
                    align_obj.add_sequence(seq_name, str(seq))

            summary_align = SummaryInfo(align_obj)
            consensus = SummaryInfo.gap_consensus(summary_align, threshold=0.51, ambiguous='N', consensus_alpha=ambiguous_dna)

            try:
                names_list = in_seqs_d_reversed[consensus]
                centroid_seq = in_seqs_d[names_list[0]]
                clust_seq_name = name_stem + "_orig_" + str(cluster).zfill(3) + "_" + str(freq).zfill(3)
                collected_cluster_centroids[centroid_seq].append(clust_seq_name)
            except:
                clust_seq_name = name_stem + "_cons_" + str(cluster).zfill(3) + "_" + str(freq).zfill(3)
                collected_cluster_centroids[consensus].append(clust_seq_name)
            #     # Todo try and get sequence with highest cluster prob, choose most abundant if mult hits,
            #     # todo then choose random if still multiple hits

        else:
            for seq_name, seq in cluster_dict.items():
                outliers[seq].append(seq_name)


    final_centroids = collections.defaultdict()
    for seq, name in collected_cluster_centroids.items():
        if len(name) > 1:
            freq = 0
            first = True
            stem = ''
            if "orig" in name:
                idx = name.index("orig")
                stem = "_".join(name[idx].split("_")[:-1])
                first = False
            for i in name:
                freq += float(i.split("_")[-1])
                if first:
                    stem = "_".join(i.split("_")[:-1])
                    first = False
            new_name = stem + "_" + str(round(freq, 3)).zfill(3)
            final_centroids[new_name] = seq
        else:
            final_centroids[name[0]] = seq

    with open(outfile, 'w') as handle:
        for name, seq in final_centroids.items():
            handle.write(">{}\n{}\n".format(name, seq))

    outliers_ourtile = outfile.replace(".fasta", "_outliers.fasta")
    with open(outliers_ourtile, 'w') as handle:
        for seq, name_list in outliers.items():
            for name in name_list:
                handle.write(">{}\n{}\n".format(name, seq))


def rand_jitter(arr):
    stdev = .015*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_clusters(clusterer, pca_array, num_clusts, outfile):
    outfile = outfile.replace(".fasta", ".png")
    palette = sns.color_palette("tab20", num_clusts)
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    plot_kwds = {'alpha': 0.8, 's': 80, 'linewidths': 0.1}
    zippy = list((zip(clusterer.labels_, clusterer.probabilities_)))
    cluster_colors = [sns.desaturate(palette[col], sat) if col != -1 else (0, 0, 0) for col, sat in zippy]
    x = rand_jitter(pca_array.T[0])
    y = rand_jitter(pca_array.T[1])
    plt.scatter(x, y, c=cluster_colors, **plot_kwds)
    w = 6.875
    h = 4
    f = plt.gcf()
    f.set_size_inches(w, h)
    # plt.show()
    plt.savefig(outfile, ext='png', dpi=600, format='png', facecolor='white', bbox_inches='tight')


def main(infile, outpath, name, min_cluster_size, pca_components):
    total_start = time.time()
    # get absolute paths
    infile = os.path.abspath(infile)
    outpath = os.path.abspath(outpath)
    out_name = name + "_clustered_haplotypes.fasta"
    outfile = os.path.join(outpath, out_name)
    print("Ourfile will be: ", outfile)

    in_seqs_d = fasta_to_dct(infile)
    total_number_seqs = len(in_seqs_d)

    in_seqs_d_reversed = fasta_to_dct_rev(infile)

    ordinal = False
    # True = use ordinal values for DNA encoding.
    # Changing this will decrease array width,
    # but will reduce accuracy of cluster identification.
    # only do this if array width limitations exist

    print("Coding DNA sequences to numerical values\nfor dimensionality collapse and clustering")

    if ordinal:
        re_coded_dna_d = recode_dna_ordinal(in_seqs_d)
        df = pd.DataFrame.from_dict(re_coded_dna_d, orient='index')
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'seq_name'}, inplace=True)
        df.fillna(0, inplace=True)
        headers = list(df)
        sequence_names = df[headers[0]]
        df.drop(headers[0], axis=1, inplace=True)
        df_array = df.values

    # recode strings into oneHot coding of list of arrays, for each sequence. Better resolution than ordinal coding.
    else:
        re_coded_dna_d = recode_dna_onehot(in_seqs_d)
        df_array = [x for k, x in re_coded_dna_d.items()]
        sequence_names = [k for k, x in re_coded_dna_d.items()]

    # Reduce dimensionality
    linear_dimension_collapse = True

    # Reduce dimensionality with PCA then with TSNE

    if linear_dimension_collapse:
        # todo get optimal number of components to cluster on
        print("computing PCA for alignment")
        reduced_dimension_array = compute_pca(df_array, pca_components)
    else:
        print("computing T-SNE for alignment")
        print("T-SNE can only take 2 components\nsetting components to 2")
        pca_components = 2
        reduced_dimension_array = compute_tsne(df_array, pca_components)

    # cluster (HDBSCAN) on first two principal components
    print("Clustering sequences by first {} components".format(pca_components))
    clusterer = cluster_hdbscan(reduced_dimension_array, min_cluster_size)

    # get seq names, clusters and cluster probability values
    names_clusts_clustprob = list(zip(sequence_names, clusterer.labels_, clusterer.probabilities_))

    # number of clusters
    num_clusts = sorted(list(set(clusterer.labels_)))[-1]

    # collect sequences in to their clusters
    clustered_seqs_d = collections.defaultdict(dict)
    for i in names_clusts_clustprob:
        seq_name = i[0]
        cluster = str(i[1]).zfill(3)
        cluster_prob = str(i[2])
        seq = in_seqs_d[seq_name]

        new_name = "{}_{}_{}".format(seq_name, cluster, cluster_prob)
        clustered_seqs_d[cluster][new_name] = seq

    get_centroids(clustered_seqs_d, total_number_seqs, in_seqs_d_reversed, in_seqs_d, outfile)

    print("{} clusters were identified".format(str((num_clusts - 1))))

    print("Clustering complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusters a fasta file into haplotypes, using principle component '
                                                 'analysis and density based clustering with HDBSCAN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--infile', default=argparse.SUPPRESS, type=str,
                        help='The input fasta file', required=True)
    parser.add_argument('-o', '--outpath', default=argparse.SUPPRESS, type=str,
                        help='The path to where the output file will be copied', required=True)
    parser.add_argument('-n', '--name', default=argparse.SUPPRESS, type=str,
                        help='The prefix for the outfile', required=True)
    parser.add_argument('-s', '--min_cluster_size', default=2, type=int,
                        help='The minimum number of sequences needed to form a cluster', required=False)
    parser.add_argument('-c', '--pca_components', default=2, type=int,
                        help='The number of PCA components to pass to the clustering algorithm', required=False)

    args = parser.parse_args()
    infile = args.infile
    outpath = args.outpath
    name = args.name
    min_cluster_size = args.min_cluster_size
    pca_components = args.pca_components

    main(infile, outpath, name, min_cluster_size, pca_components)
