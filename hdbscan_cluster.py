import os
import sys
import argparse
from itertools import groupby
import collections
from sklearn import decomposition
from sklearn import manifold
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import pathlib


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
        seq = "".join(s.strip() for s in faiter.__next__()).replace("*", "X")
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


def recode_dna_onehot(dna_dict):
    """
    Encode a DNA sequence into one hot format
    :param dna_dict: (dict) k = seq ID, v = DNA sequence
    :return: (dict) k = seq ID, v = list, oneHot encoded sequence
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


def recode_prot_onehot(prot_dict):
    """
    Encode aa seq into oneHot encoding
    :param prot_dict: (dict) key = seq name, value = amino acid sequence
    :return: (dict) key = name, value = one hot encoded AA seq
    """
    coded_prot_d = collections.defaultdict(list)
    resis = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
             "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "-", "X"]

    for seq_name, seq in prot_dict.items():
        new_seq = []
        for resi in seq:
            new_resi_array = [0] * len(resis)
            # resi is recognised AA, assign
            if resi in resis:
                resi_index = resis.index(resi)
                new_resi_array[resi_index] = 1
            # if resi is not recognised AA, assign to X char
            else:
                resi_index = -1
                new_resi_array = [0] * len(resis)
                new_resi_array[resi_index] = 1
            new_seq.extend(new_resi_array)

        coded_prot_d[seq_name] = new_seq

    return coded_prot_d


def compute_pca(df_array):
    max_pca_components = min(len(df_array), len(df_array[0]))
    pca = decomposition.PCA(n_components=max_pca_components)
    pca.fit(df_array)

    pca_array = pca.transform(df_array)

    return pca_array


def cluster_hdbscan(pca_array, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(pca_array)

    return clusterer


def d_freq_lists(dna_list):
    """
    calculate DNA base frequency per site for cluster
    :param dna_list: (list) a list of DNA sequences
    :return: (dict) a dictionary of the frequency for each base, for each site in the alignment
    """
    seq_len = len(dna_list[0])
    # dist_dict = {'A': [0]*n, 'C': [0]*n, 'G': [0]*n, 'T': [0]*n, '-': [0]*n}
    bases = ["A", "C", "G", "T", "-"]
    nucl_array = np.zeros((len(bases), seq_len))
    total_seqs = len(dna_list)
    for seq in dna_list:
        for col_num, base in enumerate(seq):
            row_num = bases.index(base)
            nucl_array[row_num][col_num] += 1

    nucl_freq_array = nucl_array / total_seqs *100

    return nucl_freq_array, bases, seq_len


def p_freq_lists(prot_list):
    """
    calculate the amino acid frequencies per site
    :param prot_list: (list) a list of AA sequences
    :return: (arr) a numpy array = column are alignment positions, rows are amino acids in alphabetical order,
    list of aa resis relating to the rows in the array, int of length of alignment/array
    """
    seq_len = len(prot_list[0])
    resis = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
             "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "-", "X"]

    aa_array = np.zeros((len(resis), seq_len))
    total_seqs = len(prot_list)

    for seq in prot_list:
        for col_num, resi in enumerate(seq):
            row_num = resis.index(resi)
            aa_array[row_num][col_num] += 1

    aa_freq_array = aa_array / total_seqs *100

    return aa_freq_array, resis, seq_len


def dna_consensus_maker(d):
    """
    Create a consensus sequence from an alignment
    :param d: (dict) dictionary of an alignment (key = seq name (str): value = aligned sequence (str))
    :return: (str) the consensus sequence
    """
    seq_list = []
    for names, seq in d.items():
        seq_list.append(seq)

    nucl_freq_array, bases, seq_len = d_freq_lists(seq_list)
    consensus = ""
    degen = {('A', 'G'): 'R', ('C', 'T'): 'Y', ('A', 'C'): 'M', ('G', 'T'): 'K', ('C', 'G'): 'S', ('A', 'T'): 'W',
             ('A', 'C', 'T'): 'H', ('C', 'G', 'T'): 'B', ('A', 'C', 'G'): 'V', ('A', 'G', 'T'): 'D',
             ('A', 'C', 'G', 'T'): 'N'}

    for col in nucl_freq_array.T:
        max_indices = np.where(col == col.max())
        if len(max_indices) > 1:
            mult_bases = []
            for idx in max_indices:
                mult_bases.append(idx[0])
            mult_bases = tuple(mult_bases)
            cons_base = degen[mult_bases]
        else:
            max_index = col.argmax()
            cons_base = bases[max_index]

        consensus += cons_base

    return consensus


def prot_consensus_maker(d):
    """
    Create a consensus sequence from an alignment
    :param d: (dict) dictionary of an alignment (key = seq name (str): value = aligned sequence (str))
    :return: (str) the consensus sequence
    """
    consensus = ""
    seq_list = [aa_seq for aa_seq in d.values()]
    # for names, seq in d.items():
    #     seq_list.append(seq)

    aa_freq_array, resis, align_len = p_freq_lists(seq_list)

    for col in aa_freq_array.T:
        max_indices = np.where(col == col.max())
        if len(max_indices) > 1:
            cons_resi = "X"
        else:
            max_index = col.argmax()
            cons_resi = resis[max_index]
        consensus += cons_resi

    return consensus


def get_centroids(clustered_seqs_d, outpath, name, total_number_seqs, in_seqs_d_reversed, in_seqs_d, fields, is_prot,
                  num_clusts):
    """
    iterate through all clustered sequences, producing a consensus for each cluster
    :param clustered_seqs_d: (dict) k = cluster number, v = (dict k = seq name, v = sequence)
    :param outpath: path to where output will be written
    :param name: the string for the outfile prefix
    :param total_number_seqs: (int) total number of sequence in the file
    :param in_seqs_d_reversed: (dict) reverse dict k = sequence, v = list of seq ID's
    :param in_seqs_d: (dict) k = seq ID, v = sequence
    :param fields: (int) the number of _ separated fields in the sequence ID to keep in the cluster consensus name
    :param is_prot: (Bool) False = DNA sequence, True = Amino Acid sequence
    :param num_clusts (int) the number of clusters identified
    :return: final_centroids (dict) k = centroid name, v = centroid sequence;
    outliers (dict) k = outlier seq, v = list of seq ID's; consensus_pids_list (dict) k = cons seq, v = list of seq IDs
    """
    print("getting consensus of cluster")
    collected_cluster_centroids = collections.defaultdict(list)
    outliers = collections.defaultdict(list)
    consensus_pids_list = collections.defaultdict(list)

    unique_cons = collections.defaultdict()
    cluster_outfiles = []
    count_outliers = 0
    count_not_outliers = 0
    # total_non_outliers = sum([len(cluster_dict) for cluster, cluster_dict in clustered_seqs_d.items() if cluster != "-01"])

    for ind, (cluster, cluster_dict) in enumerate(clustered_seqs_d.items()):
        cluster_outfile = pathlib.Path(outpath, f"{name}_{str(ind).zfill(3)}_cluster_{cluster}.fasta")

        cluster_size = len(cluster_dict)
        freq = round((cluster_size / total_number_seqs) * 100, 3)
        # if num clusters is 0 = sample is homogenous = all 1 cluster, no outliers
        if cluster != "-01" or num_clusts == 0:
            name_stem_assigned = False
            name_stem = ''
            for seq_name, seq in cluster_dict.items():
                count_not_outliers += 1
                if not name_stem_assigned:
                    name_stem = "_".join(seq_name.split("_")[:fields])
                    name_stem_assigned = True

            if is_prot:
                consensus = prot_consensus_maker(cluster_dict)
            else:
                consensus = dna_consensus_maker(cluster_dict)

            if consensus not in unique_cons.keys():
                cluster_outfiles.append(cluster_outfile)
                with open(cluster_outfile, "w") as fh:
                    for seq_name, seq in cluster_dict.items():
                        consensus_pids_list[consensus].append(seq_name)
                        fh.write(f">{seq_name}_clust_{str(cluster).zfill(3)}\n{seq}\n")
            else:
                # cluster_outfile = unique_cons[consensus]
                # cluster = str(cluster_outfile).replace(".fasta", "").split("_")[-1]
                with open(cluster_outfile, "w") as fh:
                    for seq_name, seq in cluster_dict.items():
                        consensus_pids_list[consensus].append(seq_name)
                        fh.write(f">{seq_name}_clust_{str(cluster).zfill(3)}\n{seq}\n")
            unique_cons[consensus] = cluster_outfile

            try:
                names_list = in_seqs_d_reversed[consensus]
                centroid_seq = in_seqs_d[names_list[0]]
                name_stem = "_".join(names_list[0].split("_")[:fields])
                clust_seq_name = f"{name_stem}_orig_{str(cluster).zfill(3)}_{str(round(cluster_size)).zfill(4)}_" \
                                 f"{str(round(freq, 3)).zfill(3)}_cluster"
                collected_cluster_centroids[centroid_seq].append(clust_seq_name)
            except:
                clust_seq_name = f"{name_stem}_cons_{str(cluster).zfill(3)}_{str(round(cluster_size)).zfill(4)}_" \
                                 f"{str(round(freq, 3)).zfill(3)}_cluster"
                collected_cluster_centroids[consensus].append(clust_seq_name)

        else:
            for seq_name, seq in cluster_dict.items():
                count_outliers += 1
                outliers[seq].append(seq_name)

    # combine any clusters that have the same consensus sequence and update freq
    final_centroids = collections.defaultdict()
    for seq, names in collected_cluster_centroids.items():
        stem = "_".join(names[0].split("_")[:-4])
        tag = names[0].split("_")[-1]
        clust_num = str(min([int(x.split("_")[-4]) for x in names])).zfill(3)
        cumulative_count = sum([int(x.split("_")[-3]) for x in names])
        adjust_freq = round((cumulative_count / count_not_outliers) * 100, 3)
        new_name = f"{stem}_{clust_num}_{str(cumulative_count).zfill(4)}_{str(adjust_freq).zfill(4)}_{tag}"
        cons_type = [x.split("_")[-5] for x in names]
        if "orig" in cons_type:
            new_name = new_name.replace("cons", "orig")

        final_centroids[new_name] = seq

    return final_centroids, outliers, consensus_pids_list


def customdist(s1, s2):

    if len(s1) != len(s2):
        print("sequences must be the same length")
        sys.exit()

    dist = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            dist += 1
    diff = 0
    for i in range(len(s1)-1):
        if s1[i] != s2[i]:
            if (s1[i] == "-" and s1[i+1] == "-" and s2[i+1] != "-") \
                    or (s2[i] == "-" and s2[i+1] == "-" and s1[i+1] != "-"):
                diff += 1

    return (dist-diff)


def rescue_outliers_for_freq_updating(final_centroids, total_number_seqs, outliers):

    outliers_num = len(outliers.keys())
    print("num outliers", outliers_num)

    update_freq_d = collections.defaultdict(int)
    for out_s, out_n in outliers.items():
        max_dist = len(out_s.replace("_", ""))
        max_dist_cent = None
        for cent_n, cent_s in final_centroids.items():
            dist = customdist(out_s, cent_s)
            if dist < max_dist:
                max_dist = dist
                max_dist_cent = cent_n
        if max_dist_cent is not None:
            update_freq_d[max_dist_cent] += 1

    updated_centroids = collections.defaultdict()

    for cent_n, cent_s in final_centroids.items():
        parts = cent_n.split("_")
        count = int(parts[-2])
        old_freq = float(parts[-1])
        new_count = count + update_freq_d[cent_n]
        # new_freq = round(new_count / total_number_seqs * 100, 2)
        new_freq = round(count / (total_number_seqs - outliers_num)  * 100, 2)

        parts[-2] = str(new_count)
        parts[-1] = str(new_freq)
        new_name = "_".join(parts)
        updated_centroids[new_name] = cent_s

    return updated_centroids


def write_output(final_centroids, outliers, outfile):

    sorted_final_centroids = collections.OrderedDict(sorted(final_centroids.items(),
                                                            key=lambda kv: float(kv[0].split("_")[-1]), reverse=True))
    with open(outfile, 'w') as handle:
        for name, seq in sorted_final_centroids.items():
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


def main(infile, outpath, name, min_cluster_size, fields, is_prot):

    # get absolute paths
    infile = os.path.abspath(infile)
    outpath = os.path.abspath(outpath)
    out_name = name + "_clustered_haplotypes.fasta"
    outfile = os.path.join(outpath, out_name)
    print("Ourfile will be: ", outfile)

    in_seqs_d = fasta_to_dct(infile)
    total_number_seqs = len(in_seqs_d)

    in_seqs_d_reversed = fasta_to_dct_rev(infile)

    print("Coding DNA sequences to numerical values for dimensionality collapse and clustering")

    if is_prot:
        re_coded_seq_d = recode_prot_onehot(in_seqs_d)
    else:
        re_coded_seq_d = recode_dna_onehot(in_seqs_d)

    df_array = [x for k, x in re_coded_seq_d.items()]
    sequence_names = [k for k, x in re_coded_seq_d.items()]

    # Reduce dimensionality with PCA
    print("computing PCA for alignment")
    reduced_dimension_array = compute_pca(df_array)

    # cluster (HDBSCAN) on first two principal components
    print("Clustering sequences by max components")
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

    print("Getting consensus/centroid for each cluster")
    centroids, outliers, consensus_pids_list = get_centroids(clustered_seqs_d, outpath, name, total_number_seqs,
                                                             in_seqs_d_reversed, in_seqs_d, fields, is_prot, num_clusts)

    print("Adjusting cluster frequencies based on closest outliers")
    final_centroids = rescue_outliers_for_freq_updating(centroids, total_number_seqs, outliers)

    print("Write output to file")
    # write_output(final_centroids, outliers, outfile)
    final_clusters_out = len(centroids.keys())

    write_output(centroids, outliers, outfile)

    print(f"{final_clusters_out} clusters were identified")

    print("Clustering complete")

    # return the file containing the centroids and a dict of consensus seqs : list of seq names that went into the cons
    return outfile, consensus_pids_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusters a fasta file into haplotypes, using principle component '
                                                 'analysis and density based clustering with HDBSCAN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in', '--infile', default=argparse.SUPPRESS, type=str,
                        help='The input fasta file', required=True)
    parser.add_argument('-o', '--outpath', default=argparse.SUPPRESS, type=str,
                        help='The path to where the output file will be copied', required=True)
    parser.add_argument('-n', '--name', default=argparse.SUPPRESS, type=str,
                        help='The prefix for the outfile', required=True)
    parser.add_argument('-s', '--min_cluster_size', default=2, type=int,
                        help='The minimum number of sequences needed to form a cluster', required=False)
    parser.add_argument('-f', '--fields', default=6, type=int,
                        help='The number of base name fields to keep (based on "_" delimiter. eg: -f 2, '
                             'to keep the first two fields)', required=False)
    parser.add_argument('-p', '--is_prot', default=False, action="store_true", required=False,
                        help='Use this flag if you have an amino acid alignment, default is DNA alignment')
    args = parser.parse_args()
    infile = args.infile
    outpath = args.outpath
    name = args.name
    min_cluster_size = args.min_cluster_size
    fields = args.fields
    is_prot = args.is_prot

    main(infile, outpath, name, min_cluster_size, fields, is_prot)
