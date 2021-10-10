import numpy as np
import pandas as pd
from itertools import combinations, product
from os.path import isfile

from snp.data.data_readers import SNPReader, SNPKernelReader, GeneChromoReader, SubjectReader
from snp.tasks.population_binary_classification_kernel import classify, ESTIMATORS, \
    SUBJECT_FILENAME
from snp.tasks.population_binary_classification_mit import GENE_FILENAME, SNP_FILENAME


# dietary thermogenesis
genes = [
    'BMP8A', # 1
    'OMA1', # 1
    'UCP1', # 4
    'ADRB2', # 5
    'ADRB1', # 10
    'APPL2', # 12
    'TRPV4', # 12
    'TRPV1', # 17
    'MC4R', # 18
]

# # cold adaptation
# genes = [
#     'LEPR', # 1
#     'PRDM16', # 1
#     'CREB1', # 2
#     'PPARG', # 3
#     'PRKAR2A', # 3
#     'PPARGC1A', # 4
#     'UCP1', # 4
#     'PPARGC1B', # 5
#     'HOXA1', # 7
#     'LEP', # 7
#     'NRF1', # 7
#     'PRKAR1B', # 7
#     'PRKAR2B', # 7
#     'ADRA1A', # 8
#     'ADRB3', # 8
#     'PLIN2', # 9
#     'UCP2', # 11
#     'UCP3', # 11
#     'HOXC4', # 12
#     'DIO2', # 14
#     'PLIN1', # 15
#     'FTO', # 16
#     'PRKAR1A', # 17
#     'CIDEA', # 18
#     'LIPE', # 19
#     'PLIN3', # 19
#     'PLIN5', # 19
#     'NRIP1', # 21
# ]


def samples_diff():
    with open(SNP_FILENAME, 'r') as mit_file, \
            open(f'../data/gene_data/{genes[0]}.vcf', 'r') as kernel_file:
        for _ in range(9):
            mit_header = mit_file.readline()
        mit_header = mit_file.readline().split('\t')[9:]
        mit_header[-1] = mit_header[-1][:-1]

        kernel_header = kernel_file.readline().split('\t')[9:]
        kernel_header[-1] = kernel_header[-1][:-1]

        mit_set = set(mit_header)
        kernel_set = set(kernel_header)

        # print(f'list(mit_set.intersection(kernel_set)): {sorted(list(mit_set.intersection(kernel_set)))}')

        index = sorted(list(mit_set.intersection(kernel_set)))
        drop_mit = list(mit_set - kernel_set)
        drop_kernel = list(kernel_set - mit_set)

        # print(f'index: {index}')

        return index, drop_mit, drop_kernel


def get_frequency_data(x_mit, x_kernel, n_samples, n_common_features, genes_name):
    # 1) convert x to F numpy
    # 2) convert to bool
    # 3) use vector bitwise operations
    file_name = f'../data/frequency_data/{genes_name}.npy'
    if isfile(file_name):
        print('Reading frequency...')
        with open(file_name, 'rb') as file:
            return np.load(file)
    else:
        print('Binarizing x_mit...')
        x_mit_bin = np.ones(x_mit.shape, dtype=bool, order='F')
        for j, (_, column) in enumerate(x_mit.items()):
            x_mit_bin[:, j] &= (column.values != 0)

        print('Binarizing x_kernel...')
        x_kernel_bin = np.ones(x_kernel.shape, dtype=bool, order='F')
        for j, (_, column) in enumerate(x_kernel.items()):
            x_kernel_bin[:, j] &= (column.values != '0|0')

        print('Frequency computing...')
        frequency_data = np.zeros((3, n_common_features))
        for i in range(x_mit_bin.shape[1]):
            for j in range(x_kernel_bin.shape[1]):
                idx = i * x_kernel_bin.shape[1] + j
                frequency_data[0][idx] = (x_mit_bin[:, i] | x_kernel_bin[:, j]).sum()
                frequency_data[1][idx] = i
                frequency_data[2][idx] = j

        print('Saving frequency...')
        with open(file_name, 'wb') as file:
            np.save(file, frequency_data)

        return frequency_data


def build_hist(genes_name='dietary'):
    from matplotlib import pyplot as plt

    _, drop_mit, drop_kernel = samples_diff()

    print('SNPReader...')
    snp_mit_reader = SNPReader(SNP_FILENAME)
    _, x_mit = snp_mit_reader.split_data(drop=drop_mit)

    print('SNPKernelReader...')
    snp_kernel_reader = SNPKernelReader()
    _, x_kernel = snp_kernel_reader.get_common_table(genes, drop=drop_kernel)

    n_common_features = x_mit.shape[1] * x_kernel.shape[1]
    n_samples = x_mit.shape[0]
    assert x_mit.shape[0] == x_kernel.shape[0], \
        f'shapes[0] are not equal: {x_mit.shape[0]}, {x_kernel.shape[0]}'

    frequency_data = get_frequency_data(x_mit, x_kernel, n_samples,
                                        n_common_features, genes_name)

    print('Drawing histogram...')
    plt.hist(frequency_data[0])
    plt.savefig('./results/plots/hist.png')


def get_common_x(n_features, genes_name):
    index, drop_mit, drop_kernel = samples_diff()

    print('SNPReader...')
    snp_mit_reader = SNPReader(SNP_FILENAME)
    _, x_mit = snp_mit_reader.split_data(drop=drop_mit)

    print('SNPKernelReader...')
    snp_kernel_reader = SNPKernelReader()
    _, x_kernel = snp_kernel_reader.get_common_table(genes, drop=drop_kernel)

    n_common_features = x_mit.shape[1] * x_kernel.shape[1]
    n_samples = x_mit.shape[0]
    assert x_mit.shape[0] == x_kernel.shape[0], \
        f'shapes[0] are not equal: {x_mit.shape[0]}, {x_kernel.shape[0]}'

    frequency_data = get_frequency_data(x_mit, x_kernel, n_samples, n_common_features,
                                        genes_name)

    print('Sorting frequency...')
    columns_names = list(product(x_mit.columns, x_kernel.columns))
    freq_index = ['frequency', 'mit_idx', 'kernel_idx']
    frequency = pd.DataFrame(frequency_data, columns=columns_names, index=freq_index)
    frequency.sort_values('frequency', axis=1, ascending=False, inplace=True)
    print('After sorting')

    def snps_to_number(snp_mit, snp_kernel):
        if snp_kernel == '0|0':
            return 0 if snp_mit == 0 else 1
        elif snp_kernel == '0|1':
            return 2 if snp_mit == 0 else 3
        elif snp_kernel == '1|1':
            return 4 if snp_mit == 0 else 5
        else:
            return 6

    print('Leaving best features...')
    short_column_names = frequency.columns[:n_features]
    frequency = frequency[short_column_names]
    frequency_data = frequency.to_numpy()
    print(f'frequency_data: {frequency_data}')

    print('Raw dataset preparing...')
    x = np.zeros((n_samples, n_features))
    for idx, _ in enumerate(frequency_data):
        i = int(frequency_data[1, idx])
        j = int(frequency_data[2, idx])
        for k in range(n_samples):
            x[k, idx] = snps_to_number(x_mit.iloc[k, i], x_kernel.iloc[k, j])

    return x, short_column_names, index


def compute(super_pops, pops, gene_reader, x, filename):
    print(f'{pops[0]} ({super_pops[0]})\t{pops[1]} ({super_pops[1]})')
    with open(filename, 'a') as file:
        file.write(f'{pops[0]} ({super_pops[0]})\t{pops[1]} ({super_pops[1]})\t')

    subject_reader = SubjectReader(SUBJECT_FILENAME)
    subject_reader.leave_super_pops(super_pops)
    subject_reader.leave_pops(pops)
    subjects = subject_reader.data['id']

    X = np.array(x.loc[subjects, :])
    y = np.array([subject_reader.data.loc[subject_reader.data['id'] == id]['pop'] for id in subjects]).ravel()

    best_features = {}
    for Estimator, _, params in ESTIMATORS:
        classify(Estimator, params, X, y, x.columns, best_features, filename)

    top_number = 10
    top_features = sorted(best_features.items(), key=lambda a: -a[1])[:top_number]
    top_features = [f[0] for f in top_features]
    top_info = []
    for i in range(top_number):
        mit_mapinfo, kernel_mapinfo = top_features[i][1: -1].split(', ')
        mit_info = gene_reader.get_info(int(mit_mapinfo))
        kernel_info = gene_reader.get_info(int(kernel_mapinfo))
        top_info.append(mit_info + kernel_info)

    final_top_info = []
    for feature_info in top_info:
        final_top_info.append([item[0] for item in feature_info])

    with open(filename, 'a') as file:
        for i, feature in enumerate(top_features):
            file.write(f'{feature} ({final_top_info[i]})\t')
        file.write('\n')


def main():
    print('GeneChromoReader...')
    genes_name = 'dietary'
    # genes_name = 'cold'
    gene_reader = GeneChromoReader(GENE_FILENAME)
    gene_reader.leave_chromos(['1', '4', '5', '10', '12', '17', '18'])
    # gene_reader.leave_chromos(['1', '2', '3', '4', '5', '7', '8', '9', '11', '12', '14',
    #                            '15', '16', '17', '18', '19', '21'])

    x, columns_names, index = get_common_x(2000, genes_name)

    print('One hot encoding...')
    suffixes = ['_0_00', '_1_00', '_0_01', '_1_01', '_0_11', '_1_11', '_NA']
    columns = list(str(i) + j for i, j in product(columns_names, suffixes))
    X = np.zeros((x.shape[0], len(suffixes) * x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            idx = int(len(suffixes) * j + x[i, j])
            X[i, idx] = 1
    X = pd.DataFrame(X, index, columns)

    pops_names = [
        'worldwide',
        'EAS',
        'EUR',
        'SAS',
        'AFR',
        'AMR',
    ]

    populations_list = [
        (('EAS', 'JPT'),
         ('EUR', 'TSI'),
         ('AFR', 'LWK'),
         ('SAS', 'PJL'),
         ('AMR', 'PEL'),),
        (('EAS', 'JPT'),
         ('EAS', 'CHS'),
         ('EAS', 'CDX'),
         ('EAS', 'KHV'),
         ('EAS', 'CHB'),),
        (('EUR', 'GBR'),
         ('EUR', 'FIN'),
         ('EUR', 'IBS'),
         ('EUR', 'CEU'),
         ('EUR', 'TSI'),),
        (('SAS', 'PJL'),
         ('SAS', 'BEB'),
         ('SAS', 'STU'),
         ('SAS', 'ITU'),
         ('SAS', 'GIH'),),
        (('AFR', 'ACB'),
         ('AFR', 'GWD'),
         ('AFR', 'ESN'),
         ('AFR', 'MSL'),
         ('AFR', 'YRI'),
         ('AFR', 'LWK'),
         ('AFR', 'ASW'),),
        (('AMR', 'PUR'),
         ('AMR', 'CLM'),
         ('AMR', 'PEL'),
         ('AMR', 'MXL'),),
    ]

    print('Computing...')
    for pops_name, populations in zip(pops_names, populations_list):
        for pop1, pop2 in combinations(populations, 2):
            compute((pop1[0], pop2[0]), (pop1[1], pop2[1]), gene_reader, X,
                    f'./results/mit_and_kernel_simple/{genes_name}_{pops_name}.txt')

if __name__ == '__main__':
    # build_hist()
    main()
