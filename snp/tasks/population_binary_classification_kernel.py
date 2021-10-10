import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from snp.data.data_readers import SNPKernelReader, GeneChromoReader, SubjectReader
from snp.info.chr_by_genes import chr_by_genes


from snp.tasks.population_binary_classification_mit import SUBJECT_FILENAME, \
    GENE_FILENAME, ESTIMATORS


# dietary thermogenesis
# genes = [
#     'BMP8A', # 1
#     'OMA1', # 1
#     'UCP1', # 4
#     'ADRB2', # 5
#     'ADRB1', # 10
#     'APPL2', # 12
#     'TRPV4', # 12
#     'TRPV1', # 17
#     'MC4R', # 18
# ]

# cold adaptation
genes = [
    'LEPR', # 1
    'PRDM16', # 1
    'CREB1', # 2
    'PPARG', # 3
    'PRKAR2A', # 3
    'PPARGC1A', # 4
    'UCP1', # 4
    'PPARGC1B', # 5
    'HOXA1', # 7
    'LEP', # 7
    'NRF1', # 7
    'PRKAR1B', # 7
    'PRKAR2B', # 7
    'ADRA1A', # 8
    'ADRB3', # 8
    'PLIN2', # 9
    'UCP2', # 11
    'UCP3', # 11
    'HOXC4', # 12
    'DIO2', # 14
    'PLIN1', # 15
    'FTO', # 16
    'PRKAR1A', # 17
    'CIDEA', # 18
    'LIPE', # 19
    'PLIN3', # 19
    'PLIN5', # 19
    'NRIP1', # 21
]


def make_genes_files():
    # reader = SNPKernelReader('../data/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr2.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr3.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr4.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr5.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr7.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr8.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr9.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr11.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr12.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr14.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr15.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr16.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr17.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr18.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    # reader = SNPKernelReader('../data/ALL.chr19.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')
    reader = SNPKernelReader('../data/ALL.chr21.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf')

    genes_data = chr_by_genes(genes)

    for gene, gene_data in genes_data.items():
        start, end, _ = gene_data
        reader.get_gene_data(start, end, f'../data/gene_data/{gene}.vcf')


def classify(Estimator, params, X, y, columns, features, filename):
    n = int(.1*len(columns))
    accuracy = []
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(X):
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        clf = Estimator(**params).fit(x_train, y_train)
        prediction = clf.predict(x_test)
        accuracy.append(accuracy_score(y_test, prediction))

        if hasattr(clf, 'feature_importances_'):
            best_idxs = np.argsort(clf.feature_importances_.ravel())
            best_features = columns[best_idxs][-n:]
            for feature in best_features:
                feature_name = feature.split('_')[0]
                try:
                    features[feature_name] += 1
                except KeyError:
                    features.update({feature_name: 1})

    with open(filename, 'a') as file:
        file.write(f'{np.mean(accuracy)}\t')


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
    top_info = [gene_reader.get_info(int(top_features[i])) for i in range(top_number)]

    final_top_info = []
    for feature_info in top_info:
        final_top_info.append([item[0] for item in feature_info])

    with open(filename, 'a') as file:
        for i, feature in enumerate(top_features):
            file.write(f'{feature} ({final_top_info[i]})\t')
        file.write('\n')


def main():
    gene_reader = GeneChromoReader(GENE_FILENAME)
    # gene_reader.leave_chromos(['1', '4', '5', '10', '12', '17', '18'])
    gene_reader.leave_chromos(['1', '2', '3', '4', '5', '7', '8', '9', '11', '12', '14',
                               '15', '16', '17', '18', '19', '21'])

    snp_reader = SNPKernelReader()
    _, x = snp_reader.get_common_table(genes)

    columns = list(str(i)+j for i, j in product(x.columns, ['_00', '_01', '_11', '_NA']))
    X = np.zeros((x.shape[0], 4*x.shape[1]), dtype=np.int)
    data = x.values
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if data[i, j] == '0|0':
                X[i, 4*j] = 1
            elif data[i, j] == '0|1':
                X[i, 4*j+1] = 1
            elif data[i, j] == '1|1':
                X[i, 4*j+2] = 1
            else:
                X[i][4*j+3] = 1
    X = pd.DataFrame(X, x.index, columns)
    print(X)

    # populations = (
    #     ('EAS', 'JPT'),
    #     ('EUR', 'TSI'),
    #     ('AFR', 'LWK'),
    #     ('SAS', 'PJL'),
    #     ('AMR', 'PEL'),
    # )

    # populations = (
    #     ('EAS', 'JPT'),
    #     ('EAS', 'CHS'),
    #     ('EAS', 'CDX'),
    #     ('EAS', 'KHV'),
    #     ('EAS', 'CHB'),
    # )

    # populations = (
    #     ('EUR', 'GBR'),
    #     ('EUR', 'FIN'),
    #     ('EUR', 'IBS'),
    #     ('EUR', 'CEU'),
    #     ('EUR', 'TSI'),
    # )

    # populations = (
    #     ('SAS', 'PJL'),
    #     ('SAS', 'BEB'),
    #     ('SAS', 'STU'),
    #     ('SAS', 'ITU'),
    #     ('SAS', 'GIH'),
    # )

    # populations = (
    #     ('AFR', 'ACB'),
    #     ('AFR', 'GWD'),
    #     ('AFR', 'ESN'),
    #     ('AFR', 'MSL'),
    #     ('AFR', 'YRI'),
    #     ('AFR', 'LWK'),
    #     ('AFR', 'ASW'),
    # )

    populations = (
        ('AMR', 'PUR'),
        ('AMR', 'CLM'),
        ('AMR', 'PEL'),
        ('AMR', 'MXL'),
    )

    for pop1, pop2 in combinations(populations, 2):
        compute((pop1[0], pop2[0]), (pop1[1], pop2[1]), gene_reader, X, './results/binary_classification_kernel_cold.txt')


if __name__ == "__main__":
    # make_genes_files()
    main()
