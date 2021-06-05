import numpy as np
from itertools import combinations
from snp.data.data_readers import SNPReader, SubjectReader, GeneChromoReader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


SNP_FILENAME = '../data/ALL.chrMT.phase3_callmom-v0_4.20130502.genotypes.vcf'
SUBJECT_FILENAME = '../data/s_pop.txt'
GENE_FILENAME = '../data/mart_export.txt'

ESTIMATORS = [
    (LogisticRegression, 'LogisticRegression', {'max_iter': 1000}),
    (SVC, 'SVC', {}),
    (KNeighborsClassifier, 'KNeighborsClassifier', {'n_neighbors': 3}),
    (GaussianNB, 'GaussianNB', {}),
    (DecisionTreeClassifier, 'DecisionTreeClassifier', {}),
    (RandomForestClassifier, 'RandomForestClassifier', {}),
    (XGBClassifier, 'XGBClassifier', {}),
]


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
                try:
                    features[feature] += 1
                except KeyError:
                    features.update({feature: 1})

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
    top_info = np.array([gene_reader.get_info(top_features[i]) for i in range(top_number)]).reshape(top_number, -1, 2)

    with open(filename, 'a') as file:
        for i, feature in enumerate(top_features):
            file.write(f'{feature} ({top_info[i, :, 0]})\t')
        file.write('\n')


def main():
    gene_reader = GeneChromoReader(GENE_FILENAME)
    gene_reader.leave_chromos(['MT'])

    snp_reader = SNPReader(SNP_FILENAME)
    _, x = snp_reader.split_data()

    populations = (
        ('EAS', 'JPT'),
        ('EUR', 'TSI'),
        ('AFR', 'LWK'),
        ('SAS', 'PJL'),
        ('AMR', 'PEL'),
    )

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

    # populations = (
    #     ('AMR', 'PUR'),
    #     ('AMR', 'CLM'),
    #     ('AMR', 'PEL'),
    #     ('AMR', 'MXL'),
    # )

    for pop1, pop2 in combinations(populations, 2):
        compute((pop1[0], pop2[0]), (pop1[1], pop2[1]), gene_reader, x, './results/binary_classification_mit.txt')


if __name__ == "__main__":
    main()
