import numpy as np
from snp.data.data_readers import SNPReader, SubjectReader, GeneChromoReader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
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
    (LogisticRegression, 'LogisticRegression', {}),
    (SVC, 'SVC', {}),
    (KNeighborsClassifier, 'KNeighborsClassifier', {'n_neighbors': 3}),
    (GaussianNB, 'GaussianNB', {}),
    (DecisionTreeClassifier, 'DecisionTreeClassifier', {}),
    (RandomForestClassifier, 'RandomForestClassifier', {}),
    (XGBClassifier, 'XGBClassifier', {}),
]


def classify(Estimator, name, params, X, y, n, gene_reader, columns):
    print(name)

    accuracy = []
    features = {}
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

    print(f'\taverage accuracy = {np.mean(accuracy)}')
    print(f'\tmedian accuracy = {np.median(accuracy)}\n')

    if hasattr(clf, 'feature_importances_'):
        top_features = sorted(features.items(), key=lambda a: -a[1])[:n]
        top_features = [f[0] for f in top_features]
        top_info = np.array([gene_reader.get_info(top_features[i]) for i in range(n)]).reshape(n, -1, 2)
        print('\tfeature\tgenes\tchromosomes')
        for i, feature in enumerate(top_features):
            print(f'\t{feature}:\t{top_info[i, :, 0]}\t{top_info[i, :, 1]}')


def main():
    gene_reader = GeneChromoReader(GENE_FILENAME)
    gene_reader.leave_chromos(['MT'])

    super_pops = ['EAS', 'EUR']
    pops = ['JPT', 'TSI']

    subject_reader = SubjectReader(SUBJECT_FILENAME)
    subject_reader.leave_super_pops(super_pops)
    subject_reader.leave_pops(pops)
    subjects = subject_reader.data['id']

    snp_reader = SNPReader(SNP_FILENAME)
    snp_info, x = snp_reader.split_data()
    X = np.array(x.loc[subjects, :])
    y = np.array([subject_reader.data.loc[subject_reader.data['id'] == id]['pop'] for id in subjects]).ravel()

    n = 10
    for Estimator, name, params in ESTIMATORS:
        classify(Estimator, name, params, X, y, n, gene_reader, x.columns)



if __name__ == "__main__":
    main()
