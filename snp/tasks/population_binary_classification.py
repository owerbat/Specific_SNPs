import numpy as np
from snp.data.data_readers import SNPReader, SubjectReader, GeneChromoReader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


ESTIMATORS = [LogisticRegression]

SNP_FILENAME = '../../../data/ALL.chrMT.phase3_callmom-v0_4.20130502.genotypes.vcf'
SUBJECT_FILENAME = '../../../data/s_pop.txt'
GENE_FILENAME = '../../../data/mart_export.txt'


def main():
    gene_reader = GeneChromoReader(GENE_FILENAME)

    super_pops = ['EAS', 'EUR']
    pops = ['JPT', 'TSI']

    subject_reader = SubjectReader(SUBJECT_FILENAME)
    subject_reader.leave_super_pops(super_pops)
    subject_reader.leave_pops(pops)
    subjects = subject_reader.data['id']

    snp_reader = SNPReader(SNP_FILENAME)
    snp_info, x = snp_reader.split_data()
    x = x.loc[subjects, :]
    y = np.array([subject_reader.data.loc[subject_reader.data['id'] == id]['pop'] for id in subjects]).ravel()

    for Estimator in ESTIMATORS:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=0)
        clf = Estimator().fit(x_train, y_train)
        prediction = clf.predict(x_test)

        print(f'accuracy = {accuracy_score(y_test, prediction)}')

        best_idxs = np.argsort(clf.coef_.ravel())
        best_features = x.columns[best_idxs]
        # print(f'top 10 features: {[gene_reader.get_info(best_features[i]) for i in range(-1, -11, -1)]}')



if __name__ == "__main__":
    main()
