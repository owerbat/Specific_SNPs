from snp.data.data_readers import GeneChromoReader


def chr_by_genes(genes):
    reader = GeneChromoReader('../data/mart_export.txt')
    reader.leave_genes(genes)

    result = {}
    for _, row in reader.data.iterrows():
        print(f'{row[0]}: {row[4]}')
        result.update({row[0]: (int(row[2]), int(row[3]), row[4])})

    return result


if __name__ == "__main__":
    genes = [
        'BMP8A',
        'OMA1',
        'APPL2',
        'TRPV4',
        'TRPV1',
        'UCP1',
        'ADRB1',
        'ADRB2',
        'MC4R',
    ]
    chr_by_genes(genes)
