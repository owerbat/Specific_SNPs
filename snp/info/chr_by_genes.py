from snp.data.data_readers import GeneChromoReader


def chr_by_genes(genes):
    reader = GeneChromoReader('../data/mart_export.txt')
    reader.leave_chromos([str(i) for i in range(1, 23)])
    reader.leave_genes(genes)

    result = {}
    for _, row in reader.data.iterrows():
        result.update({row[0]: (int(row[2]), int(row[3]), row[4])})

    for key in sorted(result):
        print(f'{key}: {result[key][2]}')

    return result


if __name__ == "__main__":
    # genes = [
    #     'BMP8A',
    #     'OMA1',
    #     'APPL2',
    #     'TRPV4',
    #     'TRPV1',
    #     'UCP1',
    #     'ADRB1',
    #     'ADRB2',
    #     'MC4R',
    # ]
    genes = [
        'ADRA1A', # 8
        'ADRB3', # 8
        'CIDEA', # 18
        'CREB1', # 2
        'DIO2', # 14
        'FTO', # 16
        'HOXA1', # 7
        'HOXC4', # 12
        'LIPE', # 19
        'LEP', # 7
        'LEPR', # 1
        'NRF1', # 7
        'NRIP1', # 21
        'PLIN1', # 15
        'PLIN2', # 9
        'PLIN3', # 19
        'PLIN5', # 19
        'PPARG', # 3
        'PPARGC1A', # 4
        'PPARGC1B', # 5
        'PRDM16', # 1
        'PRKAR1A', # 17
        'PRKAR2A', # 3
        'PRKAR1B', # 7
        'PRKAR2B', # 7
        'UCP1', # 4
        'UCP2', # 11
        'UCP3' # 11
    ]
    chr_by_genes(genes)
