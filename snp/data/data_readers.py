import pandas as pd


class GeneChromoReader:
    def __init__(self, filename):
        self.names = [
            'gene',
            'id',
            'start',
            'end',
            'chromo'
        ]
        self.raw_data = pd.read_csv(filename, sep='\t', header=0, names=self.names)
        self.raw_data = self.raw_data.sort_values(by=['start', 'end'])
        self.data = self.raw_data

    def reset_data(self, filename=None):
        if filename is not None:
            self.raw_data = pd.read_csv(filename, sep='\t', header=0, names=self.names)
        self.data = self.raw_data

    def leave_chromos(self, chromos):
        self.data = self.data.loc[self.data['chromo'].isin(chromos)]

    def remove_chromos(self, chromos):
        self.data = self.data.drop[self.data['chromo'].isin(chromos)]

    def get_info(self, mapinfo):
        result = []
        for _, row in self.data.iterrows():
            if row['start'] <= mapinfo <= row['end']:
                result.append((row['gene'], row['chromo']))
        return result if len(result) > 0 else [('NA', 'NA')]

    def leave_genes(self, genes):
        self.data = self.data.loc[self.data['gene'].isin(genes)]


class SubjectReader:
    def __init__(self, filename):
        self.names = [
            'id',
            'pop',
            'super_pop',
            'gender'
        ]
        self.raw_data = pd.read_csv(filename, sep='\t', header=0, names=self.names)
        self.data = self.raw_data

    def reset_data(self, filename=None):
        if filename is not None:
            self.raw_data = pd.read_csv(filename, sep='\t', header=0, names=self.names)
        self.data = self.raw_data

    def leave_pops(self, pops):
        self.data = self.data.loc[self.data['pop'].isin(pops)]

    def remove_pops(self, pops):
        self.data = self.data.drop[self.data['pop'].isin(pops)]

    def leave_super_pops(self, super_pops):
        self.data = self.data.loc[self.data['super_pop'].isin(super_pops)]

    def remove_super_pops(self, super_pops):
        self.data = self.data.drop[self.data['super_pop'].isin(super_pops)]

    def leave_gender(self, gender):
        self.data = self.data.loc[self.data['gender'] == gender]


class SNPReader:
    def __init__(self, filename):
        '''
        Need to remove # before header line in .vcf file
        '''
        self.raw_data = pd.read_csv(filename, sep='\t', header=0, index_col='POS', comment='#')
        self.data = self.raw_data

    def reset_data(self, filename=None):
        if filename is not None:
            self.raw_data = pd.read_csv(filename, sep='\t', header=0, index_col='POS', names=self.names)
        self.data = self.raw_data

    def leave_chromos(self, chromos):
        self.data = self.data.loc[self.data['CHROM'].isin(chromos)]

    def remove_chromos(self, chromos):
        self.data = self.data.drop[self.data['CHROM'].isin(chromos)]

    def split_data(self):
        return self.data.iloc[:, :8], self.data.iloc[:, 8:].transpose()


class SNPKernelReader:
    def __init__(self, filename=''):
        self.filename = filename

    def get_gene_data(self, start, end, result_filename):
        with open(self.filename, 'r') as file, open(result_filename, 'w') as result_file:
            for _ in range(252):
                file.readline()

            # header = file.readline()[:-1].split('\t')
            # header[0] = header[0][1:]
            result_file.write(file.readline())

            for line in file:
                data = line[:-1].split('\t')
                if start <= int(data[1]) <= end:
                    result_file.write(line)

    def get_common_table(self, genes):
        dfs = [pd.read_csv(f'../data/gene_data/{gene}.vcf', sep='\t', header=0, index_col='POS') for gene in genes]
        common = pd.concat(dfs)
        # print(common.head(10))

        info = common.iloc[:, :9]
        x = common.iloc[:, 9:].transpose()

        x.where(x.isin(('0|0', '0|1', '1|0', '1|1')), '_NA_', inplace=True)
        x.replace('1|0', '0|1', inplace=True)

        return info, x
