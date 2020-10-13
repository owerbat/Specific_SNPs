import numpy as np
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
        return result


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
        self.raw_data = pd.read_csv(filename, sep='\t', header=0, comment='#')
        self.data = self.raw_data

    def reset_data(self, filename=None):
        if filename is not None:
            self.raw_data = pd.read_csv(filename, sep='\t', header=0, names=self.names)
        self.data = self.raw_data

    def leave_chromos(self, chromos):
        self.data = self.data.loc[self.data['CHROM'].isin(chromos)]

    def remove_chromos(self, chromos):
        self.data = self.data.drop[self.data['CHROM'].isin(chromos)]
