from snp.data.data_readers import SubjectReader


def get_population_table():
    reader = SubjectReader('../../../data/s_pop.txt')
    super_pops = reader.data.super_pop.unique()

    for sp in super_pops:
        reader.leave_super_pops([sp])
        pops = reader.data['pop'].unique()
        print(f'{sp}:')
        for p in pops:
            print(f'\t{p} - {len(reader.data.loc[reader.data["pop"] == p])}')
        reader.reset_data()


if __name__ == "__main__":
    get_population_table()
