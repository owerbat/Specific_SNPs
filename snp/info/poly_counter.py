from collections import Counter


def parse(filename, pop):
    with open(filename, 'r') as file:
        poly = [line.split('\t')[9:-1] for line in file.readlines() if pop in line]
        poly = [int(y.split(' (')[0]) for x in poly for y in x]
        print(f'{pop}: {Counter(poly).most_common(3)}')


def poly_counter():
    mit_filename = './results/binary_classification_mit.txt'
    for pop in ['CHS', 'CDX', 'KHV', 'CHB', 'JPT']:
        parse(mit_filename, pop)
    for pop in ['GBR', 'FIN', 'IBS', 'CEU', 'TSI']:
        parse(mit_filename, pop)
    for pop in ['PJL', 'BEB', 'STU', 'ITU', 'GIH']:
        parse(mit_filename, pop)
    for pop in ['ACB', 'GWD', 'ESN', 'MSL', 'YRI', 'LWK', 'ASW']:
        parse(mit_filename, pop)
    for pop in ['PUR', 'CLM', 'PEL', 'MXL']:
        parse(mit_filename, pop)

    print('\n----------------------------------\n')

    mit_filename = './results/binary_classification_kernel.txt'
    for pop in ['CHS', 'CDX', 'KHV', 'CHB', 'JPT']:
        parse(mit_filename, pop)
    for pop in ['GBR', 'FIN', 'IBS', 'CEU', 'TSI']:
        parse(mit_filename, pop)
    for pop in ['PJL', 'BEB', 'STU', 'ITU', 'GIH']:
        parse(mit_filename, pop)
    for pop in ['ACB', 'GWD', 'ESN', 'MSL', 'YRI', 'LWK', 'ASW']:
        parse(mit_filename, pop)
    for pop in ['PUR', 'CLM', 'PEL', 'MXL']:
        parse(mit_filename, pop)


if __name__ == "__main__":
    poly_counter()
