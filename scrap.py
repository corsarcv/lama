with open('./data/symbols/gainers-52wk.csv', 'r') as f:
    for line in f:
        if line.startswith('"'):
            print('%s,"%s' % (line[1:line.index(' ') ], line[line.index(' ') + 1:-1]))
        else:
            print('%s,%s' % (line[:line.index(' ')], line[line.index(' ') + 1:-1]))