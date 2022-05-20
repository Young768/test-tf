statistics.mean(data)

doc = ['data.txt', 'async.txt']

for d in doc:
    print(d)
    with open(d) as f:
        lines = f.readlines()
        print(lines[0])