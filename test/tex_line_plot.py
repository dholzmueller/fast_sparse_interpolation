import numpy as np
import os.path

def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result

def writeToFile(filename, content):
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()

datastr = readFromFile("performance_data.csv")
rows = datastr.split("\n")[:-1]
values = [[s for s in r.split(', ')] for r in rows]

dimensions = list(set([int(r[0]) for r in values]))
dimensions.sort()

tex_str = ''

for d in dimensions:
    lines = ['({}, {})'.format(r[2], r[3]) for r in values if int(r[0]) == d]
    tex_str += '\\addplot coordinates {\n' + '\n'.join(lines) + '};\n\\addlegendentry{$d = ' + str(d) + '$}\n'

tex_str = readFromFile('test/tex_head.txt') + tex_str + readFromFile('test/tex_tail.txt')
writeToFile('runtime_plot.tex', tex_str)
