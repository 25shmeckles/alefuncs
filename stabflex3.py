import array
import sys
import time

from optparse import OptionParser

from Bio import SeqIO
import matplotlib.pyplot as plt


PROGRAM_VERSION = '0.2.1'
PROGRAM_VERSION_NOTES =\
'''
This version is a Python3 adaptation of StabFlex 0.2, made by Alessio Marcozzi <alessio.marcozzi@gmail.com> in May 2015.
'''

PROGRAM_BANNER = 'StabFlex ' + PROGRAM_VERSION + ' - Calculate stability and flexibility plots of dna sequences\n' + PROGRAM_VERSION_NOTES

PROGRAM_COPYRIGHT =\
'''
Copyright (C) 2010 Andrea Bedini <andrea.bedini@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
'''

PROGRAM_LICENCE =\
'''
            GNU GENERAL PUBLIC LICENSE
               Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.
     59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

The full text of the licene can be found at http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
'''

class SFAnalisysData:

    def __init__(self, data, description = "unknown", source = "unknown", label = "unknown"):
        self.data = data
        self.description = description
        self.source = source
        self.label = label
    
    def base_to_index(self, base):
        return 'ATCGN'.index(base.upper())

    def get(self, a, b):
        i = self.base_to_index(a)
        j = self.base_to_index(b)
        return self.data[i][j]

class SFResult:

    def __init__(self, seq, data, x, y, step, size, description, label, source):

        # save SFAnalisysData
        self.data = data

        # save Sequence
        self.seq = seq

        # save calculations
        if len(x) == len(y):
            self.x = x
            self.y = y
        else:
            raise Exception("strange internal error")
        
        self.step = step
        self.size = size
        self.description = description
        self.label = label
        self.source = source

    def save_to_table(self, filename):
        handle = open(filename, "w")
        handle.write('# %s\n' % PROGRAM_BANNER)        
        handle.write("# sequence : %s\n" % self.seq.name)
        handle.write("# data description : %s\n" % self.description)
        handle.write("# data source : %s\n" % self.source)
        handle.write("# data label : %s\n" % self.label)
        handle.write("# window size : %d\n" % self.size)
        handle.write("# window step : %d\n" % self.step)
        for i in range(len(self.x)):
            handle.write("%1.1f %1.1f\n" % (self.x[i], self.y[i]))
        handle.close()

    def load_from_table(filename):
        handle = open(filename, "r")
        import re
        line = handle.readline()
        tmp = re.compile("^ \w+$").search(line)
        if tmp:
            description = tmp
        
    def save_to_postscript(self, filename):
        plt.figure()
        plt.title(self.seq.name + " " + self.description)
#        p.aspect_ratio = 0.50
  
        plt.xlabel("position (kb)")
        x_kb = array.array('f')
        for i in range(len(self.x)):
            x_kb.append(self.x[i] / 1000)
  
        plt.ylabel(self.data.label)
        plt.plot(x_kb, self.y)
        plt.savefig(filename)

class SFAlgorithm:

    def __init__(self, step, size):
        self.step = step
        print("Algorithm window step : %d" % step)
        self.size = size
        print("Algorithm window size : %d" % size)
        

    def load_sequence(self, filename):
        print("Loading sequence from %s" % filename)
        self.seq = SeqIO.read(open(filename, "rU"), "fasta")
        print("Sequence description : " + self.seq.name)
        print("Sequence has %d bases" % len(self.seq))
    
    def analyse(self, data):
        seq = self.seq.seq
        
        length = len(seq)

        print("Using %s data from %s" % (data.description, data.source))
        print("Running ...")

        # start timer
        start = time.clock()

        offset = self.size/2

        i = 0
        x = array.array('f')
        y = array.array('f')
        finished = False
        while not finished:
            try:
                sum = 0
                for j in range(self.size):
                    a = seq[i + j]
                    b = seq[i + j + 1]
                    sum += data.get(a,b) 
                x.append(offset + i)
                y.append(sum / self.size)
                i += self.step
            except IndexError:
                finished = True

        # stop timer
        end = time.clock()

        elapsed = end - start
        print("Finished, elapsed time %.2f seconds " % elapsed +\
        "(%.2f bases/sec)" % (length / elapsed))

        return SFResult(self.seq, data, x, y, self.step, self.size,
                        description = data.description,
                        label = data.label,
                        source = data.source)

stability_data = SFAnalisysData(description = "di-nucleotide stabilities",
                                source = "Breslauer et al (1986) PNAS 83, 3746-3750",
                                label = "free energy (kcal/mo)",
                                data = (
                                    (1.9, 1.5, 1.3, 1.6, 1.6),
                                    (0.9, 1.9, 1.6, 1.9, 1.6),
                                    (1.9, 1.6, 3.1, 3.6, 2.6),
                                    (1.6, 1.3, 3.1, 3.1, 2.3),
                                    (1.6, 1.6, 2.3, 2.6, 2.0),))

flexibility_data = SFAnalisysData(description = "di-nucleotide flexibilities",
                                  source = "Sarai et al (1989) Biochemistry 28, 7842-7849",
                                  label = "deviation of the TWIST angle (degrees)",
                                  data = (
                                    ( 7.6, 25.0, 14.6,  8.2, 13.8),
                                    (12.5,  7.6,  8.8, 10.9,  9.9),
                                    (10.9,  8.2,  7.2,  8.9,  8.8),
                                    ( 8.8, 14.6, 11.1,  7.2, 10.4),
                                    ( 9.9, 13.8, 10.4,  8.8, 10.7),))

def filename_replace_extension(filename, new_extension):
    try:
        index = filename.rindex('.') + 1
        new_filename = filename[0:index] + new_extension
    except ValueError:
        new_filename = filename + new_extension

    return new_filename

def usage():
    'Print usage information'
    print('usage')

def main():

    print(PROGRAM_BANNER)
    print(PROGRAM_COPYRIGHT)
    print(PROGRAM_VERSION_NOTES)
    print()

    # option parsing

    parser = OptionParser(usage = "usage: stabflex [options] sequence")

    parser.set_defaults(show_licence = 0)
    parser.add_option('--licence', action='store_true', dest='show_licence',
                      help='show program licence')
                      
    parser.set_defaults(window_size = 500) #default 500
    parser.add_option('--size', type='int', dest='window_size',
                      help='set the size of sliding window')

    parser.set_defaults(window_step = 100) #default 100   
    parser.add_option('--step', type='int', dest='window_step',
                      help='set the step of sliding window')

    (options, args) = parser.parse_args()

    if options.show_licence:
        print(PROGRAM_LICENCE)
        sys.exit()
    else:
        print("Use the --licence option to get the full licence.\n")
    
    if len(args) != 1:
        parser.print_help()
        sys.exit()

    filename = args[0]

    algorithm = SFAlgorithm(step = options.window_step,
                            size = options.window_size)
    algorithm.load_sequence(filename)

    print('\nCalculating flexibility...')
    flexibility_result = algorithm.analyse(flexibility_data)
        
    result_filename = filename_replace_extension(filename, 'flex.table')
    flexibility_result.save_to_table(result_filename)
    print("Result saved in %s" % result_filename)

    plot_filename = filename_replace_extension(filename, 'flex.pdf')
    flexibility_result.save_to_postscript(plot_filename)
    print("Plot saved in %s" % plot_filename)

    print('\nCalculating stability...')
    stability_result = algorithm.analyse(stability_data)
        
    result_filename = filename_replace_extension(filename, 'stab.table')
    stability_result.save_to_table(result_filename)
    print("Result saved in %s" % result_filename)

    plot_filename = filename_replace_extension(filename, 'stab.pdf')
    stability_result.save_to_postscript(plot_filename)
    print("Plot saved in %s" % plot_filename)

    print('\n#########   Done!   #########')

if __name__ == '__main__':
    main()
