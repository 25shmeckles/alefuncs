### This is a collection of functions and classes I have initially wrote during my postdoc at the UMCU.
### The following code is meant to be compatible with Python 2.7 and 3.x (minor modifications may be needed in case of Python 3)
### If you use Python 2.7 please include "from __future__ import division, print_function" to your script
### The imports are included in the functions' body for clarity.
### If you need to call a function over and over then consider to move the imports at the beginning of your script.
### This collection is meant to be used by "copy/paste" not as a module (mostly due to the imports),
### altough you can use "from alefuncs import something", "from alefuncs import *", and "import alefuncs as ale", without problems.

### author:  alessio.marcozzi@gmail.com
### version: 2017_02


def print_sbar(n,m,s='|#.|',size=30,message=''):
	'''(int,int,string,int) => None
	Print a progress bar using the simbols in 's'.
	Example:
		range_limit = 1000
		for n in range(range_limit):
			print_sbar(n,m=range_limit)
			time.sleep(0.1) 
	'''
	import sys
	#adjust to bar size
	if m != size:
		n =(n*size)/m
		m = size
	#calculate ticks
	_a = int(n)*s[1]+(int(m)-int(n))*s[2]
	_b = round(n/(int(m))*100,1)
	#adjust overflow
	if _b >= 100:
		_b = 100.0
	#to stdout    
	sys.stdout.write('\r{}{}{}{} {}%     '.format(message,s[0],_a,s[3],_b))
	sys.stdout.flush()


def hash(a_string,algorithm='md5'):
	'''str => str
	Return the hash of a string calculated using various algorithms.
	Example:
		>>> hash('prova','md5')
		'189bbbb00c5f1fb7fba9ad9285f193d1'

		>>> hash('prova','sha256')
		'6258a5e0eb772911d4f92be5b5db0e14511edbe01d1d0ddd1d5a2cb9db9a56ba'
	'''
	import hashlib
	if algorithm == 'md5':
		return hashlib.md5(a_string.encode()).hexdigest()
	elif algorithm == 'sha256':
		return hashlib.sha256(a_string.encode()).hexdigest()
	else:
		raise ValueError('algorithm {} not found'.format(algorithm))


def get_first_transcript_by_gene_name(gene_name):
	'''str => str
	Return the id of the main trascript for a given gene.
	The data is from http://grch37.ensembl.org/
	'''
	from urllib import urlopen
	from pyensembl import EnsemblRelease
	data = EnsemblRelease(75)
	gene = data.genes_by_name(gene_name)
	gene_id = str(gene[0]).split(',')[0].split('=')[-1]
	gene_location = str(gene[0]).split('=')[-1].strip(')')
	url = 'http://grch37.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g={};r={}'.format(gene_id,gene_location)
	for line in urlopen(url):
		if '<tbody><tr><td class="bold">' in line:
			return line.split('">')[2].split('</a>')[0]

   
def get_exons_coord_by_gene_name(gene_name):
	'''str => OrderedDict({'exon_id':[coordinates]})
	Return an OrderedDict having as k the exon_id
	and as value a tuple containing the genomic coordinates ('chr',start,stop).

	Example:
		
	'''
	from pyensembl import EnsemblRelease
	gene = data.genes_by_name(gene_name)
	gene_id = str(gene[0]).split(',')[0].split('=')[-1]
	gene_location = str(gene[0]).split('=')[-1].strip(')')
	gene_transcript = get_first_transcript_by_gene_name(gene_name).split('.')[0]
	table = OrderedDict()
	for exon_id in data.exon_ids_of_gene_id(gene_id):
		exon = data.exon_by_id(exon_id)
		coordinates = (exon.contig, exon.start, exon.end)
		table.update({exon_id:coordinates})
	return table


class Render(QWebPage):
	from PyQt4.QtGui import * 
	from PyQt4.QtCore import * 
	from PyQt4.QtWebKit import *
	def __init__(self, url):  
		self.app = QApplication(sys.argv)  
		QWebPage.__init__(self)  
		self.loadFinished.connect(self._loadFinished)  
		self.mainFrame().load(QUrl(url))  
		self.app.exec_()  

	def _loadFinished(self, result):  
		self.frame = self.mainFrame()  
		self.app.quit()


def get_html(str_url):
	'''url => html
	Return a PyQt4-rendered html page.
	It requires the Render class'''
	r_html = Render(str_url)  
	html = r_html.frame.toHtml()
	return html


def get_exons_coord_by_gene_name(gene_name):
	'''
	Example:
			table = get_exons_coord_by_gene_name('TP53')
			for k,v in table.iteritems():
				print k,v

		>>> ENSE00002419584 ['7,579,721', '7,579,700']
			ENSE00003625790 ['7,579,590', '7,579,312']
			ENSE00003518480 ['7,578,554', '7,578,371']
			ENSE00003462942 ['7,578,289', '7,578,177']
			ENSE00003504863 ['7,577,608', '7,577,499']
			ENSE00003586720 ['7,577,155', '7,577,019']
			ENSE00003636029 ['7,576,926', '7,576,853']
			ENSE00003634848 ['7,574,033', '7,573,927']
			ENSE00002667911 ['7,579,940', '7,579,839']
			ENSE00002051192 ['7,590,799', '7,590,695']
			ENSE00002034209 ['7,573,008', '7,571,722']
			ENSE00003552110 ['7,576,657', '7,576,525']
	'''
	from collections import OrderedDict
	from pyensembl import EnsemblRelease
	data = EnsemblRelease(75)
	gene = data.genes_by_name(gene_name)
	gene_id = str(gene[0]).split(',')[0].split('=')[-1]
	gene_location = str(gene[0]).split('=')[-1].strip(')')
	gene_transcript = get_first_transcript_by_gene_name(gene_name).split('.')[0]
	url = 'http://grch37.ensembl.org/Homo_sapiens/Transcript/Exons?db=core;g={};r={};t={}'.format(gene_id,gene_location,gene_transcript)
	str_html = get_html(url)
	html = ''
	for line in str_html.split('\n'):
		try:
			#print line
			html += str(line)+'\n'
		except UnicodeEncodeError:
			pass
	blocks = html.split('\n')
	table = OrderedDict()
	for exon_id in data.exon_ids_of_gene_id(gene_id):
		for i,txt in enumerate(blocks):
			if exon_id in txt:
				if exon_id not in table:
					table.update({exon_id:[]})
				for item in txt.split('<td style="width:10%;text-align:left">')[1:-1]:
					table[exon_id].append(item.split('</td>')[0])
	return table


def split_overlap(array,size,overlap):
	'''(list,int,int) => [[...],[...],...]
	Split a list into chunks of a specific size and overlap.

	Examples:
		array = list(range(10))
		split_overlap(array,4,2)
		>>> [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]

		array = list(range(11))
		split_overlap(array,4,2)
		>>> [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10]]
	'''
	result = []
	while True:
		if len(array) <= size:
			result.append(array)
			return result
		else:
			result.append(array[:size])
			array = array[size-overlap:]


def reorder_dict(d,keys):
	'''(dict,list) => OrderedDict
	Change the order of a dictionary's keys
	without copying the dictionary (save RAM!).
	Return an OrderedDict.
	'''
	tmp = OrderedDict()
	for k in keys:
		tmp[k] = d[k]
		del d[k] #this saves RAM
	return tmp

#test = OrderedDict({'1':1,'2':2,'4':4,'3':3})
#print(test)
#test2 = reorder_dict(test,['1','2','3','4'])
#print(test)
#print(test2)
#>>> OrderedDict([('2', 2), ('3', 3), ('4', 4), ('1', 1)])
#>>> OrderedDict()
#>>> OrderedDict([('1', 1), ('2', 2), ('3', 3), ('4', 4)])


def in_between(one_number,two_numbers):
	'''(int,list) => bool
	Return true if a number is in between two other numbers.
	Return False otherwise.
	'''
	if two_numbers[0] < two_numbers[1]:
		pass
	else:
		two_numbers = sorted(two_numbers)
	return two_numbers[0] <= one_number <= two_numbers[1]


def is_overlapping(svA,svB,limit=0.9):
	'''(list,list,float) => bool
	Check if two SV ovelaps for at least 90% (limit=0.9).
	svX = [chr1,brk1,chr2,brk2]
	'''
	
	# Step 1.
	# Select the breaks in order to have lower coordinates first
	if int(svA[1]) <= int(svA[3]):
		chr1_A = svA[0]
		brk1_A = int(svA[1])
		chr2_A = svA[2]
		brk2_A = int(svA[3])
	else:
		chr2_A = svA[0]
		brk2_A = svA[1]
		chr1_A = svA[2]
		brk1_A = svA[3]        
	
	if int(svB[1]) <= int(svB[3]):
		chr1_B = svB[0]
		brk1_B = int(svB[1])
		chr2_B = svB[2]
		brk2_B = int(svB[3])
	else:
		chr2_B = svB[0]
		brk2_B = int(svB[1])
		chr1_B = svB[2]
		brk1_B = int(svB[3])
		
	# Step 2.
	# Determine who is the longest
	# Return False immediately if the chromosomes are not the same.
	# This computation is reasonable only for sv on the same chormosome.
	if chr1_A == chr2_A and chr1_B == chr2_B and chr1_A == chr1_B:       
		len_A = brk2_A - brk1_A        
		len_B = brk2_B - brk1_B
		
		if len_A >= len_B:
			len_reference = len_A
			len_sample = len_B
		else:
			len_reference = len_B
			len_sample = len_A
		
		limit = round(len_reference * limit) # this is the minimum overlap the two sv need to share
									 # to be considered overlapping
			
		# if the sample is smaller then the limit then there is no need to go further.
		# the sample segment will never share enough similarity with the reference.
		if len_sample < limit:
			return False
	else:
		return False
	
	# Step 3.
	# Determine if there is an overlap
	# >> There is an overlap if a least one of the break of an sv is in beetween the two breals of the other sv.
	overlapping = False
	for b in [brk1_A,brk2_A]:
		if in_between(b,[brk1_B,brk2_B]):
			overlapping = True
	for b in [brk1_B,brk2_B]:
		if in_between(b,[brk1_A,brk2_A]):
			overlapping = True
			
	if not overlapping:
		return False
	
	# Step 4.
	# Determine the lenght of the ovelapping part
	
	# easy case: if the points are all different then, if I sort the points,
	# the overlap is the region between points[1] and points[2]
	
	# |-----------------|             |---------------------|
	#         |--------------|              |-------------|
	points = sorted([brk1_A,brk2_A,brk1_B,brk2_B])
	if len(set(points)) == 4: # the points are all different
		overlap = points[2]-points[1]
		
	elif len(set(points)) == 3: #one point is in common
		# |-----------------|
		# |--------------|
		if points[0] == points[1]:
			overlap = points[3]-points[2]
			
		# |---------------------|
		#         |-------------|
		if points[2] == points[3]:
			overlap = points[2]-points[1]
			
		# |-----------------|
		#                   |-------------|
		if points[1] == points[2]:
			return False # there is no overlap
	else:
		# |-----------------|
		# |-----------------|
		return True # if two points are in common, then it is the very same sv
	
	if overlap >= limit:
		return True
	else:
		return False


def load_obj(file):
	'''
	Load a pickled object.
	Be aware that pickle is version dependent,
	i.e. objects dumped in Py3 cannot be loaded with Py2.
	'''
	import pickle
	try:
		with open(file,'rb') as f:
			obj = pickle.load(f)
		return obj
	except:
		return False


def save_obj(obj, file):
	'''
	Dump an object with pickle.
	Be aware that pickle is version dependent,
	i.e. objects dumped in Py3 cannot be loaded with Py2.
	'''
	import pickle
	try:
		with open(file,'wb') as f:
			pickle.dump(obj, f)
		print('Object saved to {}'.format(file))
		return True
	except:
		print('Error: Object not saved...')
		return False
	
#save_obj(hotspots_review,'hotspots_review_CIS.txt')


def query_encode(chromosome, start, end):
	'''
	Queries ENCODE via http://promoter.bx.psu.edu/ENCODE/search_human.php
	Parses the output and returns a dictionary of CIS elements found and the relative location.
	'''
	
	## Regex setup
	re1='(chr{})'.format(chromosome) # The specific chromosome
	re2='(:)'    # Any Single Character ':'
	re3='(\\d+)' # Integer
	re4='(-)'    # Any Single Character '-'
	re5='(\\d+)' # Integer
	rg = re.compile(re1+re2+re3+re4+re5,re.IGNORECASE|re.DOTALL)

	## Query ENCODE
	std_link = 'http://promoter.bx.psu.edu/ENCODE/get_human_cis_region.php?assembly=hg19&'
	query = std_link + 'chr=chr{}&start={}&end={}'.format(chromosome,start,end)
	print(query)
	html_doc = urlopen(query)
	html_txt = BeautifulSoup(html_doc, 'html.parser').get_text()
	data = html_txt.split('\n')

	## Parse the output
	parsed = {}
	coordinates = [i for i, item_ in enumerate(data) if item_.strip() == 'Coordinate']
	elements = [data[i-2].split('  ')[-1].replace(': ','') for i in coordinates]
	blocks = [item for item in data if item[:3] == 'chr']
	print(elements)
	
	try:
		i = 0
		for item in elements:
			#print(i)
			try:
				txt = blocks[i]
				#print(txt)
				m = rg.findall(txt)
				bins = [''.join(item) for item in m]
				parsed.update({item:bins})
				i += 1
				print('found {}'.format(item))
			except:
				print('the field {} was empty'.format(item))
		return parsed
	except Exception as e:
		print('ENCODE query falied on chr{}:{}-{}'.format(chromosome, start, end))
		print(e)
		return False


def compare(dict_A,dict_B):
	'''(dict,dict) => dict, dict, dict
	Compares two dicts of bins.
	Returns the shared elements, the unique elements of A and the unique elements of B.
	The dicts shape is supposed to be like this:
		OrderedDict([('1',
					  ['23280000-23290000',
					   '24390000-24400000',
					   ...]),
					 ('2',
					  ['15970000-15980000',
					   '16020000-16030000',
					   ...]),
					 ('3',
					  ['610000-620000',
					   '3250000-3260000',
					   '6850000-6860000',
					   ...])}
	
	'''
	chrms = [str(x) for x in range(1,23)] + ['X','Y']
	
	shared = OrderedDict()
	unique_A = OrderedDict()
	unique_B = OrderedDict()
	for k in chrms:
		shared.update({k:[]})
		unique_A.update({k:[]})
		unique_B.update({k:[]})

		if k in dict_A and k in dict_B:
			for bin_ in dict_A[k]:
				if bin_ in dict_B[k]:
					shared[k].append(bin_)
				else:
					unique_A[k].append(bin_)
			for bin_ in dict_B[k]:
				if bin_ not in shared[k]:
					unique_B[k].append(bin_)
		elif k not in dict_A:
			unique_B[k] = [bin_ for bin_ in dict_B[k]]
		
		elif k not in dict_B:
			unique_A[k] = [bin_ for bin_ in dict_A[k]]
			
	return shared, unique_A, unique_B


#To manage heavy files
def yield_file(infile):
	with open(infile, 'r') as f:
		for line in f:
			if line[0] not in ['#','\n',' ','']:
				yield line.strip()


#Downaload sequence from ensembl
def sequence_from_coordinates(chromosome,strand,start,end,account="a.marcozzi@umcutrecht.nl"):
	'''
	Download the nucleotide sequence from the gene_name.
	This function works only with with GRCh37.
	Inputs can be str or int e.g. 1 or '1' 
	'''
	from Bio import Entrez, SeqIO

	Entrez.email = account # Always tell NCBI who you are

	#GRCh37 from http://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.25/#/def_asm_Primary_Assembly
	NCBI_IDS = {'1':'NC_000001.10','2':'NC_000002.11','3':'NC_000003.11','4':'NC_000004.11',
				'5':'NC_000005.9','6':'NC_000006.11','7':'NC_000007.13','8':'NC_000008.10',
				'9':'NC_000009.11','10':'NC_000010.10','11':'NC_000011.9','12':'NC_000012.11',
				'13':'NC_000013.10','14':'NC_000014.8','15':'NC_000015.9','16':'NC_000016.9',
				'17':'NC_000017.10','18':'NC_000018.9','19':'NC_000019.9','20':'NC_000020.10',
				'21':'NC_000021.8','22':'NC_000022.10','X':'NC_000023.10','Y':'NC_000024.9'}       
  
	try:        
		handle = Entrez.efetch(db="nucleotide", 
							   id=NCBI_IDS[str(chromosome)], 
							   rettype="fasta", 
							   strand=strand, #"1" for the plus strand and "2" for the minus strand.
							   seq_start=start,
							   seq_stop=end)
		record = SeqIO.read(handle, "fasta")
		handle.close()
		sequence = str(record.seq)
		return sequence
	except ValueError:
		print('ValueError: no sequence found in NCBI')
		return False


#GC content calculator    
def gc_content(sequence,percent=True):
	'''
	Return the GC content of a sequence.
	'''
	sequence = sequence.upper()
	g = sequence.count("G")
	c = sequence.count("C")
	t = sequence.count("T")
	a = sequence.count("A")
	gc_count = g+c
	total_bases_count = g+c+t+a
	if total_bases_count == 0:
		print('Error in gc_content(sequence): sequence may contain only Ns')
		return None
	
	try:
		gc_fraction = float(gc_count) / total_bases_count
	except Exception as e:
		print(e)
		print(sequence)
	
	if percent:
		return gc_fraction * 100
	else:
		return gc_fraction
	
	
	   
##Flexibility calculator##  
#requires stabflex3.py

#result handler
class myResultHandler(SFResult):
	
	def report(self,verbose=True):
		self.result = []
		if verbose:
			print("# data description : {}".format(self.description))
			print("# window size : {}".format(self.size))
			print("# window step : {}".format(self.step))
		for i in range(len(self.x)):
			self.result.append(('{}-{}'.format(round(self.x[i]-self.size/2),
											   round(self.x[i]+self.size/2)),
								round(self.y[i],2)))
		if verbose:
			print("# bins examined : {}".format(len(self.result)))
			print("# max flexibility : {}".format(max([v for i,v in self.result])))
		return self.result

			
#main algorithm            
class myFlex(SFAlgorithm): #in line version of the pyflex3.py main class

	def __init__(self, sequence, window_size, step_zize):
		self.size = window_size
		self.step = step_zize
		self.seq = sequence
				
	def analyse(self, data, verbose=False):
		seq = self.seq
		length = len(seq)

		if verbose:
			print("Using {} data from {}".format(data.description, data.source))
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
		
		if verbose:
			print("Finished, elapsed time {} seconds ({} bases/sec)".format(round(elapsed,2),length/elapsed))

		return myResultHandler(self.seq, data, x, y, self.step, self.size,
							   description = data.description,
							   label = data.label,
							   source = data.source)

		
#Endpoint function to calculate the flexibility of a given sequence
def dna_flex(sequence,window_size=500,step_zize=100,verbose=False):
	'''(str,int,int,bool) => list_of_tuples
	Calculate the flexibility index of a sequence.
	Return a list of tuples.
	Each tuple contains the bin's coordinates
	and the calculated flexibility of that bin.
	Example:
		dna_flex(seq_a,500,100)
	>>> [('0-500', 9.7),('100-600', 9.77),...]
	'''
	if verbose:
		print("Algorithm window size : %d" % window_size)
		print("Algorithm window step : %d" % step_zize)
		print("Sequence has {} bases".format(len(self.seq)))
		
	algorithm = myFlex(sequence,window_size,step_zize)
	flexibility_result = algorithm.analyse(flexibility_data)
	
	return flexibility_result.report(verbose)


##Repeats scanner##

#G-quadruplex
def g4_scanner(sequence):
	'''
	G-quadruplex motif scanner.
	Scan a sequence for the presence of the regex motif:
		[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}
		Reference: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1636468/
	Return two callable iterators.
	The first one contains G4 found on the + strand.
	The second contains the complementary G4 found on the + strand, i.e. a G4 in the - strand.
	'''
	#forward G4
	pattern_f = '[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}[ACGT]{1,7}[G]{3,5}'
	result_f = re.finditer(pattern_f, sequence)
	#reverse G4
	pattern_r = '[C]{3,5}[ACGT]{1,7}[C]{3,5}[ACGT]{1,7}[C]{3,5}[ACGT]{1,7}[C]{3,5}'
	result_r = re.finditer(pattern_r, sequence)
	
	return result_f, result_r


#Repeat-masker
def parse_RepeatMasker(infile="RepeatMasker.txt",rep_type='class'):
	'''
	Parse RepeatMasker.txt and return a dict of bins for each chromosome
	and a set of repeats found on that bin.
	
	dict = {'chromosome':{'bin':set(repeats)}}
	'''
	
	chromosomes = [str(c) for c in range(1,23)]+['X','Y']
	result = {}
	
	if rep_type == 'name':
		idx = 10 #repName
	elif rep_type == 'class':
		idx = 11 #repClass
	elif rep_type == 'family':
		idx = 12 #repFamily
	else:
		raise NameError('Invalid rep_type "{}". Expected "class","family" or "name"'.format(rep_type))
			
	#RepeatMasker.txt is around 500MB!
	for line in yield_file(infile):
		data = line.split('\t')
		chromosome = data[5].replace('chr','')
		start = data[6]
		end = data[7]
		bin_ = '{}-{}'.format(start,end)
		repeat = data[idx].replace('?','')
		
		if chromosome in chromosomes:
			if chromosome not in result:
				result.update({chromosome:{bin_:set([repeat])}})
			else:
				if bin_ not in result[chromosome]:
					result[chromosome].update({bin_:set([repeat])})
				else:
					result[chromosome][bin_].add(repeat)
				
	return result

with open(hotspots_by_threshold,'rb') as f:
	hotspots_A = pickle.load(f)
if type(hotspots_A) == list:
	tmp = OrderedDict()
	for c,h,t in hotspots_A:
		if c not in tmp:
			tmp.update({c:[h]})
		else:
			tmp[c] += [h]
	hotspots_A = tmp

print(len(hotspots_A)) # 24      
		
with open(hotspots_top5percent,'rb') as f:
	hotspots_B = pickle.load(f)
print(len(hotspots_B)) # 23 since 1000G does not have chrY


def next_day(d='2012-12-04'):
	'''Return the next day in the calendar.'''
	Y,M,D = d.split('-')
	t = datetime.date(int(Y),int(M),int(D))
	_next = t + datetime.timedelta(1)
	return str(_next)
# next_day('2012-12-31')
# >>> '2013-01-01'


def previous_day(d='2012-12-04'):
	'''Return the previous day in the calendar.'''
	Y,M,D = d.split('-')
	t = datetime.date(int(Y),int(M),int(D))
	_prev = t + datetime.timedelta(-1)
	return str(_prev)
# previous_day('2013-01-01')
# >>> '2012-12-31'


def intersect(list1,list2):
	'''(list,list) => list
	Return the intersection of two lists, i.e. the item in common.
	'''
	return [item for item in list2 if item in list1]


def annotate_fusion_genes(dataset_file):
	'''
	Uses FusionGenes_Annotation.pl to find fusion genes in the dataset.
	Generates a new file containing all the annotations.
	'''
	import time
	start = time.time()
	print('annotating', dataset_file, '...')
	raw_output = run_perl('FusionGenes_Annotation.pl', dataset_file)
	raw_list = str(raw_output)[2:].split('\\n')
	outfile = dataset_file[:-4] + '_annotated.txt'
	with open(outfile, 'w') as outfile:
		line_counter = 0
		header = ['##ID', 'ChrA', 'StartA', 'EndA', 'ChrB', 'StartB', 'EndB', 'CnvType', 'Orientation',
				  'GeneA', 'StrandA', 'LastExonA', 'TotalExonsA', 'PhaseA',
				  'GeneB', 'StrandB', 'LastExonB', 'TotalExonsB', 'PhaseB',
				  'InFrame', 'InPhase']
		outfile.write(list_to_line(header, '\t') + '\n')
		for item in raw_list:
			cleaned_item = item.split('\\t')
			if len(cleaned_item) > 10: # FusionGenes_Annotation.pl return the data twice. We kepp the annotated one.
				outfile.write(list_to_line(cleaned_item, '\t') + '\n')
				line_counter += 1
	print('succesfully annotated',line_counter, 'breakpoints from', dataset_file, 'in', time.time()-start, 'seconds') 
	# track threads
	try:
		global running_threads
		running_threads -= 1
	except:
		pass
# dataset_file = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/Decipher-DeletionsOnly.txt'
# annotate_fusion_genes(dataset_file)


def blastn(input_fasta_file,db_path='/Users/amarcozzi/Desktop/BLAST_DB/',db_name='human_genomic',out_file='blastn_out.xml'):
	'''
	Run blastn on the local machine using a local database.
	Requires NCBI BLAST+ to be installed. http://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download
	Takes a fasta file as input and writes the output in an XML file.
	'''
	from Bio.Blast.Applications import NcbiblastnCommandline
	db = db_path + db_name
	blastn_cline = NcbiblastnCommandline(query=input_fasta_file, db=db, evalue=0.001, outfmt=5, out=out_file)
	print(blastn_cline)
	stdout, stderr = blastn_cline()
# to be tested


def check_line(line,unexpected_char=['\n','',' ','#']):
	'''
	Check if the line starts with an unexpected character.
	If so, return False, else True
	'''
	for item in unexpected_char:
		if line.startswith(item):
			return False
	return True


def dice_coefficient(sequence_a,sequence_b):
	'''(str, str) => float
	Return the dice cofficient of two sequences.
	'''
	a = sequence_a
	b = sequence_b
	if not len(a) or not len(b): return 0.0
	# quick case for true duplicates
	if a == b: return 1.0
	# if a != b, and a or b are single chars, then they can't possibly match
	if len(a) == 1 or len(b) == 1: return 0.0
	
	# list comprehension, preferred over list.append() '''
	a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
	b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
	
	a_bigram_list.sort()
	b_bigram_list.sort()
	
	# assignments to save function calls
	lena = len(a_bigram_list)
	lenb = len(b_bigram_list)
	# initialize match counters
	matches = i = j = 0
	while (i < lena and j < lenb):
		if a_bigram_list[i] == b_bigram_list[j]:
			matches += 2
			i += 1
			j += 1
		elif a_bigram_list[i] < b_bigram_list[j]:
			i += 1
		else:
			j += 1
	
	score = float(matches)/float(lena + lenb)
	return score


def find_path(graph, start, end, path=[]):
	'''
	Find a path between two nodes in a graph.
	Works on graphs like this:
			graph ={'A': ['B', 'C'],
					'B': ['C', 'D'],
					'C': ['D'],
					'D': ['C'],
					'E': ['F'],
					'F': ['C']}
	'''
	path = path + [start]
	if start == end:
		return path
	if not graph.has_key(start):
		return None
	for node in graph[start]:
		if node not in path:
			newpath = find_path(graph, node, end, path)
		if newpath: return newpath
	return None


def find_all_paths(graph, start, end, path=[]):
	'''
	Find all paths between two nodes of a graph.
	Works on graphs like this:
			graph ={'A': ['B', 'C'],
					'B': ['C', 'D'],
					'C': ['D'],
					'D': ['C'],
					'E': ['F'],
					'F': ['C']}
	'''
	path = path + [start]
	if start == end:
		return [path]
	if not graph.has_key(start):
		return []
	paths = []
	for node in graph[start]:
		if node not in path:
			newpaths = find_all_paths(graph, node, end, path)
			for newpath in newpaths:
				paths.append(newpath)
	return paths


def find_shortest_path(graph, start, end, path=[]):
	'''
	Find the shortest path between two nodes of a graph.
	Works on graphs like this:
			graph ={'A': ['B', 'C'],
					'B': ['C', 'D'],
					'C': ['D'],
					'D': ['C'],
					'E': ['F'],
					'F': ['C']}
	'''
	path = path + [start]
	if start == end:
		return path
	if not graph.has_key(start):
		return None
	shortest = None
	for node in graph[start]:
		if node not in path:
			newpath = find_shortest_path(graph, node, end, path)
			if newpath:
				if not shortest or len(newpath) < len(shortest):
					shortest = newpath
	return shortest


# ##
# graph = {'A': ['B', 'C'],
# 		 'B': ['C', 'D'],
# 		 'C': ['D'],
# 		 'D': ['C'],
# 		 'E': ['F'],
# 		 'F': ['C']}

# >>> find_path(graph, 'A', 'D')
#     ['A', 'B', 'C', 'D']

# >>> find_all_paths(graph, 'A', 'D')
#     [['A', 'B', 'C', 'D'], ['A', 'B', 'D'], ['A', 'C', 'D']]

# >>> find_shortest_path(graph, 'A', 'D')
#     ['A', 'C', 'D']

def gen_rnd_string(length):
	'''
	Return a string of uppercase and lowercase ascii letters.
	'''
	import random, string
	s = [l for l in string.ascii_letters]
	random.shuffle(s)
	s = ''.join(s[:length])
	return s

def gene_synonyms(gene_name):
	'''str => list()
	Queries http://rest.genenames.org and returns a list of synonyms of gene_name.
	Returns None if no synonym was found.
	'''
	import httplib2 as http
	import json
	from urllib.parse import urlparse

	result = []
	headers = {'Accept': 'application/json'}

	uri = 'http://rest.genenames.org'
	path = '/search/{}'.format(gene_name)

	target = urlparse(uri+path)
	method = 'GET'
	body = ''

	h = http.Http()

	response, content = h.request(
									target.geturl(),
									method,
									body,
									headers )

	if response['status'] == '200':
		# assume that content is a json reply
		# parse content with the json module 
		data = json.loads(content.decode('utf8'))
		for item in data['response']['docs']:
			result.append(item['symbol'])
		return result
	 
	else:
		print('Error detected: ' + response['status'])
		return None
#print(gene_synonyms('MLL3'))

def string_to_number(s):
	'''
	Convert a bytes string into a single number.
	'''
	return int.from_bytes(s.encode(), 'little')
#string_to_number('foo bar baz')
#>>> 147948829660780569073512294

def number_to_string(n):
	'''
	Convert a number into a bytes string.
	'''
	import math
	return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()
#x = 147948829660780569073512294
#number_to_string(x)
#>>> 'foo bar baz'
 
def determine_average_breaks_distance(dataset): # tested only for deletion/duplication
	'''
	Evaluate the average distance among breaks in a dataset.
	'''
	data = extract_data(dataset, columns=[1,2,4,5], verbose=False)
	to_average = []
	for item in data:
		if item[0] == item[2]:
			to_average.append(int(item[3])-int(item[1]))
	return sum(to_average)/len(to_average)
#print(determine_average_breaks_distance('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/random/sorted/rnd_dataset_100_annotated_sorted.txt'))


def dict_overview(dictionary,how_many_keys):
	'''
	Prints out how_many_elements of the target dictionary.
	Useful to have a quick look at the structure of a dictionary.
	'''
	from itertools import islice
	ks = list(islice(dictionary, how_many_keys))
	for k in ks:
		print(k)
		print(dictionary[k])


def download_human_genome(build='GRCh37', entrez_usr_email="A.E.vanvlimmeren@students.uu.nl"): #beta: works properly only forGRCh37
	'''
	Download the Human genome from enterez.
	Save each chromosome in a separate txt file.
	'''
	from Bio import Entrez, SeqIO

	Entrez.email = entrez_usr_email

	#Last available version
	NCBI_IDS = {'1':"NC_000001", '2':"NC_000002",'3':"NC_000003",'4':"NC_000004",
				'5':"NC_000005",'6':"NC_000006",'7':"NC_000007", '8':"NC_000008",
				'9':"NC_000009", '10':"NC_000010", '11':"NC_000011", '12':"NC_000012",
				'13':"NC_000013",'14':"NC_000014", '15':"NC_000015", '16':"NC_000016", 
				'17':"NC_000017", '18':"NC_000018", '19':"NC_000019", '20':"NC_000020",
				'21':"NC_000021", '22':"NC_000022", 'X':"NC_000023", 'Y':"NC_000024"}


	#GRCh37 from http://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.25/#/def_asm_Primary_Assembly
	NCBI_IDS_GRCh37 = { 'NC_000001.10','NC_000002.11','NC_000003.11','NC_000004.11',
						'NC_000005.9','NC_000006.11','NC_000007.13','NC_000008.10',
						'NC_000009.11','NC_000010.10','NC_000011.9','NC_000012.11',
						'NC_000013.10','NC_000014.8','NC_000015.9','NC_000016.9',
						'NC_000017.10','NC_000018.9','NC_000019.9','NC_000020.10',
						'NC_000021.8','NC_000022.10','NC_000023.10','NC_000024.9'}
	

	CHR_LENGTHS_GRCh37 = {	'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
							'5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
							'9' :141213431,'10':135534747,'11':135006516,'12':133851895,
							'13':115169878,'14':107349540,'15':102531392,'16':90354753,
							'17':81195210,'18':78077248,'19':59128983,'20':63025520,
							'21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}


	if build == 'GRCh37':
		NCBI_IDS = NCBI_IDS_GRCh37
		CHR_LENGTHS = CHR_LENGTHS_GRCh37
	else:
		print('This function only work with genome build GRCh37 fow now...')
		return False


	idx = 0
	for target_chromosome in NCBI_IDS:
		length = CHR_LENGTHS[idx]
		idx += 1

		sequence = False

		try:
				 # Always tell NCBI who you are
			handle = Entrez.efetch(db="nucleotide", 
								   id=target_chromosome, 
								   rettype="fasta", 
								   strand=1, 
								   seq_start=0, #this is to obtain actual start coordinates from the index
								   seq_stop=length) # this is the end of the chromosome
			record = SeqIO.read(handle, "fasta")
			handle.close()
			sequence = str(record.seq)

		except ValueError:
			print('ValueError: no sequence found in NCBI')

		with open('sequence_{}.txt'.format(target_chromosome), 'w') as f:
			f.write(sequence)	


def exponential_range(start=0,end=10000,base=10):
	'''
	Generates a range of integer that grow exponentially.
	Example: list(exp_range(0,100000,2))
	Output :[0,
			 2,
			 4,
			 8,
			 16,
			 32,
			 64,
			 128,
			 256,
			 512,
			 1024,
			 2048,
			 4096,
			 8192,
			 16384,
			 32768,
			 65536]
	'''

	if end/base < base:
		raise ValueError('"end" must be at least "base**2"')
	result = []
	 
	new_start = start
	new_end = base**2
	new_base = base
	
	while new_start < end:
		result.append(range(new_start,new_end,new_base))
		
		new_start = new_end
		new_end = new_start*base
		new_base = new_base*base
			  
	#print(result)
	for item in result:    
		for i in item:
			yield i
##list(exp_range(0,100000,10))


def extract_data(infile, columns=[3,0,1,2,5], header='##', skip_lines_starting_with='#', data_separator='\t', verbose=False ):
	'''
	Extract data from a file. Returns a list of tuples. 
	Each tuple contains the data extracted from one line of the file
	in the indicated columns and with the indicated order.
	'''
	
	extracted_data = []
	header_list = []
	header_flag = 0
	line_counter = 0

	with open(infile) as infile:
		lines = infile.readlines()

	for line in lines: # yield_file(infile) can be used instead
		line_counter += 1

		if line[:len(header)] == header: # checks the header
			header_list = line_to_list(line[len(header):], data_separator)
			header_flag += 1
			if header_flag > 1:
				raise ValueError('More than one line seems to contain the header identificator "' + header + '".')
		elif line[0] == skip_lines_starting_with or line == '' or line == '\n': # skips comments and blank lines
			pass
		else:
			list_ = line_to_list(line, data_separator)
			reduced_list=[]
			for item in columns:
				reduced_list.append(list_[item])
			extracted_data.append(tuple(reduced_list))

	if verbose == True: # Prints out a brief summary
		print('Data extracted from', infile)
		print('Header =', header_list)
		print('Total lines =', line_counter)

	return extracted_data
# extract_data('tables/clinvarCnv.txt', columns=[3,0,1,2,5], header='##', skip_lines_starting_with='#', data_separator='\t', verbose=True)


def extract_Toronto(infile, outfile):
	'''
	Ad hoc function to extract deletions and duplications out of the Toronto Genetic Variants Database.
	Returns a file ready to be annotated with FusionGenes_Annotation.pl .
	'''
	# Extract data from infile
	# Columns are: ID, Chr, Start, End, CNV_Type
	raw_data = extract_data(infile, columns=[0,1,2,3,5], verbose=True )

	# Take only deletions and duplications
	filtered_data = []
	for data in raw_data:
		if "deletion" in data or 'duplication' in data:
			filtered_data.append(data)
	print('len(row_data)      :',len(raw_data))
	print('len(filtered_data) :',len(filtered_data))


	# Write filtered_data to a text file
	header = ['##ID','ChrA','StartA','EndA','ChrB','StartB','EndB','CnvType','Orientation']
	with open(outfile, 'w') as outfile:
		outfile.write(list_to_line(header, '\t') + '\n')
		for item in filtered_data:
			if item[-1] == 'duplication':
				orientation = 'HT'
			elif item[-1] == 'deletion':
				orientation = 'TH'
			else:
				print('ERROR: unable to determine "Orientation"...')
			list_ = [item[0],item[1],item[2],item[2],item[1],item[3],item[3],item[-1].upper(),orientation]
			outfile.write(list_to_line(list_, '\t') + '\n')
	print('Done')
# infile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/GRCh37_hg19_variants_2014-10-16.txt'
# outfile = infile[:-4]+'_DelDupOnly.txt'
# extract_Toronto(infile, outfile)


def extract_Decipher(infile, outfile):
	'''
	Ad hoc function to extract deletions and duplications out of the Decipher Database.
	Returns a file ready to be annotated with FusionGenes_Annotation.pl .
	'''
	# Extract data from infile
	# Columns are: ID, Chr, Start, End, CNV_Type(here expressed as "mean_ratio")
	raw_data = extract_data(infile, columns=[0,3,1,2,4], verbose=True )
	header = ['##ID','ChrA','StartA','EndA','ChrB','StartB','EndB','CnvType','Orientation']
	with open(outfile, 'w') as outfile:
		outfile.write(list_to_line(header, '\t') + '\n')
		for item in raw_data:
			# Convert mean_ratio to CnvType
			if float(item[-1]) > 0:
				CnvType = 'DUPLICATION'
				orientation = 'HT'
			elif float(item[-1]) < 0:
				CnvType = 'DELETION'
				orientation = 'TH'
			else:
				print('ERROR: unable to determine "Orientation"...')
			# Write output
			list_ = [item[0],item[1],item[2],item[2],item[1],item[3],item[3],CnvType,orientation]
			outfile.write(list_to_line(list_, '\t') + '\n')
	print('Done')
# infile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/decipher-hg19_15-01-30.txt'
# outfile = infile[:-4]+'_DelDupOnly.txt'
# extract_Decipher(infile, outfile)


def extract_dgvMerged(infile, outfile):
	'''
	Ad hoc function to extract deletions and losses out of the dgvMerged database.
	Returns a file ready to be annotated with FusionGenes_Annotation.pl .
	'''
	#original_header = '##bin	chrom	chromStart	chromEnd	name	score	strand	thickStart	thickEnd	itemRgb	varType	reference	pubMedId	method	platform	mergedVariants	supportingVariants	sampleSize	observedGains	observedLosses	cohortDescription	genes	samples'
					#	[0]		[1]		[2]			[3]			[4]		[5]		[6]		[7]			[8]			[9]		[10]	[11]		[12]		[13]	[14]		[15]			[16]				[17]		[18]			[19]			[20]				[21]	[22]
	raw_data = extract_data(infile, columns=[4,1,2,3,10], header='##', skip_lines_starting_with='#', data_separator='\t', verbose=False )

	# Take only deletions and losses
	filtered_data = []
	for data in raw_data:
		if "Deletion" in data or 'Loss' in data:
			filtered_data.append(data)
	print('len(row_data)      :',len(raw_data))
	print('len(filtered_data) :',len(filtered_data))

	# Write filtered_data to a text file
	header = ['##ID','ChrA','StartA','EndA','ChrB','StartB','EndB','CnvType','Orientation']
	with open(outfile, 'w') as outfile:
		outfile.write(list_to_line(header, '\t') + '\n')
		for item in filtered_data:
			if item[-1] == 'Deletion' or item[-1] == 'Loss':
				cnv_type = 'DELETION'
				orientation = 'HT'
			# elif item[-1] == 'deletion':
			# 	orientation = 'TH'
			else:
				print('ERROR: unable to determine "Orientation"...')
			list_ = [item[0],item[1][3:],item[2],item[2],item[1][3:],item[3],item[3],cnv_type,orientation]
			outfile.write(list_to_line(list_, '\t') + '\n')
	print('Done')
# ## Extract deletions and Losses from dgvMerged
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks'
# file_name = 'dgvMerged.txt'
# infile = folder + '/' + file_name
# outfile = folder + '/' + 'dgvMerged-DeletionsOnly.txt'
# extract_dgvMerged(infile, outfile)
# ## annotate
# dataset_file = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/dgvMerged-DeletionsOnly.txt'
# annotate_fusion_genes(dataset_file)


def fill_and_sort(pandas_chrSeries):
	'''incomplete pandas.Series => complete and sorted pandas.Series
	Given a pandas.Series in which the first argument is the chromosome name
	and the second argument is a count " [('1', 61), ('3', 28), ..., ('X', 29)]"
	This function returns a new (sorted by chromosome) series with the missing chromosome included as ('Chr_name',0).
	
	This is useful when creating series out of subsets grouped by Chr.
	If the Chr does not contains any event, then it will be excluded from the subset.
	However, expecially for plotting reasons, you may want to have ('Chr',0) in you list instead of a missing Chr.
	
	Example.
	> series = [('1', 61), ('3', 28), ..., ('X', 29)] # in this Series Chr_2 and Chr_Y are missing.
	> fill_and_sort(series)
	>>> [('1', 61), ('2',0), ('3', 28), ..., ('X', 29), ('Y',0)] # this Series have all the chromosomes
	'''
	import pandas
	# add missing ChrA
	CHROMOSOMES = [str(c) for c in range(1,23)]+['X','Y']
	chr_list = CHROMOSOMES[:]
	complete_series = []
	for item in pandas_chrSeries.iteritems():
		chr_list.remove(item[0])
		complete_series.append(item)
	for item in chr_list:
		complete_series.append((item,0))
	
	# sort by chromosome
	sorted_ = []
	for item in CHROMOSOMES:
		for _item in complete_series:
			if _item[0]==item:
				sorted_.append(_item[1])
	return pandas.Series(sorted_, index=CHROMOSOMES)
# counts = [50,9,45,6]
# pandas_chrSeries = pandas.Series(counts, index=['1','4','X','10'])
# print(pandas_chrSeries)
# good_series = fill_and_sort(pandas_chrSeries)
# print(good_series)


def find(string, char):
	'''
	Looks for a character in a sctring and returns its index.
	'''
	# Compared to string.find(), it returns ALL the indexes, not only the first one.
	return [index for index, letter in enumerate(string) if letter == char]
# print(find('alessio', 's'))

def filter_out(word, infile, outfile):
	'''
	Reads a file line by line
	and writes an output file containing only
	the lines that DO NOT contains 'word'.
	'''
	print('Filtering out lines containing',word,'...')
	with open(infile, 'r') as infile:
		lines = infile.readlines()
	with open(outfile, 'w') as outfile:
		for line in lines: # yield_file(infile) can be used instead
			if word not in line:
				outfile.write(line)
	print('Done')
# infile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/Decipher_DelDupOnly.txt'
# outfile = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/breaks/Decipher-DeletionsOnly.txt'
# filter_out('DUPLICATION',infile, outfile)

def flatten2(l):
	'''
	Flat an irregular iterable to a list.
	Python >= 2.6 version.
	'''
	for item in l:
		if isinstance(item, collections.Iterable) and not isinstance(item, basestring):
			for sub in flatten(item):
				yield sub
		else:
			yield item


def flatten(l):
	'''
	Flat an irregular iterable to a list.
	Python >= 3.3 version.
	'''
	for item in l:
		try:
			yield from flatten(item)
		except TypeError:
			yield item

def gene_synonyms(gene_name):
	'''str => list()
	Queries http://rest.genenames.org and http://www.ncbi.nlm.nih.gov/ to figure out the best synonym of gene_name.
	'''
	import httplib2 as http
	import json
	from urllib.parse import urlparse
	from urllib.request import urlopen
	from bs4 import BeautifulSoup
	

	result = []
	tmp = []
	headers = {'Accept': 'application/json'}

	uri = 'http://rest.genenames.org'
	path = '/search/{}'.format(gene_name)
	html_doc = urlopen('http://www.ncbi.nlm.nih.gov/gene/?term={}[sym]'.format(gene_name))
	html_txt = BeautifulSoup(html_doc, 'html.parser').get_text()


	target = urlparse(uri+path)
	method = 'GET'
	body = ''

	h = http.Http()

	response, content = h.request(
									target.geturl(),
									method,
									body,
									headers )

	if response['status'] == '200':
		# assume that content is a json reply
		# parse content with the json module 
		data = json.loads(content.decode('utf8'))
		for item in data['response']['docs']:
			tmp.append(item['symbol'])
	 
	else:
		print('Error detected: ' + response['status'])
		return None

	if len(tmp) > 1:
		for gene in tmp:
			if gene in html_txt:
				result.append(gene)
		return result
	else:
		return tmp
#print(gene_synonyms('MLL3'))

def gen_controls(how_many,chromosome,GapTable_file,outfile):
	global running_threads # in case of multithreading
	list_brkps = gen_rnd_single_break(how_many, chromosome, GapTable_file, verbose=False)
	with open(outfile,'w') as f:
		for item in list_brkps:
			f.write(list_to_line(item,'\t')+'\n')
	running_threads -= 1 # in case of multithreading
# # Generate controls
# import time
# from threading import Thread
# threads = 0
# running_threads = 0
# max_simultaneous_threads = 20
# how_many=9045
# chromosome='9'
# GapTable_file='/Users/alec/Desktop/UMCU_Backup/Projects/Anne_Project/current_brkps_DB/out_ALL_gap.txt'
# while threads < 100:
# 	while running_threads >= max_simultaneous_threads:
# 		time.sleep(1)
# 	running_threads += 1
# 	outfile = '/Users/alec/Desktop/UMCU_Backup/Projects/Anne_Project/current_brkps_DB/out_chr9_control_'+str(threads)+'.txt'
# 	print('thread', threads, '|', 'running threads:',running_threads)
# 	Thread(target=gen_controls, args=(how_many,chromosome,GapTable_file,outfile)).start()
# 	threads += 1

def gen_control_dataset(real_dataset,suffix='_control.txt'):# tested only for deletion/duplication
	'''
	Generates a control dataset ad hoc.
	Takes as input an existing dataset and generates breaks
	in the same chromosomes and with the same distance (+-1bp),
	the position are however randomized.
	'''
	real_data = extract_data(real_dataset, columns=[1,2,4,5,7,8], verbose=False)
	control_data = []
	_id_list = []
	for item in real_data:
		if item[0] == item[2]: # ChrA == ChrB
			
			# generate a unique id
			_id = gen_rnd_id(16)
			while _id in _id_list:
					_id = gen_rnd_id(16)
			_id_list.append(_id)

			chromosome = item[0]
			distance = int(item[3])-int(item[1]) # 
			cnv_type = item[4]
			orientation = item[5]
			breaks = gen_rnd_breaks(how_many=1, chromosome=chromosome,
									min_distance=distance-1, max_distance=distance+1,
									GapTable_file='tables/gap.txt')
			print(breaks)
			control_data.append([_id,chromosome,breaks[0][1],breaks[0][1],chromosome,breaks[0][2],
								 breaks[0][2],cnv_type,orientation])
		else:
			print(item[0],'is no equal to',item[2],'I am skipping these breaks')
	
	header = ['##ID', 'ChrA', 'StartA', 'EndA', 'ChrB', 'StartB', 'EndB', 'CnvType', 'Orientation']
	
	filename = real_dataset[:-4]+ suffix
	with open(filename,'w') as outfile:
		outfile.write(list_to_line(header, '\t') + '\n')
		for item in control_data:
			line = list_to_line(item, '\t')
			print(line)
			outfile.write(line + '\n')

	print('Data written in',filename)
# gen_control_dataset('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/raw/clinvarCnv-DeletionsOnly.txt')

def gen_gap_table(infile='/Users/amarcozzi/Desktop/All_breakpoints_HG19_final.txt', outfile='/Users/amarcozzi/Desktop/All_breakpoints_HG19_gap.txt', resolution=10000):
	'''
	Generates a file containing a list of coordinates 
	for wich no brakpoints have been found in the input file.
	'''
	# Global constants
	CHROMOSOMES = [str(c) for c in range(1,23)]+['X','Y']
	# length of chromosomes based on GRCh37 (Data source: Ensembl genome browser release 68, July 2012)
	# http://jul2012.archive.ensembl.org/Homo_sapiens/Location/Chromosome?r=1:1-1000000
	# http://grch37.ensembl.org/Homo_sapiens/Location/Chromosome?r=1:24626643-24726643
	CHR_LENGTHS = {'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
				   '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
				   '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
				   '13':115169878,'14':107349540,'15':102531392,'16':90354753,
				   '17':81195210,'18':78077248,'19':59128983,'20':63025520,
				   '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}
	gap_list = []
	for Chr in CHROMOSOMES:
		print('-----------------------------------------------------')
		print('Analyzing breakpoints in chromosome',Chr)
		length = CHR_LENGTHS[Chr]
		# determine the intervals given the chromosome length and the resolution
		x_ax = [] # data holder
		y_ax = [] # stores breakpoint counts per inteval
		breakpoint_list = []
		
		# # Extract data from infile, chromosome by chromosome
		# with open(infile, 'r') as f:
		# 	lines = f.readlines()
		# 	for line in lines: # yield_file(infile) can be used instead
		# 		if line.startswith('chr'+Chr+':'):
		# 			tmp = line.split(':')
		# 			breakpoint = tmp[1].split('-')[0]
		# 			breakpoint_list.append(int(breakpoint))
		# print(len(breakpoint_list),'breakpoints found...')

		with open(infile, 'r') as f:
			#lines = f.readlines()
			for line in f:#lines: # yield_file(infile) can be used instead
				if line.startswith(Chr+'\t'):
					tmp = line_to_list(line,'\t')
					breakpoint = tmp[1]
					breakpoint_list.append(int(breakpoint))
		print(len(breakpoint_list),'breakpoints found...')

		for item in range(resolution,length+resolution,resolution):
			x_ax.append(item)
		print('Interval list:',len(x_ax), 'at',resolution,'bases resolution')

		for interval in x_ax:
			count = 0
			to_remove = []
			for breakpoint in breakpoint_list:
				if breakpoint <= interval:
					count += 1
					to_remove.append(breakpoint)
			y_ax.append(count)

			for item in to_remove:
				try:
					breakpoint_list.remove(item)
				except:
					print('Error',item)

		counter = 0
		for idx,count_ in enumerate(y_ax):
			if count_ == 0:
				gap = x_ax[idx]
				gap_list.append((Chr,gap))
				counter += 1
		print('Found', counter,'gaps in chromosome',Chr,'\n')

	with open(outfile, 'w') as f:
		f.write('#Gap table at '+str(resolution)+' bases resolution based on '+infile+'\n')
		f.write('##chrom'+'\t'+'chromStart'+'\t'+'chromEnd'+'\n')

		for item in gap_list:
			line = 'chr'+str(item[0])+'\t'+str(item[1]-resolution)+'\t'+str(item[1])
			f.write(line+'\n')
# import time
# start = time.time()
# gen_gap_table()
# print('Done in',time.time()-start,'seconds')
## Generate a gap table file
# import time
# start = time.time()
# gen_gap_table(infile='/Users/amarcozzi/Desktop/current_brkps_DB/out_ALL.txt', outfile='/Users/amarcozzi/Desktop/current_brkps_DB/out_ALL_gap.txt', resolution=10000)
# print('Done in',time.time()-start,'seconds')

def gen_multiple_controls(real_dataset,how_many):
	'''
	Generates how_many control datasets.
	'''
	n=0
	while n < how_many:
		suffix = '_control_'+str(n)+'.txt'
		#real_dataset = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/raw/dataset_1b.txt'
		gen_control_dataset(real_dataset,suffix)
		n+=1
	print(n,'datasets have been generated')
# gen_multiple_controls('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/raw/dataset_4.txt',1000)
# ## Generate multiple controls of datasets found in a folder
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/random'
# for item in list_of_files(folder,'txt'):
# 	gen_multiple_controls(item,1000)

def gen_deletion_dataset_from_breaks(list_of_breaks, outfile, ID_already=False):
	'''Genrates a proper deletion dataset file out of a list of breaks '''
	# Var names are not pythonic but I think it is better for readibility
	header = ['##ID', 'ChrA', 'StartA', 'EndA', 'ChrB', 'StartB', 'EndB', 'CnvType', 'Orientation']
	ID_list = [] # to check if the ID is already present
	print('writing breakpoints to', outfile, '..........')
	with open(outfile, 'w') as outfile:
		outfile.write(list_to_line(header, '\t') + '\n')
		for item in list_of_breaks:
			if ID_already == False: # the braks do not have an ID
				while True: # checks ID
					ID = gen_rnd_id(8)
					if ID not in ID_list:
						ID_list.append(ID)
						break
				ChrA = ChrB = item[0][3:]
				StartA = EndA = item[1]
				StartB = EndB = item[2]
			else: # the break do have an ID
				ID = item[0] # the ID is supposed to be the first entry
				ChrA = ChrB = item[1][3:]
				StartA = EndA = item[2]
				StartB = EndB = item[3]
			CnvType = 'DELETION'
			Orientation = 'TH'

			line = list_to_line([ID, ChrA, StartA, EndA, ChrB, StartB, EndB, CnvType, Orientation], '\t')
			outfile.write(line + '\n')
	print('OK')
# list_of_breaks = gen_rnd_breaks(how_many=100, chromosome='Y', min_distance=1000, max_distance=15000, GapTable_file='tables/gap.txt')
# gen_deletion_dataset_from_breaks(list_of_breaks, 'test_deletion_dataset.txt')
# ## Generate (m) RANDOM datasets of different length (n)
# for m in range(1000):
# 	for n in [100,1000,10000,100000,1000000]:
# 		outfile = 'rnd_dataset_'+ str(n)+'_'+str(m)+'.txt'
# 		breaks = list()
# 		for chromosome in CHROMOSOMES:	 	
# 		 	breaks.extend(gen_rnd_breaks(how_many=500, chromosome=chromosome, min_distance=0, max_distance=n))
# 		gen_deletion_dataset_from_breaks(breaks, outfile)

def gen_rnd_breaks(how_many=100, chromosome='Y', min_distance=1000, max_distance=15000, GapTable_file='tables/gap.txt'):
	'''Returns tuples containing 1)the chromosome, 2)first breakpoint, 3)second breakpoint
	Keeps only the points that do not appear in te gap table.
	gen_rnd_breaks(int, string, int, int, filepath) => [(chrX, int, int), ...]
	valid chromosomes inputs are "1" to "22" ; "Y" ; "X"
	The chromosome length is based on the build GRCh37/hg19.'''

	import random
	# CHR_LENGTHS is based on GRCh37
	CHR_LENGTHS = {'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
			   '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
			   '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
			   '13':115169878,'14':107349540,'15':102531392,'16':90354753,
			   '17':81195210,'18':78077248,'19':59128983,'20':63025520,
			   '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}

	# Genrates a chromosome-specific gap list
	print('generating', how_many, 'breakpoints in Chr', chromosome, '..........')
	with open(GapTable_file,'r') as infile:
		lines = infile.readlines()
	
	full_gap_list = []
	chr_specific_gap = []
	for line in lines:
		if '#' not in line: # skip comments
			full_gap_list.append(line_to_list(line, '\t'))
	
	for item in full_gap_list:
		if 'chr' + chromosome in item:
			# Database/browser start coordinates differ by 1 base
			chr_specific_gap.append((item[2],item[3]))

	# Merge contiguous gaps
	merged_gaps = []
	n = 0
	left_tick = False
	while n < len(chr_specific_gap):
		if left_tick == False:
			left_tick = chr_specific_gap[n][0]
		try:
			if chr_specific_gap[n][1] == chr_specific_gap[n+1][0]:
				n += 1
			else:
				right_tick = chr_specific_gap[n][1]
				merged_gaps.append((left_tick,right_tick))
				left_tick = False
				n += 1
		except:
			n += 1

	# Genrates breakpoint list
	list_of_breakpoints = []
	while len(list_of_breakpoints) < how_many:
		try:
			start = random.randint(0,CHR_LENGTHS[chromosome])
		except KeyError:
			if chromosome == '23':
				chromosome = 'X'
				start = random.randint(0,CHR_LENGTHS[chromosome])
			elif chromosome == '24':
				chromosome = 'Y'
				start = random.randint(0,CHR_LENGTHS[chromosome])
			else:
				print('ERROR: Wrong chromosome name!!')


		end = random.randint(start+min_distance, start+max_distance)
		are_points_ok = True # assumes that the points are ok
		
		for item in merged_gaps:
			# checks whether the points are ok for real
			if start < int(item[0]) or start > int(item[1]):
				if end < int(item[0]) or end > int(item[1]):
					pass
				else: are_points_ok = False
			else: are_points_ok = False

		if are_points_ok == True:			
			list_of_breakpoints.append(('chr'+chromosome, start, end))
	print('OK')
	return list_of_breakpoints
# print(gen_rnd_breaks(how_many=100, chromosome='Y', min_distance=1000, max_distance=15000, GapTable_file='tables/gap.txt'))

def gen_rnd_id(length):
	import random
	'''Generates a random string made by uppercase ascii chars and digits'''
	chars = string.ascii_uppercase + string.digits
	return ''.join(random.choice(chars) for char in range(length))
# print(gen_rnd_id(16))

#@profile
def gen_rnd_single_break(how_many=100, chromosome='1', GapTable_file='/Users/amarcozzi/Desktop/All_breakpoints_HG19_gap_10k.txt', verbose=False):
	'''Returns tuples containing 1)the chromosome, 2)the breakpoint
	Keeps only the points that do not appear in te gap table.
	gen_rnd_breaks(int, string, filepath) => [(chrX, int), ...]
	valid chromosomes inputs are "1" to "22" ; "Y" ; "X"
	The chromosome length is based on the build GRCh37/hg19.
	Prerequisites: The gap_list file is in the form:
														##chrom	chromStart	chromEnd
														chr1	0	10000
														chr1	30000	40000
														chr1	40000	50000
														chr1	50000	60000
	'''
	import random
	if verbose == True:
		import time
		start_time = time.time()

	# CHR_LENGTHS is based on GRCh37
	CHR_LENGTHS = {'1':249250621,'2' :243199373,'3' :198022430,'4' :191154276,
			   '5' :180915260,'6' :171115067,'7' :159138663,'8' :146364022,
			   '9' :141213431,'10':135534747,'11':135006516,'12':133851895,
			   '13':115169878,'14':107349540,'15':102531392,'16':90354753,
			   '17':81195210,'18':78077248,'19':59128983,'20':63025520,
			   '21':48129895,'22':51304566,'X' :155270560,'Y' :59373566}

	# Genrates a chromosome-specific gap list
	with open(GapTable_file, 'r') as infile:
		lines = infile.readlines()
	
	full_gap_list = []
	chr_specific_gap = []
	for line in lines:
		if '#' not in line: # skip comments
			full_gap_list.append(line_to_list(line, '\t'))
	
	for item in full_gap_list:
		if 'chr' + chromosome in item:
			chr_specific_gap.append((item[1],item[2]))

	# Merge contiguous gaps
	merged_gaps = merge_gaps(chr_specific_gap)
	# merged_gaps = []
	# while len(chr_specific_gap) > 0:
	# 	try:
	# 		if chr_specific_gap[0][1] == chr_specific_gap[1][0]:
	# 			tmp = (chr_specific_gap[0][0],chr_specific_gap[1][1])
	# 			chr_specific_gap.pop(0)
	# 			chr_specific_gap[0] = tmp
	# 		else:
	# 			merged_gaps.append(chr_specific_gap.pop(0))
	# 	except:
	# 		merged_gaps.append(chr_specific_gap.pop(0))

	# Genrates breakpoint list
	if verbose == True: print('generating', how_many, 'breakpoints in Chr', chromosome)
	list_of_breakpoints = []
	while len(list_of_breakpoints) < how_many:
		try:
			start = random.randint(0,CHR_LENGTHS[chromosome])
			# if verbose == True: print(start)
		except KeyError:
			if chromosome == '23':
				chromosome = 'X'
				start = random.randint(0,CHR_LENGTHS[chromosome])
			elif chromosome == '24':
				chromosome = 'Y'
				start = random.randint(0,CHR_LENGTHS[chromosome])
			else:
				print('ERROR: Wrong chromosome name!!')

		#end = random.randint(start+min_distance, start+max_distance)
		are_points_ok = True # assumes that the points are ok
		
		for item in merged_gaps:
			# checks whether the points are ok for real
			if start <= int(item[0]) or start >= int(item[1]):
				pass
			else:
				are_points_ok = False
				if verbose == True: print(start,'is in a gap and will be discarded')
			
		if are_points_ok == True:			
			list_of_breakpoints.append((chromosome, start))
			if verbose == True: print(start,'is OK',len(list_of_breakpoints),'good breaks generated out of',how_many)

	if verbose == True: print(how_many,'breakpoint have been generated in chromosome',chromosome,'in',time.time()-start_time,'seconds')
	return list_of_breakpoints
# gen_rnd_single_break(verbose=True)
# ## Generate single breaks dataset
# import time
# start = time.time()
# breaks_on_1 = gen_rnd_single_break(how_many=19147,verbose=False)
# for item in breaks_on_1:
# 	print(str(item[0])+'\t'+str(item[1]))
# print('Done in', time.time()-start,'seconds..')
# ## Generate a control file
# list_brkps = gen_rnd_single_break(how_many=20873, chromosome='1', GapTable_file='/Users/amarcozzi/Desktop/current_brkps_DB/out_ALL_gap.txt', verbose=True)
# with open('/Users/amarcozzi/Desktop/current_brkps_DB/out_chr1_control.txt','w') as f:
# 	for item in list_brkps:
# 		f.write(list_to_line(item,'\t')+'\n')
# ## Generate multiple controls
# import time
# from threading import Thread
# start_time = time.time()
# threads = 0
# running_threads = 0
# max_simultaneous_threads = 20
# GapTable_file = '/Users/amarcozzi/Desktop/Projects/Anne_Project/current_brkps_DB/out_ALL_gap.txt'
# chromosome = 'Y'
# infile = '/Users/amarcozzi/Desktop/Projects/Anne_Project/current_brkps_DB/out_chr'+chromosome+'.txt'
# how_many = 0
# for line in yield_file(infile):
# 	if line.startswith(chromosome+'\t'):
# 		how_many += 1
# print('found',how_many,'breakpoints in chromosome',chromosome)
# while threads < 100:
# 	while running_threads >= max_simultaneous_threads:
# 		time.sleep(1)
# 	running_threads += 1
# 	outfile = '/Users/amarcozzi/Desktop/Projects/Anne_Project/current_brkps_DB/controls/out_chr'+chromosome+'_control_'+str(threads)+'.txt'
# 	print('thread', threads, '|', 'running threads:',running_threads)
# 	Thread(target=gen_controls, args=(how_many,chromosome,GapTable_file,outfile)).start()
# 	threads += 1
# print('Waiting for threads to finish...')
# while running_threads > 0:
# 	time.sleep(1)
# end_time = time.time()
# print('\nDone in',(end_time-start_time)/60,'minutes')

def kmers_finder(sequence_dict, motif_length, min_repetition):
	'''(dict, int, int) => OrderedDict(sorted(list))
	Find all the motifs long 'motif_length' and repeated at least 'min_repetition' times.
	Return an OrderedDict having motif:repetition as key:value sorted by value. 
	'''
	from collections import OrderedDict 
	from operator import itemgetter
	motif_dict = {}
	for _id, sequence in sequence_dict.items():
		#populate a dictionary of motifs (motif_dict)
		for i in range(len(sequence) - motif_length +1):
			motif = sequence[i:i+motif_length]
			if motif not in motif_dict:
				motif_dict[motif] = 1
			else:
				motif_dict[motif] += 1

	#remove from motif_dict all the motifs repeated less than 'repetition' times
	keys_to_remove = [key for key, value in motif_dict.items() if value < min_repetition]
	for key in keys_to_remove:
		del motif_dict[key]
	
	#Return a sorted dictionary
	return OrderedDict(sorted(motif_dict.items(), key=itemgetter(1), reverse=True))

def kmers_finder_with_mismatches(sequence, motif_length, max_mismatches, most_common=False):
	'''(str, int, int) => sorted(list)
	Find the most frequent k-mers with mismatches in a string.
	Input: A sequence and a pair of integers: motif_length (<=12) and max_mismatch (<= 3).
	Output: An OrderedDict containing all k-mers with up to d mismatches in string.

	Sample Input:	ACGTTGCATGTCGCATGATGCATGAGAGCT 4 1
	Sample Output:	OrderedDict([('ATGC', 5), ('ATGT', 5), ('GATG', 5),...])
	'''
	from collections import OrderedDict
	from operator import itemgetter

	#check passed variables
	if not motif_length <= 12 and motif_length >= 1:
		raise ValueError("motif_length must be between 0 and 12. {} was passed.".format(motif_length))
	if not max_mismatches <= 3 and max_mismatches >= 1:
		raise ValueError("max_mismatch must be between 0 and 3. {} was passed.".format(max_mismatches))

	motif_dict = {}
	for i in range(len(sequence) - motif_length +1):
		motif = sequence[i:i+motif_length]
		if motif not in motif_dict:
			motif_dict[motif] = 1
		else:
			motif_dict[motif] += 1

	motif_dict_with_mismatches = {}
	for kmer in motif_dict:
		motif_dict_with_mismatches.update({kmer:[]})
		for other_kmer in motif_dict:
			mismatches = 0
			for i in range(len(kmer)):
				if kmer[i] != other_kmer[i]:
					mismatches += 1
			if mismatches <= max_mismatches:
				motif_dict_with_mismatches[kmer].append([other_kmer,motif_dict[other_kmer]])

	tmp = {}
	for item in motif_dict_with_mismatches:
		count = 0
		for motif in motif_dict_with_mismatches[item]:
			count += motif[-1]
		tmp.update({item:count})

	result = OrderedDict(sorted(tmp.items(), key=itemgetter(1), reverse=True))
	
	if most_common:
		commons = OrderedDict()
		_max = result.items()[0][1]
		for item in result:
			if result[item] == _max:
				commons.update({item:result[item]})
			else:
				return commons

	return result

def line_to_list(line, char):
	'''Makes a list of string out of a line. Splits the word at char.'''
	# Allows for more customization compared with string.split()
	split_indexes = find(line, char)
	list_ = []
	n = 0
	for index in split_indexes:
		item = line[n:index].replace('\n','').replace('\r','') # cleans up the line
		if item != '': # skips empty 'cells'
			list_.append(item)
		n = index + 1
	list_.append(line[n:].replace('\n','').replace('\r','')) # append the last item	
	return list_
# print(line_to_list('Makes a list of string out of a line. Splits the word at char.', ' '))

def list_to_line(list_, char):
	'''Makes a string out of a list of items'''
	# Allows for more customization compared with string.split()
	string = ''
	for item in list_:
		string += str(item) + char
	return string.rstrip(char) # Removes the last char
#print(list_to_line(['prova', '1', '2', '3', 'prova'], '---'))

def list_of_files(path, extension):
	'''
	Return a list of filepath for each file into path with the target extension.
	'''
	import glob
	return glob.glob(str(path + '/*.' + extension))
# print(list_of_files('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test datasets', 'txt'))

def merge_gaps(gap_list):
	'''
	Merges overlapping gaps in a gap list.
	The gap list is in the form: [('3','4'),('5','6'),('6','7'),('8','9'),('10','11'),('15','16'),('17','18'),('18','19')]
	Returns a new list containing the merged gaps: [('3','4'),('5','7'),('8','9'),('10','11'),('15','16'),('17','19')]
	'''
	merged_gaps = []
	while len(gap_list) > 0:
		try:
			if int(gap_list[0][1]) >= int(gap_list[1][0]):
				tmp = (gap_list[0][0],gap_list[1][1])
				gap_list.pop(0)
				gap_list[0] = tmp
			else:
				merged_gaps.append(gap_list.pop(0))
		except:
			merged_gaps.append(gap_list.pop(0))
	return merged_gaps
# gap_list = [('3','4'),('5','6'),('6','7'),('8','9'),('10','11'),('15','16'),('17','18'),('18','19')]
# expected = [('3','4'),('5','7'),('8','9'),('10','11'),('15','16'),('17','19')]
# prova = merge_gaps(gap_list)
# print(prova)
# print(expected)

def merge_sort(intervals):
	'''
	Merges and sorts the intervals in a list.
	It's an alternative of merge_gaps() that sort the list before merging.
	Should be faster but I haven't campared them yet.
	'''
	sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
	merged = []

	for higher in sorted_by_lower_bound:
		if not merged:
			merged.append(higher)
		else:
			lower = merged[-1]
			# test for intersection between lower and higher:
			# we know via sorting that lower[0] <= higher[0]
			if higher[0] <= lower[1]:
				upper_bound = max(lower[1], higher[1])
				merged[-1] = (lower[0], upper_bound)  # replace by merged interval
			else:
				merged.append(higher)
	return merged 

def multi_threads_fusion_genes_annotation(folder_path, extension, max_simultaneous_threads):
	''' Executes annotate_fusion_genes() for each dataset file in a folder.
	Each execution run on a different thread.'''
	from threading import Thread
	global running_threads
	dataset_files = list_of_files(folder_path, extension)
	threads = 0
	running_threads = 0
	for file_ in dataset_files:
		while running_threads >= max_simultaneous_threads:
			time.sleep(1)
		threads += 1
		running_threads += 1
		print('thread', threads, '|', 'running threads:',running_threads)
		Thread(target=annotate_fusion_genes, args=(file_,)).start() # with multithreading
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public'
# multi_threads_fusion_genes_annotation(folder, 'txt',50)

def pandize_dataset(annotated_dataset, verbose=True):
	'''
	Prepares a dataset to be "pandas ready".
	Takes a file path as input.
	'''
	import pandas
	# Parse
	if verbose == True:
		message = 'parsing ' + annotated_dataset.split('/')[-1]
		spacer = (100-len(message))*'.'
		print(message, spacer)

	dataset = pandas.io.parsers.read_table(annotated_dataset, dtype={'ChrA':'str','ChrB':'str'}, sep='\t', index_col=0)
	if verbose == True:
		print('OK')
	
	# Clean
	if verbose == True:
		message = 'cleaning ' + annotated_dataset.split('/')[-1]
		spacer = (100-len(message))*'.'
		print(message, spacer)

	dataset = dataset.replace('In Frame', 1)
	dataset = dataset.replace('Not in Frame', 0)
	dataset = dataset.replace('In Phase', 1)
	dataset = dataset.replace('Not in Phase', 0)
	if verbose == True:
		print('OK')

	return dataset
# pandize_dataset('test_data_annotated.txt')
# pandize_dataset('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/control_dataset_100-1000-150000_annotated.txt')

def parse_blastXML(infile):
	'''
	Parses a blast outfile (XML).
	'''
	from Bio.Blast import NCBIXML
	for blast_record in NCBIXML.parse(open(infile)):
		for alignment in blast_record.alignments:
			for hsp in alignment.hsps:
				print("*****Alignment****")
				print("sequence:", alignment.title)
				print("length:", alignment.length)
				print("e-value:", hsp.expect)
				print(hsp.query)
				print(hsp.match)
				print(hsp.sbjct)
# to be tested


def reverse(sequence):
	r = ''
	for i in xrange(len(sequence),0,-1):
		r += sequence[i-1]
	return r

def complement(sequence):
	d = {'A':'T',
		 'T':'A',
		 'C':'G',
		 'G':'C'}
	r = ''
	for b in sequence.upper():
		r += d[b]
	return r

def get_mismatches(template,primer,maxerr,overlapped=False):
	import regex
	error = 'e<={}'.format(maxerr)
	return regex.findall('({}){{{}}}'.format(primer,error), template, overlapped=overlapped)


def pcr(template,primer_F,primer_R,circular=False):
	import re
	if circular: ##works only with primers without 5' overhang
		i = template.upper().find(primer_F.upper())
		template = template[i:]+template[:i]
	
	#Find primer_F, or the largest 3'part of it, in template
	for n in range(len(primer_F)):
		ix_F = [m.end() for m in re.finditer(primer_F[n:].upper(),
										   template.upper())]
		if len(ix_F) == 1: #it's unique
			#print(ix_F)
			#print(primer_F[n:])
			break
		n += 1
	#print(ix_F)
	#Find primer_R, or the largest 5'part of it, in template
	rc_R = reverse(complement(primer_R))
	for n in range(len(primer_R)):
		ix = [m.start() for m in re.finditer(rc_R[:n].upper(),
										   template.upper())]
		if len(ix) == 1: #it's unique
			ix_R = ix[:]

		if len(ix) < 1: #it's the largest possible
			#print(ix_R)
			#print(rc_R[:n])
			break
		n += 1
	#Build the product
	return primer_F + template[ix_F[0]:ix_R[0]] + rc_R
##template = 'CTAGAGAGGGCCTATTTCCCATGATT--something--GCCAATTCTGCAGACAAATGGGGTACCCG'
##primer_F = 'GACAAATGGCTCTAGAGAGGGCCTATTTCCCATGATT'
##primer_R = 'TTATGTAACGGGTACCCCATTTGTCTGCAGAATTGGC'
##product = pcr(template,primer_F,primer_R)
##expected = 'GACAAATGGCTCTAGAGAGGGCCTATTTCCCATGATT--something--GCCAATTCTGCAGACAAATGGGGTACCCGTTACATAA'
##expected == result

def pip_upgrade_all():
	'''
	Upgrades all pip-installed packages.
	Requires a bash shell. (Unix)
	'''
	from subprocess import call
	#pip
	print('upgrading pip...')
	call('pip install --upgrade pip', shell=True)
	call("pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U", shell=True)
	#pip2
	print('upgrading pip2...')
	call('pip2 install --upgrade pip2', shell=True)
	call("pip2 freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip2 install -U", shell=True)
	#pip3
	print('upgrading pip3...')
	call('pip3 install --upgrade pip3', shell=True)
	call("pip3 freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U", shell=True)
	#pypy
	print('upgrading pypy-pip...')
	call('pypy -m pip install --upgrade pip',shell=True)
	call("pypy -m pip freeze --local | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pypy -m pip install -U", shell=True)

def probability(p,n,k):
	'''
	Simple probability calculator.
	Calculates what is the probability that k events occur in n trials.
	Each event have p probability of occurring once.
	Example: What is the probability of having 3 Heads by flipping a coin 10 times?
	probability = prob(0.5,10,3)
	print(probability) => (15/128) = 0.1171875
	'''
	from math import factorial
	p = float(p)
	n = float(n)
	k = float(k)
	C = factorial(n) / ( factorial(k) * factorial(n-k) )
	probability = C * (p**k) * (1-p)**(n-k)
	return probability
#from math import factorial
#print(probability(0.5,10,3))
#print(probability(0.5,1,1))

def process(real_dataset):
	'''
	Generates, annotates and sorts a controll dataset for the given real dataset.
	'''
	gen_control_dataset(real_dataset)
	control_filename = real_dataset[:-4]+'_control.txt'

	#annotate_fusion_genes(real_dataset)
	annotate_fusion_genes(control_filename)

	control_filename = control_filename[:-4]+'_annotated.txt'
	#dataset_filename = real_dataset[:-4]+'_annotated.txt'
	
	#sort_dataset(dataset_filename)
	sort_dataset(control_filename)

	print(real_dataset,'processed. All OK.')
#process('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/clinvarCnv-DeletionsOnly.txt')
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/random'
# for item in list_of_files(folder,'txt'):
# 	process(item)


def query_encode(chromosome, start, end):
	'''
	Queries ENCODE via http://promoter.bx.psu.edu/ENCODE/search_human.php
	Parses the output and returns a dictionary of CIS elements found and the relative location.
	'''
	import re
	from urllib.request import urlopen
	from bs4 import BeautifulSoup

	## Regex setup
	re1='(chr{})'.format(chromosome) # The specific chromosome
	re2='(:)'    # Any Single Character ':'
	re3='(\\d+)' # Integer
	re4='(-)'    # Any Single Character '-'
	re5='(\\d+)' # Integer
	rg = re.compile(re1+re2+re3+re4+re5,re.IGNORECASE|re.DOTALL)


	## Query ENCODE
	std_link = 'http://promoter.bx.psu.edu/ENCODE/get_human_cis_region.php?assembly=hg19&'
	query = std_link + 'chr=chr{}&start={}&end={}'.format(chromosome,start,end)
	print(query)
	html_doc = urlopen(query)
	html_txt = BeautifulSoup(html_doc, 'html.parser').get_text()
	data = html_txt.split('\n')


	## Parse the output
	parsed = {}
	coordinates = [i for i, item_ in enumerate(data) if item_.strip() == 'Coordinate']
	elements = [data[i-2].split('  ')[-1].replace(': ','') for i in coordinates]
	blocks = [item for item in data if item[:3] == 'chr']
	#if len(elements) == len(blocks):
	i = 0
	for item in elements:
		txt = blocks[i]
		m = rg.findall(txt)
		bins = [''.join(item) for item in m]
		parsed.update({item:bins})
		i += 1
			
	return parsed
#cis_elements = query_encode(2,10000,20000)


def run_perl(perl_script_file, input_perl_script):
	'''
	Run an external perl script and return its output
	'''
	from subprocess import check_output
	return check_output(["perl", perl_script_file, input_perl_script])
#print(run_perl('FusionGenes_Annotation.pl', 'test_data.txt'))

def run_py(code, interp='python3'):
	'''Run an block of python code using the target interpreter.'''
	from subprocess import check_output
	with open('tmp.py', 'w') as f:
		for line in code.split('\n'):
			f.write(line+'\n')
	return check_output([interpr, 'tmp.py'])


def run_pypy(code, interpr='pypy3'):
	'''Run an block of python code with PyPy'''
	from subprocess import check_output
	with open('tmp.py', 'w') as f:
		for line in code.split('\n'):
			f.write(line+'\n')
	return check_output([interpr, 'tmp.py'])



def sequence_from_coordinates(chromosome,strand,start,end): #beta hg19 only
	'''
	Download the nucleotide sequence from the gene_name.
	'''
	from Bio import Entrez, SeqIO
	Entrez.email = "a.marcozzi@umcutrecht.nl" # Always tell NCBI who you are
	
	#GRCh37 from http://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.25/#/def_asm_Primary_Assembly
	NCBI_IDS = {'1':'NC_000001.10','2':'NC_000002.11','3':'NC_000003.11','4':'NC_000004.11',
				'5':'NC_000005.9','6':'NC_000006.11','7':'NC_000007.13','8':'NC_000008.10',
				'9':'NC_000009.11','10':'NC_000010.10','11':'NC_000011.9','12':'NC_000012.11',
				'13':'NC_000013.10','14':'NC_000014.8','15':'NC_000015.9','16':'NC_000016.9',
				'17':'NC_000017.10','18':'NC_000018.9','19':'NC_000019.9','20':'NC_000020.10',
				'21':'NC_000021.8','22':'NC_000022.10','X':'NC_000023.10','Y':'NC_000024.9'}       
  
	try:        
		handle = Entrez.efetch(db="nucleotide", 
							   id=NCBI_IDS[str(chromosome)], 
							   rettype="fasta", 
							   strand=strand, #"1" for the plus strand and "2" for the minus strand.
							   seq_start=start,
							   seq_stop=end)
		record = SeqIO.read(handle, "fasta")
		handle.close()
		sequence = str(record.seq)
		return sequence
	except ValueError:
		print('ValueError: no sequence found in NCBI')
		return False
#a = sequence_from_coordinates(9,'-',21967751,21994490)
#print(a)

def sequence_from_gene(gene_name): #beta
	'''
	Download the nucleotide sequence from the gene_name.
	'''
	from Bio import Entrez, SeqIO
	from pyensembl import EnsemblRelease
	data = EnsemblRelease(75, auto_download=True)
	Entrez.email = "a.marcozzi@umcutrecht.nl" # Always tell NCBI who you are
	NCBI_IDS = {'1':"NC_000001", '2':"NC_000002",'3':"NC_000003",'4':"NC_000004",
				'5':"NC_000005",'6':"NC_000006",'7':"NC_000007", '8':"NC_000008",
				'9':"NC_000009", '10':"NC_000010", '11':"NC_000011", '12':"NC_000012",
				'13':"NC_000013",'14':"NC_000014", '15':"NC_000015", '16':"NC_000016", 
				'17':"NC_000017", '18':"NC_000018", '19':"NC_000019", '20':"NC_000020",
				'21':"NC_000021", '22':"NC_000022", 'X':"NC_000023", 'Y':"NC_000024"}
	
	gene_obj = data.genes_by_name(gene_name)
	target_chromosome = NCBI_IDS[gene_obj[0].contig]
	seq_start = int(gene_obj[0].start)
	seq_stop = int(gene_obj[0].end)
	strand = 1 if gene_obj[0].strand == '+' else 2
		
	try:
			 
		handle = Entrez.efetch(db="nucleotide", 
							   id=target_chromosome, 
							   rettype="fasta", 
							   strand=strand, #"1" for the plus strand and "2" for the minus strand.
							   seq_start=seq_start,
							   seq_stop=seq_stop)
		record = SeqIO.read(handle, "fasta")
		handle.close()
		sequence = str(record.seq)
		return sequence

	except ValueError:
		print('ValueError: no sequence found in NCBI')
		return False

def sortby_chr(string):
	'''
	Helps to sort datasets grouped by ChrA/B.
	To use with sorted().
	'''
	# since the ChrA/B value is a string, when sorting by chr may return ['1','10','11'...'2','20'...'3'...'X','Y']
	# instead I want sorted() to return ['1','2',...'9','10','11'...'X','Y']
	if string == 'X':
		return 23
	elif string == 'Y':
		return 24
	else:
		return int(string)
# prova = ['1','10','11','9','2','20','3','X','Y']
# print('sorted()', sorted(prova))
# print('sortby_chr()', sorted(prova, key=sortby_chr))

def sort_dataset(dataset_file, overwrite=False):
	'''
	Sort a dataset by ChrA. It helps during plotting
	'''
	from operator import itemgetter
	text = []
	header_counter = 0
	header = False
	print('Sorting...')
	with open(dataset_file, 'r') as infile:
		#lines = infile.readlines()
		for line in infile:
			list_ = line_to_list(line, '\t')
			if line[:2] == '##':
				header = list_
				header_counter += 1
			else:
				text.append(list_)
	#checkpoint
	if header == False or header_counter > 1:
		print('Something is wrong with the header line...', header_counter, header)
		return None		
	# sort by the second element of the list i.e. 'ChrA'
	text.sort(key=lambda x: sortby_chr(itemgetter(1)(x))) 
	# Write output
	if overwrite == False:
		outfile = dataset_file[:-4]+'_sorted.txt'
	else:
		outfile = dataset_files
	with open(outfile, 'w') as outfile:
		outfile.write(list_to_line(header, '\t') + '\n')
		for list_ in text:
			outfile.write(list_to_line(list_, '\t') + '\n')
	print('Done!')
# sort_dataset('test_data.txt')
# folder = '/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public'
# for item in list_of_files(folder, 'txt'):
# 	sort_dataset(item)
# sort_dataset('/home/amarcozz/Documents/Projects/Fusion Genes/Scripts/test_datasets/public/annotated/dgvMerged-DeletionsOnly_annotated.txt')

def split_fasta_file(infile): #beta
	'''
	Split a fasta file containing multiple sequences
	into multiple files containing one sequence each.
	One sequence per file.
	'''
	flag = False
	length = 0
	with open(infile,'r') as f:
		for line in f:
			if line.startswith('>'):
				if flag == False:
					flag = True
					outfile = '{}.txt'.format(line[1:].strip())
					print('writing {}'.format(outfile))
					lines = [line]
				else:
					with open(outfile, 'w') as out:
						for _ in lines:
							out.write(_)
						print('{} bases written'.format(length))
						length = 0

					outfile = '{}.txt'.format(line[1:].strip())
					print('writing {}'.format(outfile))
					lines = [line]
			else:
				lines.append(line)
				length += len(line.strip())

		#Write last file        
		with open(outfile, 'w') as out:
			for _ in lines:
				out.write(_)
		print('{} bases written'.format(length))	

def substract_datasets(infile_1, infile_2, outfile, header=True):
	'''
	Takes two files containing tab delimited data, comapares them and return a file
	containing the data that is present only in infile_2 but not in infile_1.
	The variable by_column is an int that indicates which column to use
	as data reference for the comparison.
	'''
	header2 = False
	comment_line = '# dataset generated by substracting ' + infile_1 + ' to ' + infile_2 + '\n'

	with open(infile_1) as infile_1:
		lines_1 = infile_1.readlines()
	with open(infile_2) as infile_2:
		lines_2 = infile_2.readlines()

	row_to_removes = []
	for line in lines_1:
		if line[0] != '#': # skips comments
			if header == True:
				header2 = True # to use for the second file
				header = False # set back header to false since the first line will be skipped
				first_line = line
				pass
			else:
				item = line_to_list(line, '\t')
				row_to_removes.append(item)

	result_list = []
	for line in lines_2:
		if line[0] != '#': # skips comments
			if header2 == True:
				header2 = False # set back header to false since the first line will be skipped
				pass
			else:
				item = line_to_list(line, '\t')
				if item not in row_to_removes:
					result_list.append(item)
	
	with open(outfile, 'w') as outfile:
		outfile.write(comment_line)
		outfile.write(first_line)
		for item in result_list:
			outfile.write(list_to_line(item, '\t') + '\n')
	print('substraction of two datasets DONE')
# substract_datasets('dataset_1_b.txt', 'dataset_1.txt', 'dataset_1-1b.txt', header=True)

def yield_file(filepath):
	'''
	A simple generator that yield the lines of a file.
	Good to read large file without running out of memory.
	'''
	with open(filepath, 'r') as f:
		for line in f:
			yield line
# for line in yield_file('GRCh37_hg19_variants_2014-10-16.txt'):
# 	print(line[:20])
