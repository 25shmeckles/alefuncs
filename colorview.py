#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from alefuncs import list_of_files, parse_fastq

fastqs = list(list_of_files('/Users/alec/Downloads/to_ale', 'fq'))
refs = list(list_of_files('/Users/alec/Downloads/to_ale_ref', 'fa'))

for fastq in fastqs:
    _id = fastq.split('/')[-1].split('.fq')[0]
    ref = [f for f in refs if _id in f]
    assert len(ref) == 1
    ref = ref[0]
    print('...')
    print(_id)
    print(fastq)
    print(ref)
    print()
    get_ipython().system('seqtk seq -a $fastq | head -c 500 | dnacol')
    get_ipython().system("echo '\\n'")
    get_ipython().system('cat $ref | dnacol')

