#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:14:21 2023

@author: shhe
"""

import os
#import PyPDF2
from pypdf import PdfReader

pdf_dir='training/datasets/'

filelst=[]
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        filelst.append(filename)
        
slst=[5,14]
elst=[191,376]

text = ''
for pn in range(2):
    filename=filelst[pn]
    pdf_file = open(os.path.join(pdf_dir, filename), 'rb')
    pdf_reader = PdfReader(pdf_file)

    #for i in range(1,len(pdf_reader.pages)):
    #    page = pdf_reader.pages[i]
    #    tt=page.extract_text()
    #    try: 
    #        if tt.split('\n')[1]=="Introduction":
    #            startid=i
    #            break
    #    except:
    #        continue
    for i in range(slst[pn],elst[pn]):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    pdf_file.close()
#txt_filename = os.path.splitext(filename)[0] + '.txt'
txt_file = open(os.path.join(pdf_dir, 'DAVID_GOGGINS.txt'), 'w')
txt_file.write(text)
txt_file.close()


#%%
import os
from io import StringIO
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

path=pdf_dir+f1
manager = PDFResourceManager()
retstr = StringIO()
layout = LAParams(all_texts=False, detect_vertical=False)
device = TextConverter(manager, retstr, laparams=layout)
interpreter = PDFPageInterpreter(manager, device)
with open(path, 'rb') as filepath:
    for page in PDFPage.get_pages(filepath, check_extractable=True):
        interpreter.process_page(page)
text = retstr.getvalue()
device.close()
retstr.close()