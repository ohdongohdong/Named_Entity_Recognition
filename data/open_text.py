from __future__ import print_function
import argparse
import os
import codecs
from pandas import Series, DataFrame
import pandas as pd


def text_checker(chn):
    data = pd.read_csv('NER_DB_ne50.csv',sep=',', index_col=False) 

    selected = data.loc[data['Hanmun'] == chn]
    text = selected.values[0][-1]
    
    return text

def open_text(text):
   
    data_path = '/mnt/disk3/ohdonghun/Projects/CRC/data/chosun_utf8'
    
    for (path, dir, files) in os.walk(data_path):
        for filename in files:
            if filename == text:
                with codecs.open(os.path.join(path,filename),encoding='utf-8',mode='r') as f:
                    lines = f.read()   
    print(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('Hanmun', type=str, help='enter Hanmun sentence')
    
    args = parser.parse_args()
    Hanmun = args.Hanmun
    text = text_checker(Hanmun)
    open_text(text)

if __name__=="__main__":
    main()

