import numpy as np
import re
import xml.etree.cElementTree as ET
from os import listdir
from os.path import join
from nltk.tokenize import word_tokenize
from PorterStemmer import PorterStemmer
import nltk


def readfile(self):
    files = listdir(self.writepath)
    textALL = []
    filePath = []

    files = sorted(files)
    for f in files:
        fullpath = join(self.writepath, f)
        filePath.append(fullpath)
        with open(fullpath, 'rb') as file:
            content = file.read().decode("utf-8").lower()
            textALL.append(content)
    return textALL, files


# 將載下來的XML檔做前處理  拿出title abstract
def preprocesstext(readpath, writepath):
    mypath = "text"
    files = listdir(mypath)
    for f in files:
        r_textpath = join(readpath, f)
        w_textpath = join(writepath, f)
        pre_ncbi(r_textpath, w_textpath)


def pre_ncbi(fullpath, writepath):
    with open(fullpath, 'rb') as file:
        content = file.read().decode("utf-8")
        content = str(content)
        content = re.sub('<sup>.*?</sup>', ' ', content)
        content = re.sub('<sub>.*?</sub>', ' ', content)
        text = ""
        root = ET.fromstring(content)
        num = 1
        for child_of_root in root.iterfind('PubmedArticle/MedlineCitation/Article'):
            try:
                text += child_of_root.find('ArticleTitle').text
                text += "\n"
            except:
                text += ""
            for i in child_of_root.findall('Abstract/AbstractText'):
                try:
                    text += i.text
                except:
                    text += ""
            with open(writepath + str(num), 'wb') as f:
                f.write(text.encode('utf-8'))
            num += 1
            text = ""


# 去除特殊符號
def remove_tag(text):
    string = re.sub('[\s+\.\!\/_\\,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：:.,?$@%@()<>;]+', " ", text)
    return string


# 句子斷字
def tokenize(msg):
    msg = word_tokenize(msg)
    return (msg)


# 文章斷句
def sents_tokenize(msg):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_tokenizer.tokenize(msg)
    return (sents)


# poter
def poter(word):
    p = PorterStemmer()
    return p.stem(word, 0, len(word) - 1)


