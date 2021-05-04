#!/usr/bin/python
# coding: utf-8

import MeCab
import random

def create_title(file_path):
    wakati = MeCab.Tagger(r"-Owakati -d D:\neologd")
    words = []
    combined_title = ""

    with open(file_path, "r", encoding="utf-8") as f:
        titles = f.readlines()
        for title in titles:
            title_words = wakati.parse(title).split()
            words = words + title_words
    
    nouns = [line.split()[0] for line in MeCab.Tagger(r"-Ochasen -d D:\neologd").parse("".join(words)).splitlines()
               if "名詞" in line.split()[-1]]
    print(nouns)

    random.shuffle(words)
    rand_num = random.randint(10,20)
    for i in range(rand_num):
        combined_title = combined_title + words[i]
    
    print(combined_title)

if __name__ == "__main__":
    create_title("./titles.dat")