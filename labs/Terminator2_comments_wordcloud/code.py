# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import jieba.posseg as pseg
import matplotlib.pyplot as plt
from os import path
import re
import requests
from scipy.misc import imread
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
import numpy as np
import time


def fetch_douban_comments():
    PATTERN = re.compile('"title":(.*?),')
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    cookies = { 'cookie': 'bid=lYQOsRcej_8; __guid=236236167.1056829666077886000.1525765977089.4163; __yadk_uid=oTZbiJ2I8VYoUXCoZzHcWoBroPcym2QB; gr_user_id=24156fa6-1963-48f2-8b87-6a021d165bde; viewed="26708119_24294210_24375031"; ps=y; __utmt=1; _vwo_uuid_v2=DE96132378BF4399896F28BD0E9CFC4FD|8f3c433c345d866ad9849be60f1fb2a0; ue="2287093698@qq.com"; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1527272795%2C%22https%3A%2F%2Faccounts.douban.com%2Fsafety%2Funlock_sms%2Fresetpassword%3Fconfirmation%3Dbf9474e931a2fa9a%26alias%3D%22%5D; _ga=GA1.2.262335411.1525765981; _gid=GA1.2.856273377.1527272802; dbcl2="62325063:TQjdVXw2PtM"; ck=csZ5; monitor_count=11; _pk_id.100001.8cb4=7b30f754efe8428f.1525765980.15.1527272803.1527269181.; _pk_ses.100001.8cb4=*; push_noty_num=0; push_doumail_num=0; __utma=30149280.262335411.1525765981.1527266658.1527269182.62; __utmb=30149280.9.10.1527269182; __utmc=30149280; __utmz=30149280.1527266658.61.22.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/login; __utmv=30149280.6232; ap=1; ll="108289"'}
    start=0
    with open('subjects.txt', 'w', encoding='utf-8') as f:
        while(start<2000):#max is 17499 can be adjusted
            BASE_URL="https://movie.douban.com/subject/1291844/comments?start={}&limit=20&sort=new_score&status=P".format(start)
            r = requests.get(BASE_URL,headers=headers,cookies=cookies)
            soup=BeautifulSoup(r.text,'lxml')
            pattern=soup.find_all('span',{'class':'short'})
            for items in pattern:
                f.write(items.string)
            start+=20
            print(start)
            #time.sleep(1)
    f.close()


def extract_words():
    with open('subjects.txt', 'r') as f:
        news_subjects = f.readlines()
    stop_words = set(line.strip() for line in open('stopwords.txt'))
    newslist = []
    for subject in news_subjects:
        if subject.isspace():
            continue
        # segment words line by line
        p = re.compile("n[a-z0-9]{0,2}")  # n, nr, ns, ... are the flags of nouns
        word_list = pseg.cut(subject)
        for word, flag in word_list:
            if word not in stop_words and p.search(flag) != None:
                newslist.append(word)
    content = {}
    for item in newslist:
        content[item] = content.get(item, 0) + 1
    d = path.dirname(__file__)
    mask_image = np.array(Image.open(path.join(d, "./T2.jpg")))
    #this for pic2
    #wc = WordCloud(font_path="./simhei.ttf", background_color="white", mask=mask_image, max_words=1000,max_font_size=600)
    #this for pic1
    wc = WordCloud(scale=4,font_path="./simhei.ttf", background_color="white", mask=mask_image,max_words=1000)
    #wc.generate(newslist)
    wc.fit_words(content)
    # Display the generated image:
    plt.imshow(wc,interpolation='bilinear')
    plt.axis("off")
    wc.to_file('Terminator2_4_wordcloud.jpg')
    plt.show()
    print("The End")


if __name__ == "__main__":
    fetch_douban_comments()
    extract_words()
