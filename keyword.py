from bs4 import BeautifulSoup as bs

htmlfile = 'result.hocr'
keyword = 'CAS#:'

html_doc = open(htmlfile)
soup = bs(html_doc, 'html.parser')

tag = soup.find('span', text=keyword)
print(tag)
title_contents = tag['title'].split()

# coordinates saved. last coordinate has a semicolon to be stripped
left, top, right, bottom = title_contents[1:5]
bottom = bottom[:-1]

# last value in title is always the word confidence value
word_confidence = title_contents[-1]
print(left, top, right, bottom, word_confidence)