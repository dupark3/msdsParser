from bs4 import BeautifulSoup as bs
from PIL import Image, ImageSequence

# take command line arguments and create regex with keyword
import sys, re


if len(sys.argv) < 3:
    print('usage: python3 keyword.py htmlfile keyword')
    sys.exit()


htmlfile = sys.argv[1]
keyword = sys.argv[2]

html_doc = open(htmlfile)
soup = bs(html_doc, 'html.parser')

tag = soup.find('span', text=re.compile('^'+keyword))
print(tag)

title_contents = tag['title'].split()

# coordinates saved. last coordinate has a semicolon to be stripped
left, top, right, bottom = [int(word.strip(';')) for word in title_contents[1:5]]

# last value in title is always the word confidence value
word_confidence = title_contents[-1] + '%'
print(left, top, right, bottom, word_confidence)

# ignore comma and newline and go to the next span element
value = tag.next_sibling.next_sibling
print(value)

title_contents = value['title'].split()
leftv, topv, rightv, bottomv = [int(word.strip(';')) for word in title_contents[1:5]]

word_confidence = title_contents[-1] + '%'
pageNumber = value['id'].split('_')[1]

print(leftv, topv, rightv, bottomv, word_confidence)
print('Key found on page ' + pageNumber )

img = Image.open('msds1.tiff')

for i, page in enumerate(ImageSequence.Iterator(img)):
    # i is indexed from 0 but html indexes pages from 1
    if i + 1 == int(pageNumber):
        print(type(page))
        section = page.crop(box= (leftv, topv, rightv, bottomv))
        section.save('section.tiff')
        section.show()
