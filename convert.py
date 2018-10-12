from PIL import Image, ImageSequence
import pytesseract
from sys import argv
from subprocess import check_call
import warnings

if len(argv) < 2:
    print('Usage: ./convert.py pdfname')

# ignore tiff metadata warning expecting 1 entry in tag 282 and 283 of tiff file
warnings.filterwarnings('ignore', message="Metadata Warning")

# if pdf passed, convert to tiff first using subprocess
if argv[1][-3:] == 'pdf':
    pdfname = argv[1]
    imagename = pdfname.split('.')[0] + '.tiff'
    imageMagickParams = ['convert', 
                         '-density', '300', 
                         '-depth', '8', 
                         '-strip', 
                         '-background', 'white', 
                         '-alpha', 'off', 
                         pdfname, imagename]
    check_call(imageMagickParams)
else:
    imagename = argv[1]

img = Image.open(imagename)

for i, imgPage in enumerate(ImageSequence.Iterator(img)):
    print('Page <{}>'.format(i))
    print(pytesseract.image_to_string(imgPage, lang="eng", config="hocr"))