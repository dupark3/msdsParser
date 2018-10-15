from PIL import Image, ImageSequence
import pytesseract, warnings, openpyxl
from sys import argv
from subprocess import check_call
import warnings
import os.path

def verify_CAS(word):
    sections = word.split('-')
    if len(sections) != 3:
        return False
    for section in sections:
        if not section.isdigit():
            return False

    # The check digit is found by taking the last digit times 1, the previous digit times 2, the previous digit times 3 tc., adding all these up and computing the sum modulo 10.
    number1 = int(sections[0])
    number2 = int(sections[1])
    checkDigit = int(sections[2])
    checkSum = 0
    count = 1
    while number2 > 0:
        checkSum += (number2 % 10) * count
        number2 //= 10
        count += 1
    while number1 > 0:
        checkSum += (number1 % 10) * count
        number1 //= 10
        count += 1

    return checkDigit == checkSum % 10

def findTargetIndex(target, words):
    if target in words:
        return words.index(target)
    else:
        return -1

if len(argv) < 2:
    print('Usage: ./convert.py pdfname')

# ignore tiff metadata warning expecting 1 entry in tag 282 and 283 of tiff file
warnings.filterwarnings('ignore', message="Metadata Warning")

# if pdf passed, convert to tiff first using subprocess
if argv[1][-3:] == 'pdf':
    pdfname = argv[1]
    imageName = pdfname.split('.')[0] + '.tiff'
    imageMagickParams = ['convert', 
                         '-density', '300', 
                         '-depth', '8', 
                         '-strip', 
                         '-background', 'white', 
                         '-alpha', 'off', 
                         pdfname, imageName]
    check_call(imageMagickParams)
elif argv[1][-4:] == 'tiff':
    imageName = argv[1]
else:
    print('Pass a .pdf file or .tiff file')

img = Image.open(imageName)
outputName = imageName[:-5] + '.txt'


# write into text file if file not found
if not os.path.isfile('./' + outputName):
    outputFile = open(outputName, 'w')
    for i, imgPage in enumerate(ImageSequence.Iterator(img)):
        print('Page <{}>'.format(i))
        outputFile.write(pytesseract.image_to_string(imgPage, lang="eng", config="hocr"))
    outputFile.close()

outputFile = open(outputName, 'r')
words = [word.strip(':') for word in outputFile.read().split()]
words.insert(0, 'N/A')
# parse the string for categories
targetIndex = findTargetIndex('Name', words)
chemicalName = words[targetIndex + 1]
print(chemicalName)

targetIndex = findTargetIndex('CAS', words)
CASNumber = 'N/A'
for i in range(targetIndex, targetIndex + 12, 1):
    if verify_CAS(words[i]):
        CASNumber = words[i]
        break
print(CASNumber)

targetIndex = findTargetIndex('Formula', words)
chemicalFormula = words[targetIndex + 1]
print(''.join(chemicalFormula.split('-')))

targetIndex = findTargetIndex('Molecular', words)
molecularWeight = words[targetIndex + 2]
print(molecularWeight)

targetIndex = findTargetIndex('Boiling', words)
boilingPoint = words[targetIndex + 2]
print(boilingPoint.strip('°C'))

targetIndex = findTargetIndex('Melting', words)
meltingPoint = words[targetIndex + 2]
print(meltingPoint.strip('°C'))

workbook = openpyxl.load_workbook('msdsdata.xlsx')
sheet = workbook['Sheet1']

# chemical_name
# CAS_number
# chemical_formula
# molecular_weight
# boiling_point
# melting_point

sheet.append([chemicalName, CASNumber, chemicalFormula, molecularWeight, boilingPoint, meltingPoint])
workbook.save('msdsdata.xlsx')


