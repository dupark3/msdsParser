from subprocess import check_call

tesseractParams = ['tesseract', 'msds1.tiff', 'result', 'hocr']
check_call(tesseractParams)