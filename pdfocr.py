# Import libraries 
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os 
  
# Path of the pdf 
# PDF_file = "C:\\Users\\Kunj\\Desktop\\Notes\\Kunj_Mehta_Offer_letter.pdf"

def pdfOCR(PDF_file):

    pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Kunj\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"

    # Store all the pages of the PDF in a variable
    pages = convert_from_path(PDF_file, 500)

    # Counter to store images of each page of PDF to image
    image_counter = 1

    # Iterate through all the pages stored above
    for page in pages:

        filename = "page_"+str(image_counter)+".jpg"

        # Save the image of the page in system
        page.save(filename, 'JPEG')

        # Increment the counter to update filename
        image_counter = image_counter + 1


    # Variable to get count of total number of pages
    filelimit = image_counter-1

    # Creating a text file to write the output
    outfile = "out_text_pdf.txt"

    # Open the file in append mode so that
    # All contents of all images are added to the same file
    f = open(outfile, "a")

    # Iterate from 1 to total number of pages
    for i in range(1, filelimit + 1):

        filename = "page_"+str(i)+".jpg"

        # Recognize the text as string in image using pytesserct
        text = str(((pytesseract.image_to_string(Image.open(filename)))))

        # The recognized text is stored in variable text
        # Any string processing may be applied on text
        # Here, basic formatting has been done:
        # In many PDFs, at line ending, if a word can't
        # be written fully, a 'hyphen' is added.
        # The rest of the word is written in the next line
        # Eg: This is a sample text this word here GeeksF-
        # orGeeks is half on first line, remaining on next.
        # To remove this, we replace every '-\n' to ''.
        text = text.replace('-\n', '')

        # Finally, write the processed text to the file.
        f.write(text)

    # Close the file after writing all the text.
    f.close()
    remove_intermediate_files()
    return outfile

def remove_intermediate_files():
    dir_name = os.getcwd()
    files = os.listdir(dir_name)

    for file in files:
        if file.endswith(".jpg"):
            os.remove(os.path.join(dir_name, file))