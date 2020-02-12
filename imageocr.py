# Import libraries 
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os 
import cv2
  
# def remove_files():
#     dir_path = "C:\\Users\\Kunj\\Downloads\\PDFOCR\\crop_"
#     filesToRemove = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
#     for f in filesToRemove:
#         os.remove(f)


"""Check image resolution"""
def check_image_size(file_name):
    img = Image.open(file_name)
    width, height = img.size
    # print(width)
    # print(height)

    # RETURN TO SYSTEM UPLOAD SCREEN
    if width < 300 or height < 300:
        print("Upload image again")
        return False
    return True

"""Write to file"""
def writetofile(output, outfile):
    # Creating a text file to write the output
    # Open the file in append mode so that  
    # All contents of all images are added to the same file 
    f = open(outfile, "a")
    f.write(output)
    f.close()


""" Process ouput"""
def process_output(file_name, outfile):
    # dir_path = "C:\\Users\\Kunj\\Desktop\\OCR\\crop_\\"
    pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Kunj\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"
    custom_config = "--oem 3 --psm 3 -l eng"
    # output_list = []

    output = pytesseract.image_to_string(file_name,config = custom_config)
    writetofile(output, outfile)


""" Image preprocessing"""
def img_preprocess(file_name, outfile):
    
    # img = cv2.imread(file_name)
    # img_final = cv2.imread(file_name)
    # img2gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("mask.jpg", mask)

    # image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    # cv2.imwrite("and.jpg", image_final)

    # ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    # cv2.imwrite("thresh.jpg", new_img)

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))  # to manipulate the orientation of dilution , large x
    #                                                              # means horizonatally dilating  more,
    #                                                              # large y means vertically dilating more
    # dilated = cv2.dilate(new_img, kernel, iterations=6)  # dilate , more the iteration more the dilation
    # cv2.imwrite('dilated.jpg', dilated)

    # # kernel = np.ones((5,5),np.uint8)
    # # open = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
    # # cv2.imwrite('open.jpg', open)

    # # for cv2.x.x
    # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 vars

    # # for cv3.x.x comment above line and uncomment line below
    # #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # # Bounding boxes
    # index = 0
    # # out = []
    # for contour in contours:
    #     # get rectangle bounding contour
    #     [x, y, w, h] = cv2.boundingRect(contour)

    #     # Don't plot small false positives that aren't text
    #     if w < 50 and h < 50:
    #         continue

    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    #     cropped = img_final[y :y +  h , x : x + w]
        
    #     # corrected = deskew(cropped)
    #     # cv2.imwrite("Deskew.jpg",corrected )

    #     s = 'C:\\Users\\Kunj\\Desktop\\OCR\\crop_\\' + str(index) + '.jpg' 
    #     cv2.imwrite(s , cropped)
    #     index = index + 1

    # # write original image with added contours to disk
    # cv2.imwrite('captcha_result.jpg', img)
    # # print(out)
    process_output(file_name, outfile)


# file_name = "C:\\Users\\Kunj\\Downloads\\PDFOCR\\page_1.jpg"
def imageOCR(file_name):
    if check_image_size(file_name):
        outfile = "out_text_image.txt"
        img_preprocess(file_name ,outfile)
        return outfile