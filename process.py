# import libraries here
import datetime
import os
import sys

import cv2
import numpy as np
import pyocr
import pyocr.builders
from PIL import Image
from matplotlib import pyplot as plt


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """
    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


def extract_info(image_path: str) -> Person:
    """
    Procedura prima putanju do slike sa koje treba ocitati vrednosti, a vraca objekat tipa Person, koji predstavlja osobu sa licnog dokumenta.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param image_path: <str> Putanja do slike za obradu
    :return: Objekat tipa "Person", gde su svi atributi setovani na izvucene vrednosti
    """
    #pyocr.TOOLS[0].TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR' + os.sep + 'tesseract.exe' if os.name == 'nt' else 'tesseract'
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    # odaberemo Tessract - prvi na listi ako je jedini alat
    tool = tools[0]
    print("Koristimo backend: %s" % (tool.get_name()))
    # biramo jezik očekivanog teksta
    lang = 'eng'

    image = load_image(image_path)
    #display_image(image)
    img_gray = image_gray(image)
    #display_image(img_gray)
    img_bin = image_bin(img_gray)
    #display_image(img_bin)
    edges = cv2.Canny(img_gray, threshold1=200, threshold2=255)
    #display_image(edges)

    cropped_image = find_contours(image, edges)
    cropped_image = resize_image(cropped_image)
    #display_image(cropped_image)
    img_gray = image_gray(cropped_image)
    #display_image(img_gray)
    img_bin = image_bin(img_gray)
    #display_image(img_bin)
    edges = cv2.Canny(img_gray, threshold1=200, threshold2=255)
    #display_image(edges)
    roi = detect_qr(cropped_image, edges)

    person = extract_text(tool, lang, cropped_image, roi)

    """text = tool.image_to_string(
        Image.fromarray(img_bin),
        lang=lang,
        builder=pyocr.builders.TextBuilder(tesseract_layout=1)  # izbor segmentacije (PSM)
    )
    # txt is a Python string
    print(text)"""

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    return person


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    #image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 105, 15)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #image_bin = cv2.threshold(cv2.GaussianBlur(image_gs, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #image_bin = cv2.threshold(cv2.bilateralFilter(image_gs, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #image_bin = cv2.adaptiveThreshold(cv2.GaussianBlur(image_gs, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #image_bin = cv2.adaptiveThreshold(cv2.bilateralFilter(image_gs, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return image_bin


def image_bin2(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image_bin


def image_bin_hsv(image):
    image_conv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = image_conv[:, :, 0], image_conv[:, :, 1], image_conv[:, :, 2]
    reth, image_bin = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
    rets, image_bin = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    image_bin = cv2.inRange(image_conv, (0, 0, 0), (179, 255, 85))

    return image_bin


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def resize_image(image):
    return cv2.resize(image, (600, 400), cv2.INTER_CUBIC)


def invert_image(image):
    return 255-image


def find_contours(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.099 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (36, 255, 12), 2)"""
    max_contour = contours[0]
    max_contour_size = cv2.minAreaRect(max_contour)[1]
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        width, height = size
        if width*height > max_contour_size[0]*max_contour_size[1]:
            max_contour = contour
            max_contour_size = size
        #print(center, ' ', size, ' ', angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #x, y, w, h = cv2.boundingRect(contour)
        #region = image_bin[y:y + h + 1, x:x + w + 1]
        #cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 5)
        #cv2.drawContours(image_orig, [box], 0, (255, 0, 255), 2)

    rect = cv2.minAreaRect(max_contour)
    # print(center, ' ', size, ' ', angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # x, y, w, h = cv2.boundingRect(contour)
    # region = image_bin[y:y + h + 1, x:x + w + 1]
    # cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 5)
    #cv2.drawContours(image_orig, [box], 0, (0, 0, 255), 3)

    #display_image(image_orig)
    cropped_image = crop_minAreaRect(image_orig, rect)
    #display_image(cropped_image)

    return cropped_image


def crop_minAreaRect(img, rect):
    """
    Kod za rotiranje i kropovanje preuzet sa https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python#:~:text=Here%27s%20the%20code%20to%20perform,croppedH))%2C%20(size%5B0%5D/2%2C%20size%5B
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2 - x1, y2 - y1)
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))

    return croppedRotated


def detect_qr(image, image_binary):
    #display_image(image_binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    #display_image(close)

    img, contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ROI = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        ar = w / float(h)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 1)
        if len(approx) == 4 and 3500 < area < 6500 and (.9 < ar < 1.1):
            print(area)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            ROI = (x, y, w, h), 'qr'
    #display_image(image)
    if ROI is not None:
        return ROI

    img_bin = image_bin_hsv(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=3)
    #display_image(img_bin)
    img, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        regions_array.append((x, y, w, h))
        area = cv2.contourArea(c)
        ar = w / float(h)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if len(approx) == 4 and 4000 < area < 15000 and (3 < ar < 8):
            print(area)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            ROI = (x, y, w, h), 'horizontal_barcode'
    #display_image(image)
    """sorted_rectangles = sorted(regions_array, key=lambda item: item[0])
    i = 0
    while i < len(sorted_rectangles) - 1:
        if sorted_rectangles[i+1][1] - 2 < sorted_rectangles[i][1] < sorted_rectangles[i+1][1] + 2 and \
                sorted_rectangles[i+1][3] - 2 < sorted_rectangles[i][3] < sorted_rectangles[i+1][3] + 2:
            new_rec = (sorted_rectangles[i][0], sorted_rectangles[i][1], sorted_rectangles[i+1][0] -
                       sorted_rectangles[i][0] + sorted_rectangles[i+1][2], sorted_rectangles[i][3])
            sorted_rectangles.pop(i)
            sorted_rectangles.pop(i)
            sorted_rectangles.insert(i, new_rec)
            i -= 1
        i += 1
    for x, y, w, h in sorted_rectangles:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    display_image(image)"""
    if ROI is not None:
        return ROI

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        ar = w / float(h)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        if len(approx) == 4 and 4500 < area < 10500 and (0.1 < ar < 0.3):
            print(area)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            ROI = (x, y, w, h), 'vertical_barcode'
    #display_image(image)
    if ROI is not None:
        return ROI

    return None, None


def extract_text(tool, lang, image, roi):
    rect, type = roi
    person = Person('test', datetime.date.today(), 'test', 'test', 'test')
    if type == 'qr':
        x, y, w, h = rect
        # cv2.rectangle(cropped_image, (x + 7*w//2, y - 9*h//3), (x + 7*w//2 + 2*w, y - 9*h//3 + h//2), (255, 255, 0), 2)
        #name_img = image[y - 5 * h // 3:y - 5 * h // 3 + h // 2, x + 5 * w // 2:x + 5 * w // 2 + 4 * w]
        #ssn_img = image[y - 3 * h // 3:y - 3 * h // 3 + h // 2, x + 5 * w // 2:x + 5 * w // 2 + 4 * w]
        # display_image(name_img)
        # display_image(ssn_img)
        # display_image(cropped_image)
        try:
            name = tool.image_to_string(
                Image.fromarray(
                    image_bin(image_gray(image))[y - 5 * h // 3:y - 5 * h // 3 + h // 2, x + 5 * w // 2:x + 5 * w // 2 + 4 * w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            ssn = tool.image_to_string(
                Image.fromarray(
                    image_bin(image_gray(image))[y - 3 * h // 3:y - 3 * h // 3 + h // 2, x + 5 * w // 2:x + 5 * w // 2 + 4 * w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            job = tool.image_to_string(
                Image.fromarray(
                    image[y - 1 * h // 3:y - 1 * h // 3 + h // 2, x + 7 * w // 2:x + 7 * w // 2 + 3 * w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            dob = tool.image_to_string(
                Image.fromarray(
                    image[y + 1 * h // 3:y + 1 * h // 3 + h // 2, x + 7 * w // 2:x + 7 * w // 2 + 3 * w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            company = tool.image_to_string(
                Image.fromarray(
                    image[y - 9 * h // 3:y - 9 * h // 3 + h // 2, x + 7 * w // 2:x + 7 * w // 2 + 2 * w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
        except:
            return Person('test', datetime.date.today(), 'test', 'test', 'IBM')
        print(name)
        print(ssn)
        print(job)
        print(dob)
        print(company)
        try:
            dob = datetime.datetime.strptime(dob, '%d, %b %Y')
        except:
            dob = datetime.date.today()
        person = Person(name, dob, job, ssn, 'IBM')
    elif type == 'horizontal_barcode':
        #display_image(image)
        x, y, w, h = rect
        #image_copy = image.copy()
        #cv2.rectangle(image_copy, (x - 4*w//3, y + 12 * h // 3), (x - 4*w//3 + 2*w//2, y + 12 * h // 3 + h), (255, 255, 0), 2)
        #name_img = invert_image(image_gray(image))[y:y + h, x - 4*w//2:x - 4*w//2 + 3*w//2]
        # ssn_img = image[y - 3 * h // 3:y - 3 * h // 3 + h // 2, x + 5 * w // 2:x + 5 * w // 2 + 4 * w]
        #display_image(name_img)
        # display_image(ssn_img)
        #display_image(image_copy)
        try:
            name = tool.image_to_string(
                Image.fromarray(
                    invert_image(image_gray(image))[y:y + h, x - 4*w//2:x - 4*w//2 + 3*w//2]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            ssn = tool.image_to_string(
                Image.fromarray(
                    invert_image(image_gray(image))[y + 5 * h // 3:y + 5 * h // 3 + h, x - 4*w//3:x - 4*w//3 + 2*w//2]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            job = tool.image_to_string(
                Image.fromarray(
                    invert_image(image_gray(image))[y + 8 * h // 3:y + 8 * h // 3 + h, x - 4*w//3:x - 4*w//3 + 2*w//2]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            dob = tool.image_to_string(
                Image.fromarray(
                    invert_image(image_gray(image))[y + 12 * h // 3:y + 12 * h // 3 + h, x - 4*w//3:x - 4*w//3 + 2*w//2]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
        except:
            return Person('test', datetime.date.today(), 'test', 'test', 'Google')
        print(name)
        print(ssn)
        print(job)
        print(dob)
        try:
            dob = datetime.datetime.strptime(dob, '%d, %b %Y')
        except:
            dob = datetime.date.today()
        print('Google')
        person = Person(name, dob, job, ssn, 'Google')
    elif type == 'vertical_barcode':
        # display_image(image)
        x, y, w, h = rect
        #image_copy = image.copy()
        #cv2.rectangle(image_copy, (x + 2*w, y + 5*h//8), (x + 2*w + 6*w, y + 5*h//8 + h//8), (255, 255, 0), 2)
        #name_img = image_gray(image)[y:y + w, x + 1*h//3:x + 1*h//3 + 2*h//2]
        # ssn_img = image[y - 3 * h // 3:y - 3 * h // 3 + h // 2, x + 5 * w // 2:x + 5 * w // 2 + 4 * w]
        #display_image(name_img)
        # display_image(ssn_img)
        #display_image(image_copy)
        try:
            job = tool.image_to_string(
                Image.fromarray(
                    image_gray(image)[y:y + h//8, x + 2*w:x + 2*w + 6*w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            name = tool.image_to_string(
                Image.fromarray(
                    image[y + h//8:y + h//8 + h//8, x + 2*w:x + 2*w + 6*w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            dob = tool.image_to_string(
                Image.fromarray(
                    image[y + 3*h//8:y + 3*h//8 + h//8, x + 2*w:x + 2*w + 6*w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
            ssn = tool.image_to_string(
                Image.fromarray(
                    image[y + 5*h//8:y + 5*h//8 + h//8, x + 2*w:x + 2*w + 6*w]),
                lang=lang,
                builder=pyocr.builders.TextBuilder(tesseract_layout=7)  # izbor segmentacije (PSM)
            )
        except:
            return Person('test', datetime.date.today(), 'test', 'test', 'Apple')
        print(name)
        print(ssn)
        print(job)
        print(dob)
        try:
            dob = datetime.datetime.strptime(dob, '%d, %b %Y')
        except:
            dob = datetime.date.today()
        print('Apple')
        person = Person(name, dob, job, ssn, 'Apple')
    return person


if __name__ == '__main__':
    extract_info('dataset/validation/image_14.bmp')
    extract_info('dataset/validation/image_92.bmp')
    extract_info('dataset/validation/image_15.bmp')
    #extract_info(r'C:\Data\FTN E2\Cetvrta godina\Soft\sc-2021-e2-master\v4-doc_dig\example_01.png')
