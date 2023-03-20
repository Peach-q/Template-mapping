import numpy as np
import argparse
import imutils
import cv2

# создание анализатора аргументов и проведение анализа аргументов
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--puzzle", required = True, help = "C:/Users/Mi/PycharmProjects/pythonProject/puzzle.png")
ap.add_argument("-w", "--waldo", required = True, help = "C:/Users/Mi/PycharmProjects/pythonProject/waldo.png")

args = vars(ap.parse_args())

# загрузка изображения головоломки и Уолдо
puzzle = cv2.imread(args["puzzle"])
waldo = cv2.imread(args["waldo"])
(waldoHeight, waldoWidth) = waldo.shape[:2]

# поиск Уолдо в головоломке
result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

# беру ограничивающую рамку Уолдо и извлекаю его из изображения головоломки
topLeft = maxLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

# создание затемненного прозрачного "слоя", чтобы затемнить все в головоломке, кроме Уолдо
mask = np.zeros(puzzle.shape, dtype = "uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

# помещаю оригинального Уолдо обратно на изображение, чтобы он был "ярче", чем остальная часть изображения
puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

# отображение изображения
cv2.imshow("Puzzle", imutils.resize(puzzle, height = 650))
cv2.imshow("Waldo", waldo)
cv2.waitKey(0)