from numpy.linalg import det
# визначник матриці

from math import fabs, sqrt
from random import randrange
from _pydecimal import Decimal
from scipy.stats import f, t

# Значення за варіантом
m = 3
N = 4
p = 0.95

min_x1 = 20
max_x1 = 70
min_x2 = -15
max_x2 = 45
min_x3 = 20
max_x3 = 35

min_y = 200 + int((min_x1 + min_x2 + min_x3) / 3)
max_y = 200 + int((max_x1 + max_x2 + max_x3) / 3)


def matrixGenerator():
    # Генерує матрицю
    matrix_y = [[randrange(min_y, max_y)
                 for y in range(m)] for x in range(4)]
    return matrix_y


class CritValues:
    # Критичні значення
    @staticmethod
    def cohrenValue(selectionSize, selectionQty, significance):
        # Значення критерію Кохрена
        selectionSize += 1
        partResult1 = significance / (selectionSize - 1)
        params = [partResult1, selectionQty, (selectionSize - 1 - 1) * selectionQty]
        fisher = f.isf(*params)
        result = fisher / (fisher + (selectionSize - 1 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def studentValue(f3, significance):
        # Значення критерію Стьюдента
        return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def fisherValue(f3, f4, significance):
        # Значення критерію Фішера
        return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()


def middle_y(arr):
    # Середнє значення у
    middle = []
    for k in range(len(arr)):
        middle.append(sum(arr[k]) / len(arr[k]))
    return middle


def middle_x(arr):
    # Середнє значення х
    middle = [0, 0, 0]
    for k in range(4):
        middle[0] += arr[k][0] / 4
        middle[1] += arr[k][1] / 4
        middle[2] += arr[k][2] / 4
    return middle


print("Рівняння регресії: у = b0 + b1*X1 + b2*X2 + b3*X3")
print("Матриця планування експеременту:")
matrixExp = [[1, -1, -1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]]

for i in range(len(matrixExp)):
    for j in range(len(matrixExp[i])):
        print("{:4d}".format(matrixExp[i][j]), end="")
    print()

tf = True
# tf = true/false

while tf:
    matrix_y = matrixGenerator()
    matrix_x = [[min_x1, min_x2, min_x3], [min_x1, max_x2, max_x3], [max_x1, min_x2, max_x3], [max_x1, max_x2, min_x3]]
    a1, a2, a3, a11, a22, a33, a12, a13, a23 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    matrix = []

    mean_y = middle_y(matrix_y)
    mean_x = middle_x(matrix_x)

    for i in range(4):
        a1 += matrix_x[i][0] * mean_y[i] / 4
        a2 += matrix_x[i][1] * mean_y[i] / 4
        a3 += matrix_x[i][2] * mean_y[i] / 4
        a11 += matrix_x[i][0] ** 2 / 4
        a22 += matrix_x[i][1] ** 2 / 4
        a33 += matrix_x[i][2] ** 2 / 4
        a12 += matrix_x[i][0] * matrix_x[i][1] / 4
        a13 += matrix_x[i][0] * matrix_x[i][2] / 4
        a23 += matrix_x[i][1] * matrix_x[i][2] / 4

    a21 = a12
    a31 = a13
    a32 = a23

    my = sum(mean_y) / len(mean_y)
    numb0 = [[my, mean_x[0], mean_x[1], mean_x[2]], [a1, a11, a12, a13], [a2, a21, a22, a23], [a3, a31, a32, a33]]
    numb1 = [[1, my, mean_x[1], mean_x[2]], [mean_x[0], a1, a12, a13], [mean_x[1], a2, a22, a23],
             [mean_x[2], a3, a32, a33]]
    numb2 = [[1, mean_x[0], my, mean_x[2]], [mean_x[0], a11, a1, a13], [mean_x[1], a21, a2, a23],
             [mean_x[2], a31, a3, a33]]
    numb3 = [[1, mean_x[0], mean_x[1], my], [mean_x[0], a11, a12, a1], [mean_x[1], a21, a22, a2],
             [mean_x[2], a31, a32, a3]]
    dividerB = [[1, mean_x[0], mean_x[1], mean_x[2]], [mean_x[0], a11, a12, a13],
                [mean_x[1], a21, a22, a23], [mean_x[2], a31, a32, a33]]

    b0 = det(numb0) / det(dividerB)
    b1 = det(numb1) / det(dividerB)
    b2 = det(numb2) / det(dividerB)
    b3 = det(numb3) / det(dividerB)

    f1 = m - 1
    f2 = N
    q = 1 - p
    dispersion_y = [0, 0, 0, 0]

    for i in range(m):
        dispersion_y[0] += ((matrix_y[0][i] - mean_y[0]) ** 2) / 3
        dispersion_y[1] += ((matrix_y[1][i] - mean_y[1]) ** 2) / 3
        dispersion_y[2] += ((matrix_y[2][i] - mean_y[2]) ** 2) / 3
        dispersion_y[3] += ((matrix_y[3][i] - mean_y[3]) ** 2) / 3

    Gp = max(dispersion_y) / sum(dispersion_y)
    print("\nКритерій Кохрена")
    Gt = CritValues.cohrenValue(f2, f1, q)

    if Gt > Gp:
        print("Дисперсія однорідна при рівні значимості {:.2f}!\nЗбільшувати m не потрібно.".format(q))
        tf = False
    else:
        print("Дисперсія не однорідна при рівні значимості {:.2f}!".format(q))
        m += 1
    if m > 23:
        exit()

for i in range(4):
    matrix.append(matrix_x[i] + matrix_y[i])

print("Матриця з натуральних значень факторів")
print("  X1 X2 X3 Y1  Y2  Y3  ")
for i in range(len(matrix)):
    print("", end=" ")
    for j in range(len(matrix[i])):
        print(matrix[i][j], end=" ")
    print("")

print("Рівняння регресії")
print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 = ŷ".format(b0, b1, b2, b3))
print("Перевірка")
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, min_x1, b2, min_x2, b3, min_x3)
      + str(b0 + b1 * min_x1 + b2 * min_x2 + b3 * min_x3))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, min_x1, b2, max_x2, b3, max_x3)
      + str(b0 + b1 * min_x1 + b2 * max_x2 + b3 * max_x3))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, max_x1, b2, min_x2, b3, max_x3)
      + str(b0 + b1 * max_x1 + b2 * min_x2 + b3 * max_x3))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = ".format(b0, b1, max_x1, b2, max_x2, b3, min_x3)
      + str(b0 + b1 * max_x1 + b2 * max_x2 + b3 * min_x3))

print("\nКритерій Стьюдента")
f3 = f1 * f2
S_2b = sum(dispersion_y) / (N * N * m)
S_b = sqrt(S_2b)
beta_0 = (mean_y[0] + mean_y[1] + mean_y[2] + mean_y[3]) / N
beta_1 = (-mean_y[0] - mean_y[1] + mean_y[2] + mean_y[3]) / N
beta_2 = (-mean_y[0] + mean_y[1] - mean_y[2] + mean_y[3]) / N
beta_3 = (-mean_y[0] + mean_y[1] + mean_y[2] - mean_y[3]) / N
t_0 = fabs(beta_0) / S_b
t_1 = fabs(beta_1) / S_b
t_2 = fabs(beta_2) / S_b
t_3 = fabs(beta_3) / S_b

Tt = CritValues.studentValue(f1 * f2, q)
arr_t = [t_0, t_1, t_2, t_3]
arr_b = [b0, b1, b2, b3]

for i in range(4):
    if arr_t[i] > Tt:
        continue
    else:
        arr_t[i] = 0

for j in range(4):
    if arr_t[j] != 0:
        continue
    else:
        arr_b[j] = 0

print("Перевірка значемих коефіціентів:")
yj1 = arr_b[0] + arr_b[1] * min_x1 + arr_b[2] * min_x2 + arr_b[3] * min_x3
yj2 = arr_b[0] + arr_b[1] * min_x1 + arr_b[2] * max_x2 + arr_b[3] * max_x3
yj3 = arr_b[0] + arr_b[1] * max_x1 + arr_b[2] * min_x2 + arr_b[3] * max_x3
yj4 = arr_b[0] + arr_b[1] * max_x1 + arr_b[2] * max_x2 + arr_b[3] * min_x3

print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(arr_b[0], arr_b[1], min_x1, arr_b[2], min_x2, arr_b[3], min_x3) + str(yj1))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(arr_b[0], arr_b[1], min_x1, arr_b[2], max_x2, arr_b[3], max_x3) + str(yj2))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(arr_b[0], arr_b[1], max_x1, arr_b[2], min_x2, arr_b[3], max_x3) + str(yj3))
print("{:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} + {:.3f} * {:.3f} = "
      "".format(arr_b[0], arr_b[1], max_x1, arr_b[2], max_x2, arr_b[3], min_x3) + str(yj4))

print("\nКритерій Фішера:")
for i in range(3):
    if arr_b[i] == 0:
        del arr_b[i]

d = len(arr_b)
f4 = N - d
S_2ad = m * ((yj1 - mean_y[0]) ** 2 + (yj2 - mean_y[1]) ** 2 + (yj3 - mean_y[2]) ** 2 + (
        yj4 - mean_y[3]) ** 2) / f4
Fp = S_2ad / S_2b
Ft = CritValues.fisherValue(f1 * f2, f4, q)

if Fp > Ft:
    print("Рівняння регресії неадекватно оригіналу при рівні значимості {:.2f}".format(q))
else:
    print("Рівняння регресії адекватно оригіналу при рівні значимості {:.2f}".format(q))
