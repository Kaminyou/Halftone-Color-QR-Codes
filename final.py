import string

import cv2
import numpy as np
import qrcode

'''QRcode = cv2.imread('input/qr1000.png',cv2.IMREAD_GRAYSCALE)

print(np.sum(QRcode == 0), np.sum(QRcode == 255))

Rows, Cols = QRcode.shape

for r in range(Rows):
    for c in range(Cols):
        if QRcode[r, c]>127: QRcode[r, c]=255
        else: QRcode[r, c] = 0

print(np.sum(QRcode == 0), np.sum(QRcode == 255))

for r in range(25):
    for c in range(25):
        inconsistant = False
        color = QRcode[r*40, c*40]
        for mr in range(40):
            for mc in range(40):
                if QRcode[r*40+mr, c*40+mc] != color:
                    inconsistant = True
                    #print(r,c, mr,mc)
                    #pass
print(inconsistant)

result = QRcode
cv2.imwrite("result.png",result)'''

# 讀進來然後重新整理
img = cv2.imread("input/qrcode-generator.png", cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
target_module_count = 41  # 或你預估的數量
fixed_img = cv2.resize(
    bin_img, (target_module_count, target_module_count), interpolation=cv2.INTER_NEAREST,
)
clean_img = cv2.resize(
    fixed_img, (target_module_count*6, target_module_count*6), interpolation=cv2.INTER_NEAREST,
)

for r in range(41):
    for c in range(41):
        inconsistant = False
        color = clean_img[r*6, c*6]
        for mr in range(6):
            for mc in range(6):
                if clean_img[r*6+mr, c*6+mc] != color:
                    inconsistant = True
                    # print(r,c, mr,mc)
                    # pass
print(inconsistant)

result = clean_img
cv2.imwrite("result_.png", result)


# 用code生QR code
def generate_clean_qrcode(text, version=6, box_size=10, border=0):
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,  # 每 module 幾個 pixel（整數）
        border=border       # 不要加白邊
    )
    qr.add_data(text)
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.convert("L")  # 灰階
    return img


img = generate_clean_qrcode("https://www.csie.ntu.edu.tw/", version=6, box_size=3)
img.save("clean_qrcode.png")
clean_qr = cv2.imread('clean_qrcode.png', cv2.IMREAD_GRAYSCALE)
for r in range(41):
    for c in range(41):
        inconsistant = False
        color = clean_qr[r*3, c*3]
        for mr in range(3):
            for mc in range(3):
                if clean_qr[r*3+mr, c*3+mc] != color:
                    inconsistant = True
                    # print(r,c, mr,mc)
                    # pass
print(inconsistant)
print(np.sum(clean_qr == 0), np.sum(clean_qr == 255))


def get_alignment_pattern_locations(version):
    if version == 1:
        return []
    pos_count = version // 7 + 2
    step = 0
    if version != 32:
        step = (21 + 4 * (version - 1) - 13) // (pos_count - 1)
    positions = [6]
    for i in range(1, pos_count - 1):
        positions.append(6 + i * step)
    positions.append(21 + 4 * (version - 1) - 7)
    return positions


def build_qr_mask(version=6):
    size = 21 + 4 * (version - 1)
    mask = np.ones((size, size), dtype=np.uint8)  # 1: 可改，0: 不能改

    # 定位圖形（左上、右上、左下）
    pos = [(0, 0), (0, size - 7), (size - 7, 0)]
    for (r, c) in pos:
        mask[r:r+7, c:c+7] = 0

    # Timing pattern（橫線和直線）
    mask[6, :] = 0
    mask[:, 6] = 0

    # 格式資訊區塊
    mask[8, 0:9] = 0
    mask[0:9, 8] = 0
    mask[size - 8:, 8] = 0
    mask[8, size - 8:] = 0

    # Version info（右上、左下各一塊 6×3 和 3×6）
    if version >= 7:
        mask[0:6, size-11:size-8] = 0
        mask[size-11:size-8, 0:6] = 0

    # Alignment pattern
    align_pos = get_alignment_pattern_locations(version)
    for r in align_pos:
        for c in align_pos:
            if (r < 7 and c < 7) or (r < 7 and c > size - 8) or (r > size - 8 and c < 7):
                continue  # 跳過與定位圖重疊的
            mask[r-2:r+3, c-2:c+3] = 0

    return mask


def replace_modules(img, mask, module_size=3, insert_img=np.full_like(img, 255)):  # 預設塞全白圖
    h, w = img.shape
    size = h // module_size
    new_img = np.full_like(img, 255)

    for r in range(size):
        for c in range(size):
            if mask[r, c] == 0:
                # 不改動，保留原 module
                new_img[r*module_size:(r+1)*module_size, c*module_size:(c+1)*module_size] = \
                    img[r*module_size:(r+1)*module_size, c*module_size:(c+1)*module_size]
            else:
                # 替換，只保留中心像素
                center_val = img[r*module_size + module_size//2, c*module_size + module_size//2]
                new_img[r*module_size:(r+1)*module_size, c*module_size:(c+1)*module_size] = \
                    insert_img[r*module_size:(r+1)*module_size, c*module_size:(c+1)*module_size]
                new_img[r*module_size + module_size // 2, c*module_size + module_size // 2] = center_val  # 填中間  # noqa

    return new_img


mask = build_qr_mask(version=6)
new_qr = replace_modules(clean_qr, mask, module_size=3)
cv2.imwrite("stylized_qr_with_white_img.png", new_qr)


def errDiffusion(inImg: np.ndarray, filterName: string):
    Rows, Cols = inImg.shape
    inImg_float = inImg.astype(np.float32)
    outImg = np.zeros(inImg.shape, dtype=np.uint8)

    weight = np.empty([2, 3])
    inv_weight = np.empty([2, 3])
    filter_pos = (0, 0)  # (row pos, col pos)
    if filterName == "fs":
        filter_pos = (0, 1)
        weight = np.array([[0, 0, 7], [3, 5, 1]], dtype=np.float64) / 16
        inv_weight = np.array([[7, 0, 0], [1, 5, 3]], dtype=np.float64) / 16
    elif filterName == "j":
        filter_pos = (0, 2)
        weight = np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]], dtype=np.float64) / 48
        inv_weight = np.array([[5, 7, 0, 0, 0], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]], dtype=np.float64) / 48
    elif filterName == "Atkinson":
        filter_pos = (0, 1)
        weight = np.array([[0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]]) / 8
        inv_weight = np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]]) / 8

    for r in range(Rows):
        curFilter = 0
        colRange = 0
        if r % 2 == 0:  # even row, ->
            curFilter = weight
            colRange = range(Cols)
        else:  # odd row, <-
            curFilter = inv_weight
            colRange = range(Cols - 1, -1, -1)
        for c in colRange:
            old_pixel = inImg_float[r, c]
            new_pixel = 255 if old_pixel > 127 else 0
            outImg[r, c] = new_pixel
            error = old_pixel - new_pixel
            for dr in range(curFilter.shape[0]):
                for dc in range(curFilter.shape[1]):
                    nr, nc = r + dr - filter_pos[0], c + dc - filter_pos[1]
                    if 0 <= nr < Rows and 0 <= nc < Cols:
                        inImg_float[nr, nc] += error * curFilter[dr, dc]

    return outImg


sample1 = cv2.imread('input/sample1.png', cv2.IMREAD_GRAYSCALE)
halftoneImg = errDiffusion(sample1, "j")
new_qr = replace_modules(clean_qr, mask, module_size=3, insert_img=halftoneImg)
cv2.imwrite("stylized_qr_with_hw4_sample1.png", new_qr)
