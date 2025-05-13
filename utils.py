import random
import typing as t

import cv2
import numpy as np
import numpy.typing as npt
import qrcode


def generate_clean_qrcode(
    text: str,
    version: int = 6,
    box_size: int = 10,
    border: int = 0,
) -> npt.NDArray[np.uint8]:

    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,  # number of pixel per module
        border=border,  # whether to add white edges
    )
    qr.add_data(text)
    qr.make(fit=False)
    img = qr.make_image(fill_color='black', back_color='white')
    img = img.convert('L')  # grayscale
    return np.array(img)


def get_alignment_pattern_locations(
    version: int,
) -> t.List[int]:
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


def generate_qrcode_mask(
    version: int = 6,
) -> npt.NDArray[bool]:

    size = 21 + 4 * (version - 1)
    mask = np.ones((size, size), dtype=bool)  # 1: mutable; 0: immutable

    # anchors (top-left, top-right, bottom-left)
    pos = [(0, 0), (0, size - 7), (size - 7, 0)]
    for (r, c) in pos:
        mask[r:r+7, c:c+7] = 0

    # timing pattern (vertical and horizontal lines)
    mask[6, :] = 0
    mask[:, 6] = 0

    # format information
    mask[8, 0:9] = 0
    mask[0:9, 8] = 0
    mask[size - 8:, 8] = 0
    mask[8, size - 8:] = 0

    # version information（top-right 6×3 and bottom-left 3×6）
    if version >= 7:
        mask[0:6, size - 11:size - 8] = 0
        mask[size - 11:size - 8, 0:6] = 0

    # alignment pattern
    align_pos = get_alignment_pattern_locations(version)
    for r in align_pos:
        for c in align_pos:
            if (r < 7 and c < 7) or (r < 7 and c > size - 8) or (r > size - 8 and c < 7):
                continue  # skip those overlapping with anchors
            mask[r - 2:r + 3, c - 2:c + 3] = 0

    return mask


def error_diffusion(
    image: npt.NDArray[np.uint8],
    method: str = 'j',
) -> npt.NDArray[np.uint8]:

    h, w = image.shape
    image_float = image.astype(np.float32)
    result_image = np.zeros_like(image)

    filter_pos = (0, 0)  # (row, col) anchor
    weight = None
    inv_weight = None
    if method == 'fs' or method == 'Floyd-Steinberg':
        filter_pos = (0, 1)
        weight = np.array([[0, 0, 7], [3, 5, 1]], dtype=np.float64) / 16
        inv_weight = np.array([[7, 0, 0], [1, 5, 3]], dtype=np.float64) / 16
    elif method == 'j' or method == 'Jarvis-Judice-Ninke':
        filter_pos = (0, 2)
        weight = np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]], dtype=np.float64) / 48
        inv_weight = np.array([[5, 7, 0, 0, 0], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]], dtype=np.float64) / 48
    elif method == 'Atkinson':
        filter_pos = (0, 1)
        weight = np.array([[0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]]) / 8
        inv_weight = np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]]) / 8
    else:
        raise NotImplementedError

    for r in range(h):
        current_filter = None
        col_positions = None

        if r % 2 == 0:  # even row: ->
            current_filter = weight
            col_positions = range(w)
        else:  # odd row: <-
            current_filter = inv_weight
            col_positions = range(w - 1, -1, -1)

        for c in col_positions:
            old_pixel = image_float[r, c]
            new_pixel = 255 if old_pixel > 127 else 0
            result_image[r, c] = new_pixel
            error = old_pixel - new_pixel
            for dr in range(current_filter.shape[0]):
                for dc in range(current_filter.shape[1]):
                    nr, nc = r + dr - filter_pos[0], c + dc - filter_pos[1]
                    if 0 <= nr < h and 0 <= nc < w:
                        image_float[nr, nc] += error * current_filter[dr, dc]

    return result_image


def replace_modules(
    qrcode_image: npt.NDArray[np.uint8],
    qrcode_mask: npt.NDArray[np.uint8],
    module_size: int = 3,
    insert_image: t.Optional[npt.NDArray[np.uint8]] = None,
    drop_prob: float = 0.0,
) -> npt.NDArray[np.uint8]:

    if insert_image is None:
        insert_image = np.full_like(qrcode_image, 255)  # default empty (white) insert image

    h, w = qrcode_image.shape
    module_num = h // module_size
    styled_qrcode_image = np.full_like(qrcode_image, 255)
    drop_cnt = 0

    for r in range(module_num):
        for c in range(module_num):
            if qrcode_mask[r, c] == 0:
                # immutable: preserve the entire module
                styled_qrcode_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size] = \
                    qrcode_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size]
            else:
                # mutable: only preserve the middle pixel in the module
                # obtain the center value of source qrcode
                center_value = qrcode_image[r * module_size + module_size // 2, c * module_size + module_size // 2]
                center_value_on_insert_image = insert_image[r * module_size + module_size // 2, c * module_size + module_size // 2]  # noqa
                the_same = center_value == center_value_on_insert_image

                # replace entire module with the insert image
                styled_qrcode_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size] = \
                    insert_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size]

                # fill the center value of source qrcode if not the same
                # with drop_prob, some of them will not be replaced
                if not the_same:
                    rnd = random.random()
                    if rnd < drop_prob:
                        drop_cnt += 1
                        continue
                    styled_qrcode_image[r * module_size + module_size // 2, c * module_size + module_size // 2] = center_value  # noqa
    print(f'Drop {drop_cnt} values')
    return styled_qrcode_image


def write_image(
    image_path: str,
    image: npt.NDArray[np.uint8],
) -> None:
    if len(image.shape) == 3:
        image = image[:, :, ::-1]  # RGB -> BGR
    cv2.imwrite(image_path, image)


def is_consistant(
    qrcode_image: npt.NDArray[np.uint8],
    module_num: int,
    module_size: int,
) -> bool:
    consistant = True
    for r in range(module_num):
        for c in range(module_num):
            color = qrcode_image[r * module_size, c * module_size]
            for mr in range(module_size):
                for mc in range(module_size):
                    if qrcode_image[r * module_size + mr, c * module_size + mc] != color:
                        consistant = False
    return consistant
