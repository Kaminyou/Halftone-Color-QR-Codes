import math
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
    is_rgb = image.ndim == 3 and image.shape[2] == 3

    if not is_rgb:
        image = image[..., np.newaxis]  # shape becomes (H, W, 1)

    h, w, ch = image.shape
    image_float = image.astype(np.float32)
    result_image = np.zeros_like(image, dtype=np.uint8)

    filter_pos = (0, 0)
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
        current_filter = weight if r % 2 == 0 else inv_weight
        col_positions = range(w) if r % 2 == 0 else range(w - 1, -1, -1)

        for c in col_positions:
            for k in range(ch):
                old_pixel = image_float[r, c, k]
                new_pixel = 255 if old_pixel > 127 else 0
                result_image[r, c, k] = new_pixel
                error = old_pixel - new_pixel
                for dr in range(current_filter.shape[0]):
                    for dc in range(current_filter.shape[1]):
                        nr, nc = r + dr - filter_pos[0], c + dc - filter_pos[1]
                        if 0 <= nr < h and 0 <= nc < w:
                            image_float[nr, nc, k] += error * current_filter[dr, dc]

    return result_image.squeeze()  # Remove channel if input was grayscale


def replace_modules(
    qrcode_image: npt.NDArray[np.uint8],
    qrcode_mask: npt.NDArray[np.bool],
    module_size: int = 3,
    insert_image: t.Optional[npt.NDArray[np.uint8]] = None,
    drop_ratio: float = 0.0,
    salient_mask: t.Optional[npt.NDArray[np.bool]] = None,  # H, W
) -> npt.NDArray[np.uint8]:

    if insert_image is None:
        insert_image = np.full_like(qrcode_image, 255)  # default empty (white) insert image

    h, w = qrcode_image.shape
    module_num = h // module_size
    styled_qrcode_image = np.full_like(qrcode_image, 255)

    # collect position for replacing
    normal_candidates = []
    salient_candidates = []
    block_num = 0
    for r in range(module_num):
        for c in range(module_num):
            if qrcode_mask[r, c]:  # mutable
                block_num += 1
                center_value = qrcode_image[r * module_size + module_size // 2, c * module_size + module_size // 2]
                center_value_on_insert_image = insert_image[r * module_size + module_size // 2, c * module_size + module_size // 2]  # noqa
                the_same = center_value == center_value_on_insert_image
                if not the_same:  # it should be changed
                    if salient_mask is not None and salient_mask[r * module_size + module_size // 2][c * module_size + module_size // 2]:  # noqa # it is in a salient region
                        salient_candidates.append((r, c))
                    else:
                        normal_candidates.append((r, c))

    # now salient_candidates is a subset of candidates
    # calculate how many module blocks can be fully preserved for styled image
    keep_num = int(block_num * drop_ratio)
    keep_positions = []
    # sample from salient_candidates first
    salient_candidates_keep_num = min(len(salient_candidates), keep_num)
    keep_positions += random.sample(salient_candidates, salient_candidates_keep_num)
    # sample from candidates
    normal_candidates_keep_num = min(len(normal_candidates), keep_num - salient_candidates_keep_num)
    keep_positions += random.sample(normal_candidates, normal_candidates_keep_num)

    keep_positions = set(keep_positions)

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
                # if (r, c) in keep_positions, we would not change it
                if (r, c) in keep_positions:
                    continue
                styled_qrcode_image[r * module_size + module_size // 2, c * module_size + module_size // 2] = center_value  # noqa

    print(f'Drop {len(keep_positions)} values (salient: {salient_candidates_keep_num}); Total {block_num} values')
    return styled_qrcode_image


def replace_modules_color(
    qrcode_image: npt.NDArray[np.uint8],  # H, W
    qrcode_mask: npt.NDArray[np.bool],  # H, W
    module_size: int = 3,
    insert_image: t.Optional[npt.NDArray[np.uint8]] = None,  # H, W, C
    drop_ratio: float = 0.0,
    salient_mask: t.Optional[npt.NDArray[np.bool]] = None,  # H, W
) -> npt.NDArray[np.uint8]:

    if insert_image is None:
        insert_image = np.full_like(qrcode_image, 255)  # default empty (white) insert image
        insert_image = np.tile(insert_image[:, :, np.newaxis], 3)

    h, w = qrcode_image.shape
    module_num = h // module_size
    styled_qrcode_image = np.full_like(insert_image, 255)

    # collect position for replacing
    normal_candidates = []
    salient_candidates = []
    block_num = 0
    for r in range(module_num):
        for c in range(module_num):
            if qrcode_mask[r, c]:  # mutable
                block_num += 1
                if salient_mask is not None and salient_mask[r * module_size + module_size // 2][c * module_size + module_size // 2]:  # noqa # it is in a salient region
                    salient_candidates.append((r, c))
                else:
                    normal_candidates.append((r, c))

    # now salient_candidates is a subset of candidates
    # calculate how many module blocks can be fully preserved for styled image
    keep_num = int(block_num * drop_ratio)
    keep_positions = []
    # sample from salient_candidates first
    salient_candidates_keep_num = min(len(salient_candidates), keep_num)
    keep_positions += random.sample(salient_candidates, salient_candidates_keep_num)
    # sample from candidates
    normal_candidates_keep_num = min(len(normal_candidates), keep_num - salient_candidates_keep_num)
    keep_positions += random.sample(normal_candidates, normal_candidates_keep_num)

    keep_positions = set(keep_positions)

    for r in range(module_num):
        for c in range(module_num):
            if qrcode_mask[r, c] == 0:
                # immutable: preserve the entire module
                styled_qrcode_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size, :] = \
                    qrcode_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size, np.newaxis]  # noqa
            else:
                # mutable: only preserve the middle pixel in the module
                # obtain the center value of source qrcode
                center_value = qrcode_image[r * module_size + module_size // 2, c * module_size + module_size // 2]

                # replace entire module with the insert image
                styled_qrcode_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size] = \
                    insert_image[r * module_size:(r + 1) * module_size, c * module_size:(c + 1) * module_size]

                # fill the center value of source qrcode if not the same
                # if (r, c) in keep_positions, we would not change it
                if (r, c) in keep_positions:
                    continue
                styled_qrcode_image[r * module_size + module_size // 2, c * module_size + module_size // 2, :] = center_value  # noqa

    print(f'Drop {len(keep_positions)} values (salient: {salient_candidates_keep_num}); Total {block_num} values')
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


def pad_image(image: npt.NDArray[np.uint8], pad_size: int = 5) -> npt.NDArray[np.uint8]:
    if pad_size == 0:
        return image

    pad_dim = ((pad_size, pad_size), (pad_size, pad_size))

    if image.ndim == 3:
        pad_dim = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))

    return np.pad(
        image,
        pad_width=pad_dim,
        mode='constant',
        constant_values=255
    )


def gradient_orientation_image(
    image: npt.NDArray[np.uint8],
    K: int = 2,
) -> npt.NDArray[np.uint8]:
    # reflect padding with p=1
    p = 1
    # use astype int32 to avoid overflow
    padded_image = np.pad(image, 1, 'reflect').astype(np.int32)
    h, w = image.shape

    grad_image = image.copy()
    orientation_image = np.zeros(image.shape, dtype=float)
    for row in range(h):
        for col in range(w):
            # row gradient 1/(k+2)*[(A2+kA3+A4)-(A0+kA7+A6)]
            right_col = padded_image[row - 1 + p, col + 1 + p] + K * padded_image[row + p, col + 1 + p] + padded_image[row + 1 + p, col + 1 + p]  # noqa
            left_col = padded_image[row - 1 + p, col - 1 + p] + K * padded_image[row + p, col - 1 + p] + padded_image[row + 1 + p, col - 1 + p]  # noqa
            G_R = 1.0 / float(K + 2) * float(right_col - left_col)

            # col gradient 1/(k+2)*[(A6+kA5+A4)-(A0+kA1+A2)]
            top_row = padded_image[row - 1 + p, col - 1 + p] + K * padded_image[row - 1 + p, col + p] + padded_image[row - 1 + p, col + 1 + p]  # noqa
            bottom_row = padded_image[row + 1 + p, col - 1 + p] + K * padded_image[row + 1 + p, col + p] + padded_image[row + 1 + p, col + 1 + p]  # noqa
            G_C = 1.0 / float(K + 2) * float(bottom_row - top_row)

            grad_image[row, col] = math.sqrt((G_R ** 2) + (G_C ** 2))
            orientation_image[row, col] = np.arctan2(G_C, G_R) * 180 / np.pi  # this handle G_R=0, convert to degree

    return grad_image, orientation_image


def thresholding(
    grad_image: npt.NDArray[np.uint8],
) -> npt.NDArray[np.bool]:
    h, w = grad_image.shape
    # find the threshold of top 5% large value
    # Initialize histogram array with zeros (256 bins for 0-255 values)
    histo = np.zeros(256, dtype=int)
    for row in range(h):
        for col in range(w):
            histo[grad_image[row, col]] += 1

    cumulative_pixel = 0
    total_pixel = h * w
    for i in range(256):
        cumulative_pixel += histo[i]
        if cumulative_pixel > total_pixel * 0.95:  # find pixel value larger than 95% of gradImg
            T = i
            # print(float(cumulative_pixel / total_pixel))
            break

    # use threshold to generate edge map
    # there's not common for png that only have 0/1 binary img so we set 0/255
    output = np.zeros(grad_image.shape, dtype=bool)
    for row in range(h):
        for col in range(w):
            if grad_image[row, col] > T:
                output[row, col] = 1  # one for edge place
            else:
                output[row, col] = 0

    return output


def edge_detector(
    image: npt.NDArray[np.uint8],
) -> npt.NDArray[np.bool]:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_image, _ = gradient_orientation_image(image, K=2)
    edge_image = thresholding(grad_image)
    return edge_image
