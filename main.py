import argparse

import cv2

from utils import (error_diffusion, generate_clean_qrcode,
                   generate_qrcode_mask, is_consistant, replace_modules,
                   write_image)


def argument():
    parser = argparse.ArgumentParser('Halftoning QRCode')
    parser.add_argument(
        '-t',
        '--text',
        type=str,
        default='https://www.csie.ntu.edu.tw/',
        help='A text to generate a qrcode',
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='input/sample1.png',
        help='Input style image path',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/stylized_qrcode.png',
        help='Output styled qrcode path',
    )
    parser.add_argument(
        '--meta',
        action='store_true',
        help='Whether to save meta data or not',
    )
    args = parser.parse_args()
    return args


def main():
    args = argument()

    qrcode = generate_clean_qrcode(args.text, version=6, box_size=3)  # 123x123: (3x41)
    if args.meta:
        write_image('output/clean_qrcode.png', qrcode)

    if not is_consistant(qrcode, module_num=41, module_size=3):
        raise ValueError('QR code is not valid')

    qrcode_mask = generate_qrcode_mask(version=6)

    simplified_qrcode = replace_modules(qrcode, qrcode_mask, module_size=3)
    if args.meta:
        write_image('output/simplified_qrcode.png', simplified_qrcode)

    style_image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    style_image = cv2.resize(style_image, qrcode.shape)
    halftone_image = error_diffusion(style_image, method='j')
    styled_qrcode = replace_modules(qrcode, qrcode_mask, module_size=3, insert_image=halftone_image)
    write_image(args.output, styled_qrcode)


if __name__ == '__main__':
    main()
