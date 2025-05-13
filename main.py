import argparse

import cv2

from utils import (error_diffusion, generate_clean_qrcode,
                   generate_qrcode_mask, is_consistant, replace_modules,
                   write_image, replace_modules_color, pad_image)


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
        '-v',
        '--version',
        type=int,
        default=6,
        help='QRCode version',
    )
    parser.add_argument(
        '-b',
        '--box-size',
        type=int,
        default=3,
        help='QRCode module box size',
    )
    parser.add_argument(
        '-d',
        '--drop-prob',
        type=float,
        default=0.0,
        help='Dropping value prob.',
    )
    parser.add_argument(
        '--meta',
        action='store_true',
        help='Whether to save meta data or not',
    )
    parser.add_argument(
        '--color',
        action='store_true',
        help='Whether use RGB image',
    )
    parser.add_argument(
        '--pad-size',
        type=int,
        default=5,
        help='Pad the output QRCode.',
    )
    args = parser.parse_args()
    return args


def main():
    args = argument()
    version = args.version
    box_size = args.box_size

    qrcode_image = generate_clean_qrcode(args.text, version=version, box_size=box_size)
    print(f'Generate QRCode with version: {version} and box size: {box_size}')
    print(f'Output QRcode has a shape of {qrcode_image.shape[0]}x{qrcode_image.shape[1]}')

    if args.meta:
        write_image('output/clean_qrcode.png', qrcode_image)

    module_num = qrcode_image.shape[0] // box_size
    if not is_consistant(qrcode_image, module_num=module_num, module_size=box_size):
        raise ValueError('QR code is not valid')

    qrcode_mask = generate_qrcode_mask(version=version)

    simplified_qrcode = replace_modules(qrcode_image, qrcode_mask, module_size=box_size)
    if args.meta:
        write_image('output/simplified_qrcode.png', simplified_qrcode)

    replace_modules_func = None
    if args.color:
        replace_modules_func = replace_modules_color
        style_image = cv2.imread(args.input)[:, :, ::-1]
    else:
        replace_modules_func = replace_modules
        style_image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    style_image = cv2.resize(style_image, qrcode_image.shape)
    halftone_image = error_diffusion(style_image, method='j')
    styled_qrcode = replace_modules_func(
        qrcode_image,
        qrcode_mask,
        module_size=box_size,
        insert_image=halftone_image,
        drop_prob=args.drop_prob,
    )
    styled_qrcode = pad_image(styled_qrcode, pad_size=args.pad_size)
    write_image(args.output, styled_qrcode)


if __name__ == '__main__':
    main()
