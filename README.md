[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![version](https://img.shields.io/badge/version-1.0.0-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Kaminyou/Halftone-Color-QR-Codes/blob/main/LICENSE)
![linting workflow](https://github.com/Kaminyou/Halftone-Color-QR-Codes/actions/workflows/main.yml/badge.svg)

# Halftone-Color-QR-Codes
DIP 2025 Final Project. This project is an extension of the [`Halftone QR codes`](https://dl.acm.org/doi/abs/10.1145/2508363.2508408?casa_token=qEK6M8ONCBgAAAAA:_kX5-qfsL0PlUF43oS5tABcnE2dwqLaF8w2_sQAzXx_M4__73qmvk9WaAv7BPZZoZkGEhzHpKlQWvg).

## Install
```sh
$ pip install -r requirements.txt
```

## Usage Examples
```sh
# gray example
$ python main.py -t 'https://www.csie.ntu.edu.tw/' -i 'input/gray_dog.png' -o 'output/stylized_qrcode.png' --meta -v 6 -b 3
# color example
$ python main.py -t 'https://www.csie.ntu.edu.tw/' -i 'input/color_hokkaido.png' -o 'output/stylized_qrcode.png' --meta -v 6 -b 5 --color
```

## Details
```
Command-line arguments for the Halftoning QRCode Generator.

Arguments:
    -t, --text (str): 
        The text or URL to encode into a QR code.
        Default: 'https://www.csie.ntu.edu.tw/'

    -i, --input (str): 
        Path to the input style image used to stylize the QR code.
        Default: 'input/sample1.png'

    -o, --output (str): 
        Path to save the stylized output QR code image.
        Default: 'output/stylized_qrcode.png'

    -v, --version (int): 
        QR code version (controls size and data capacity).
        Range: 1–40.
        Default: 6

    -b, --box-size (int): 
        Size (in pixels) of each QR code module (box).
        Default: 3

    -d, --drop-ratio (float): 
        Ratio (0.0–1.0) of randomly dropping a QR code module 
        for stylization or artistic effect.
        Default: 0.0

    --meta (flag): 
        If set, saves additional metadata associated with the QR code generation.
        Default: False (disabled)

    --color (flag): 
        If set, enables RGB mode (stylization with color image).
        Default: False (grayscale)
    
    --edge-enhance (flag): 
        If set, edges will be detected for module region keeping.
        Default: False (disabled)

    -e, --edge-ratio(float): 
        Ratio (0.0–1.0) of pixel of edge in the input image.
        Default: 0.1

    --wo-halftone (flag): 
        If set, our function will turn off the halftoning process.
        Default: False (Do halftoning)

    --pad-size (int): 
        Padding (in pixels) to apply around the final output QR code.
        Default: 5
```

## Acknowledgement
This project is an extension of the `Halftone QR codes`.
```
@article{chu2013halftone,
  title={Halftone QR codes},
  author={Chu, Hung-Kuo and Chang, Chia-Sheng and Lee, Ruen-Rone and Mitra, Niloy J},
  journal={ACM transactions on graphics (TOG)},
  volume={32},
  number={6},
  pages={1--8},
  year={2013},
  publisher={ACM New York, NY, USA}
}
```