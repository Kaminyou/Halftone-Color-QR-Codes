# NTU-DIP-Final-2025-Spring
DIP Final Project

## Usage
```sh
# gray example
$ python main.py -t 'https://www.csie.ntu.edu.tw/' -i 'input/sample1.png' -o 'output/stylized_qrcode.png' --meta -v 6 -b 3
# color example
$ python main.py -t 'https://www.csie.ntu.edu.tw/' -i 'input/sample1.png' -o 'output/stylized_qrcode.png' --meta -v 6 -b 5 --color
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

    -d, --drop-prob (float): 
        Probability (0.0–1.0) of randomly dropping a QR code module 
        for stylization or artistic effect.
        Default: 0.0

    --meta (flag): 
        If set, saves additional metadata associated with the QR code generation.
        Default: False (disabled)

    --color (flag): 
        If set, enables RGB mode (stylization with color image).
        Default: False (grayscale)

    --pad-size (int): 
        Padding (in pixels) to apply around the final output QR code.
        Default: 5
```