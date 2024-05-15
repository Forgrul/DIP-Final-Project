# Digital Image Processing Group 5 Final Project: Image dehazing

## Usage
python scripts/dehaze.py -i [input_path] -o [output_path] -t [atmospheric_value's threshold]

## Transmission map
把maxFilter和blurFilter uncomment掉了，並且把Lmsrcr = delta * Lmsrcr + kappa給comment掉了。 \
論文騙我。\
但human3.png這樣反而效果很爛，所以對這張圖需要把delta和kappa加回去，並且需要再把maxFilter和blurFilter comment掉。

## get_atmospheric_value
論文中是先取前0.1%再做filter，但我試過這樣做的話，在一些input上可能會把前0.1%的東東全部給filter掉 \
所以method 2是先filter再取0.1% \
效果其實差不多，但method 2比較不會有error

## Others
那個simplest_color_balance是用來調整最終的顏色的，就只是為了讓圖片看起來更漂亮，跟這篇論文沒關係
