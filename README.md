# Digital Image Processing Group 5 Final Project: Image dehazing

## Usage
python scripts/dehaze.py -i [input_path] -o [output_path] -t [atmospheric_value's threshold]

## Transmission map
在optimize_transmission中加了maxfilter和blurfilter之後結果很爛 \
optimize完的T，如果不normalize到比較小的值，結果會直接爆掉變成白白一片 \
normalize的範圍會有影響，目前看來0~1應該是比較可以接受的 

## get_atmospheric_value
論文中是先取前0.1%再做filter，但我試過這樣做的話，在一些input上可能會把前0.1%的東東全部給filter掉 \
所以method 2是先filter再取0.1% \
效果其實差不多，但method 2比較不會有error

## Threshold
atmoshperic value的threshold取值跟論文內不太一樣，論文內取6 \
對部分圖片來說6沒問題，但human3.png的結果會很黑，實驗結果取64比較好 \
目前感覺threshold越低結果就會越黑，越高就越白（朦朧） \
表現比較正常的是static2, static3, human3和architecture2 \
對於有不同深度的圖片效果不太好，會把比較靠近的部分弄得很黑，比較遠的就還是朦朧一片，像是static2, architecture1, architecture4, plant2, plant3, landscape1, landscape2 \
landscape2表現很爛，threshold <= 151，前面很黑後面正常， >=152，前面OK後面很白 \
不知道是這篇本來就不太能處理深度的問題？還是我爆掉了 

## Others
那個simplest_color_balance是用來調整最終的顏色的，就只是為了讓圖片看起來更漂亮，跟這篇論文沒關係
