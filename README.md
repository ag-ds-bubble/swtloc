# SWTloc : Stroke Width Transform Text Localizer
<img style="float: right;" src="swtloc/static/logo.png" height=139 width=418 align="right">

[![PyPI Latest Release](https://img.shields.io/pypi/v/swtloc)](https://pypi.org/project/swtloc/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/swtloc)](https://pypi.org/project/swtloc/)
[![Python Versions](https://img.shields.io/pypi/pyversions/swtloc)](https://pypi.org/project/swtloc/)

## Description

This repo contains a python implementation structured as a python package pertaining to the text localization method as in a natural image as outlayed in the Research Paper :- 

[Detecting Text in Natural Scenes with Stroke Width Transform.
Boris Epshtein, Eyal Ofek & Yonatan Wexler
(June, 2010)](https://www.microsoft.com/en-us/research/publication/detecting-text-in-natural-scenes-with-stroke-width-transform/)

<img style="float: right;" src="swtloc/static/logo.gif" align="centre">

This library extends the transformation stage of the image for textual content by giving the ability to :

- Localize `Letter`'s : through `SWTImage.localizeLetters`
- Localize `Words`'s, via fusing individual `Letter`'s : through `SWTImage.localizeWords`

The process flow of is depicted in the image below : 

<img style="float: right;" src="swtloc/static/SWTLoc_Process_Flow.png" align="centre">


### Installation 
```py
pip install swtloc
```

## Speed Benchmarking
Below is the speed comparison between different versions of ``SWTLoc`` and their various engines. The time measured for
each test image was calculated based on 10 iterations of 10 runs each.

Test Image | SWT v1.1.1 (Python) | SWT v1.1.1 (Python) [x] | SWT v2.0.0 (Python) | SWT v2.0.0 (Python) [x] | SWT v2.0.0 (numba) | SWT v2.0.0 (numba) [x]
--- | --- | --- | --- |--- |--- |--- 
test_img1.jpg | 15.614 seconds | 1.0x | 8.071 seconds | 1.935x | 0.308 seconds | 100.66x
test_img2.jpg | 9.644 seconds | 1.0x | 4.173 seconds | 2.311x | 0.176 seconds | 54.829x
test_img3.jpg | 4.386 seconds | 1.0x | 2.638 seconds | 1.663x | 0.083 seconds | 53.104x
test_img4.jpeg | 7.225 seconds | 1.0x | 3.887 seconds | 1.858x | 0.14 seconds | 51.42x
test_img5.jpg | 16.338 seconds | 1.0x | 7.592 seconds | 2.152x | 0.3 seconds | 54.405x
test_img6.jpg | 4.831 seconds | 1.0x | 2.873 seconds | 1.682x | 0.083 seconds | 57.853x


## Usage 
These code blocks can be found in [SWTloc Usage [v2.0.0 onwards].ipynb](examples/SWTloc-Usage-[v2.0.0-onwards].ipynb)
notebook in examples/.

### Initialisation of ``SWTLocalizer``
- Initialising the  - This is the entry point, which can accept either single image path (str)/
multiple image paths (List[str])/ single image (np.ndarray)/ multiple images (List[np.ndarray]).
```py
from swtloc import SWTLocalizer
imgpath = 'images/test_image_4/test_img4.jpeg'
respath = 'images/test_image_4/usage_results/'
swtl = SWTLocalizer(image_paths=imgpath)
swtImgObj = swtl.swtimages[0]
print(swtImgObj, type(swtImgObj))
swtImgObj.showImage()
```
> SWTImage-test_img4 <class 'swtloc.abstractions.SWTImage'>
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_01.jpg" align="centre">

### Stroke Width Transformation using ``SWTImage.transformImage``
```py
swt_mat = swtImgObj.transformImage(text_mode='lb_df',
                                   auto_canny_sigma=1.0,
                                   maximum_stroke_width=20)
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_01_02_03_04.jpg" align="centre">

### Finding Connected Components and Pruning them using ``SWTImage.findAndPruneConnectedComponents``
```py
image_1C, pruned_image_1C = swtImgObj.findAndPruneConnectedComponents(minimum_pixels_per_cc=100,
                                                                      maximum_pixels_per_cc=10_000,
                                                                      acceptable_aspect_ratio=0.2)
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_04_06_07_09.jpg" align="centre">

### Localizing Letters using ``SWTImage.localizeLetters``
```py
localized_letters = swtImgObj.localizeLetters()
letter_labels = list([int(k) for k in localized_letters.keys()])
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_11_12_13.jpg" align="centre">

### Query a Letter using ``SWTImage.getLetter``
```py
letter_label = letter_labels[3]
locletter = swtImgObj.getLetter(key=letter_label)
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_17_18.jpg" align="centre">

### Localizing Letters using ``SWTImage.localizeWords``
```py
localized_words = swtImgObj.localizeWords()
word_labels = list([int(k) for k in localized_words.keys()])
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_14_15_16.jpg" align="centre">

### Query a Word using ``SWTImage.getWord``
```py
word_label = word_labels[12]
locword = swtImgObj.getWord(key=word_label)
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_19_20.jpg" align="centre">

### Accessing intermediary stage images using ``SWTImage.showImage`` and saving them
```py
swtImgObj.showImage(image_codes=[IMAGE_ORIGINAL,
                                 IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS],
                    plot_title='Original & Bubble Mask')
```
> <img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_01_16.jpg" align="centre">


### Saving Crops of the localized letters and words
```py
# Letter Crops
swtImgObj.saveCrop(save_path=respath, crop_of='letters', crop_key=4, crop_type='min_bbox', crop_on=IMAGE_ORIGINAL)
swtImgObj.saveCrop(save_path=respath, crop_of='letters', crop_key=4, crop_type='min_bbox', crop_on=IMAGE_SWT_TRANSFORMED)

# Word Crops
swtImgObj.saveCrop(save_path=respath, crop_of='words', crop_key=13, crop_type='bubble', crop_on=IMAGE_ORIGINAL)
swtImgObj.saveCrop(save_path=respath, crop_of='words', crop_key=13, crop_type='bubble', crop_on=IMAGE_SWT_TRANSFORMED)
```

<img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_letters-4_min_bbox_IMAGE_ORIGINAL_CROP.jpg" align="left" width="200" height="100">
<img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_letters-4_min_bbox_IMAGE_SWT_TRANSFORMED_CROP.jpg" align="right" align="left" width="200" height="100">
<img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_words-13_bubble_IMAGE_ORIGINAL_CROP.jpg" align="left" align="left" width="200" height="100">
<img style="float: right;" src="examples/images/test_image_4/usage_results/test_img4_words-13_bubble_IMAGE_SWT_TRANSFORMED_CROP.jpg" align="right" align="left" width="200" height="100">

### Other examples
```py
Images from other transformations as coming out of the SWT Usage notebook.
```


