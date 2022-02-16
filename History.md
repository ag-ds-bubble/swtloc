## History Logs
****
<u>v2.0.0 : Major Release - Refactoring, New Engines, Abstraction Addition</u>
- Following refactoring Additions/Changes were made: 
  - Core algorithms moved to [core.py](swtloc/core.py)
  - Older files deprecated and names changed, these files would be removed in v2.0.1
    - `bubble_bbox.py` → `_bubble_bbox.py`
    - `swt.py` → `_swt.py`
    - `utils.py` → `_utils.py`
  - Add [README.md](README.md) (v2.0.0 onwards) and deprecated the old ones
  - Add [History.md](History.md) files to house history logs
  - Newer files added:
    - [core.py](swtloc/core.py) : To house all the core algorithms - `findStrokes`, `Fusion`.
    - [abstractions.py](swtloc/abstractions.py) : To house all the abstractions - `SWTImage`, `Letter` and `Word`.
- Dependency Changes
  - (+) Numba
  - (-) imutils
- Algorithmic Changes:
  - (+/-) Major deprecation in `SWTLocalizer`, almost all codebase moved to other locations. This will be, henceforth, a driver class.
  - (+) New [abstractions.py](swtloc/abstractions.py) file:
    - Addition of class `SWTImage` - An abstraction for an individual images sent in for processing. Has following major functions : 
      - transformImage : To transform the original image to its stroke widths (*1)
      - findAndPruneConnectedComponents : To find and prune the connected components
      - localizeLetters : To localize prospective letters
      - localizeWords : To group prospective letters into words
      - getLetter : To retrieve an individual letters
      - getWord : To retrieve an individual word
      - saveImage : To save an image through is bytecode
      - saveCrop : To crop and save a letter or a word
      - showImage: To display one/multiple images using the Image Codes defined in [configs.py](swtloc/configs.py)
    - Addition of class `Letter` - Represent and houses properties of possible letters
      - Functionality : Add various localization to input image
    - Addition of class `Words` - Represent and houses properties of possible words
      - Functionality : Add various localizations to input image
  - (+) New [core.py](swtloc/core.py) file
    - Addition of `swt_strokes` & `swt_strokes_jitted` function corresponding to the `python` and `numba` engines
    - Addition of `Fusion` & `ProxyLetter` for grouping of letters into probable letters
  - (+) New [base.py](swtloc/base.py) file
    - Addition of `IndividualComponentBase` : A base class to be inherited by `Letter`
    - Addition of `GroupedComponentsBase` : A base class to be inherited by `Word`
    - Addition of `TextTransformBase` : A base class to be inherited by `SWTImage`
  - (+) New [configs.py](swtloc/configs.py) file
    - Houses configurations for the Stroke Width Transform
  - (+) Add [Improvements in v2.0.0.ipynb](examples/Improvements-in-v2.0.0.ipynb) notebook
  - (+) Add [StackOverflow QnA [DEPRECATED] (Will be removed in v2.1.0).ipynb](examples/StackOverflow-QnA-[DEPRECATED]-(Will-be-removed-in-v2.1.0).ipynb) notebook
  - (+) Add [SWTloc Usage [v2.0.0 onwards].ipynb](examples/SWTloc-Usage-[v2.0.0-onwards].ipynb) notebook

<u>v1.1.1 : Refine Versioning System</u>
- New versioning system defined : x[Major Update].x[Minor Update].x[Fixes]
- Tag 1.1.x Represents all bug fixed versions of 1.1. 
- Bug Fixes

<u>v1.0.0.3 : Add Individual Image Processing</u>
- Functionality to transform pre-loaded image
- Minor Bug Fixes
- Add support for Python 3.6
- Reduce Dependency

<u>v1.0.0.2 : Few bug fixes and addendum's</u>
- Add parameter to govern the width of BubbleBBox 
- Add Examples - StackOverflow Q/A
- Add image resizing utility function to the utils.py

<u>v1.0.0.1 : Original Package</u>
- Add SWTlocaliser to the package
- Add the logic for Bubble Bounding Boxes
- Add Examples

