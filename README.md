# Softbank_proj
An application for softbank cooperation project. On this project, we aim to use pre-registered parameterized 3D tetrapods and multi-viewing video sequences to estimate the 3D pose and shape of the tetrapods and rebuild the animation of the tetrapods.

## Set up the environment
Please make sure you have virtualenv installed. If not, you can install it by `pip install virtualenv`. We use poetry to manage the dependencies.


```bash
python -m virtualenv -p 3.10 venv
cd venv
source bin/activate
```
For windows platform, run:
```bash
python -m virtualenv -p 3.10 venv
venv\Scripts\activate.ps1
```
Then install the project through poetry:
```bash
poetry install
pip install chumpy==0.70 --no-build-isolation
```
It may take about 2 minutes.

Then, go to the path that Chumpy is installed, and replace the `__init__modified_chumpy.py` with `__init__.py` and rename it to `__init__.py`. This should resolve the issue of Chumpy importing some deprecated numpy components.

## Next Steps

1. **Download the SMAL model**: You'll need to download the SMAL model file (`smal_CVPR2017.pkl` and `smal_CVPR2017_data.pkl`) from the [official repository](https://smal.is.tue.mpg.de/download.php).
2. Put the model file in the `data` folder, so it should be like this:
```
data/
├── smal_CVPR2017.pkl
└── smal_CVPR2017_data.pkl
```
1. Download the cow model from [here](https://smal.is.tue.mpg.de/download.php), Cows (Bovidae family)
2. Put the cow model in the `data/cows` folder, so it should be like this:
```
data/
├── smal_CVPR2017.pkl
├── smal_CVPR2017_data.pkl
└── cows/
```
## Usage

You can run the viewer:

```bash
poetry run smal-viewer --model data/smal_CVPR2017.pkl --cow data/cows/cow_alph4.pkl --use-smpl 
```

### Controls:

- Left-click and drag: Rotate the model
- Right-click and drag: Zoom in/out


## License

This project is licensed under the MIT License - see the LICENSE file for details.

3. **Extend the application**:
   - Add more controls for adjusting shape parameters
   - Implement loading of real animation data
   - Add texture mapping support
   - Implement exporting to common 3D formats

This implementation provides a basic framework for working with SMAL models in Python 3.10. The core functionality includes loading the model, applying shape and pose parameters, and visualizing the model with a simple animation. You can extend this framework to suit your specific needs.

## Credits
This project is based on the [SMAL](https://smal.is.tue.mpg.de/index.html) repository.

Please cite their paper if you use this code:
 ``` 
 @inproceedings{Zuffi:CVPR:2017,
        title = {{3D} Menagerie: Modeling the {3D} Shape and Pose of Animals},
        author = {Zuffi, Silvia and Kanazawa, Angjoo and Jacobs, David and Black, Michael J.},
        booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        month = jul,
        year = {2017},
        month_numeric = {7}
      }
```

Developer: Zixuan Vicky Lu(birdpeople1984@gmail.com)
