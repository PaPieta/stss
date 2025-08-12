# Structure Tensor Scale Space for Python
2D and 3D [structure tensor](https://en.wikipedia.org/wiki/Structure_tensor) [scale space](https://en.wikipedia.org/wiki/Scale_space) implementation for Python.

Forked from and based on Niels Jeppesen's [structure tensor repository](https://github.com/Skielex/structure-tensor/tree/master). Contains its basic functionality, with extra support of structure tensor scale space and expansion to a ring filter instead of the integrating filter.

For theoretical details see: [Paper](https://ieeexplore.ieee.org/document/10833649), [ArXiv](https://arxiv.org/abs/2409.13389)

In-depth examples, as well as reproduced figures from the publication can be found and interactively tested in the associated [Code Ocean capsule](https://codeocean.com/capsule/8105965/tree/v2).

## Installation

``` python
pip install stss
```
# Prerequisites
Library requires ```numpy``` and ```scipy```, they can be installed via provided requirements file:
```sh
  pip install -r requirements.txt
```

## Tiny Examples
The only parameter necessary for this version of structure tensor calculation is $\sigma$ (```sigma```), which is a scalar.

It is possible to disable the ring filter and use the original structure tensor definition, then another scalar parameter $\rho$ (```rho```) is necessary.

### Single scale for 2D and 3D 
The ```st2ss``` package supports running either 2D or 3D structure tensor analysis. The appropriate algorithm is chosen based on the dimensionality of the provided array. Eigenvalues (```val```) are sorted ascending.

``` python
import numpy as np
from stss import st

sigma = 1.5

# Load 2D or 3D data (np.array).
image = np.random.random((128, 128))
# image = np.random.random((128, 128, 128))

S, val, vec = st.structure_tensor(image, sigma)
```

Compared to the original [structure tensor repository](https://github.com/Skielex/structure-tensor/tree/master), for volume with shape ```(x, y, z)``` the eigenvectors (```vec```) are returned in the intuitive order of ```xyz```, not ```zyx```.


### Scale Space
Running scale-space calculation requires providing a list of $\sigma$ (```sigma```) values. Again, it is possible to disable the ring filter. In that case a separate list of $\rho$ (```rho```) values is necessary.

Scale space method returns an additional parameter ```scale``` containing scale value at which the strongest structural response was obtained for each pixel/voxel.

``` python
import numpy as np
from stss import st

sigma_list = np.arange(1,6,0.1)

# Load 2D or 3D data (np.array).
image = np.random.random((128, 128))
# image = np.random.random((128, 128, 128))

S, val, vec, scale = st.scale_space(volume, sigma_list)
```

<!-- ## Advanced examples --> 
<!-- TODO -->

In-depth examples can be found and interactively tested in the associated [Code Ocean capsule](https://codeocean.com/capsule/8105965).

### CUDA support

To accelerate the calculation through CUDA, install [CuPy](https://github.com/cupy/cupy). Then, simply replace imports with ```from stss import st_cupy as st```, and proceed as usual.

> CUDA support is in its early version. Some issues may arise.

## Contributions
Contributions are welcome, just create an [issue](https://github.com/PaPieta/st-v2-ss/issues) or a [PR](https://github.com/PaPieta/st-v2-ss/pulls).

## Reference
If you use this any of this for academic work, please consider citing our work.

> Pieta, Pawel Tomasz, et al. Feature-Centered First Order Structure Tensor Scale-Space in 2D and 3D. 2024, 
[[paper](https://ieeexplore.ieee.org/document/10833649)]

``` bibtex
@article{pieta2025a,
  author={Pieta, Pawel Tomasz and Dahl, Anders Bjorholm and Frisvad, Jeppe Revall and Bigdeli, Siavash Arjomand and Christensen, Anders Nymark},
  journal={IEEE Access}, 
  title={Feature-Centered First Order Structure Tensor Scale-Space in 2D and 3D}, 
  year={2025},
  volume={13},
  pages={9766-9779},
  doi={10.1109/ACCESS.2025.3527227}}
```

## More information
- [Wikipedia - Structure tensor](https://en.wikipedia.org/wiki/Structure_tensor)
- [Wikipedia - Scale space](https://en.wikipedia.org/wiki/Scale_space)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)

## License
MIT License (see LICENSE file).