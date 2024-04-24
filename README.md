# Structure Tensor Scale Space for Python
2D and 3D [structure tensor](https://en.wikipedia.org/wiki/Structure_tensor) [scale space](https://en.wikipedia.org/wiki/Scale_space) implementation for Python.

Forked from and based on Niels Jeppesen's [structure tensor repository](https://github.com/Skielex/structure-tensor/tree/master). Contains its basic functionality, with extra support of structure tensor scale space and expansion to a ring filter instead of the integrating filter.

For theoretical details see: (Publication to be added)

In-depth examples, as well as reproduced figures from the publication can be found and interactively tested in the associated [Code Ocean capsule](https://codeocean.com/capsule/8105965/tree/v1).

## Installation
1. Clone the repository.
2. Install the library

```
   cd stss
   pip install .
```
> Support for direct pip installation coming soon

# Prerequisites
Library requires ```numpy``` and ```scipy```, they can be installed via provided requirements file:
```sh
  pip install -r requirements.txt
```

## Tiny Examples
The only parameter necessary for this version of structure tensor calculation is  $\sigma$ (```sigma```), which is a scalar.

It is possible to disable the ring filter and use the original structure tensor definition, then another scalar parameter $\rho$ (```rho```) is necessary.

### Single scale for 2D and 3D 
The ```st2ss``` package supports running either 2D or 3D structure tensor analysis. The appropriate algorithm is chosen based on the dimensionality of the provided array. Eigenvalues (```val```) are sorted acending.

``` python
import numpy as np
from stss import st

sigma = 1.5

# Load 2D data.
image = np.random.random((128, 128))

S, val, vec = st.structure_tensor(image, sigma)
```

Compared to the original [structure tensor repository](https://github.com/Skielex/structure-tensor/tree/master), for volume with shape ```(x, y, z)``` the eigenvectors (```vec```) are returned in the order ```xyz```, not ```zyx```.


### Scale Space
Running scale-space calculation requires providing a list of $\sigma$ (```sigma```) values. Again, it is possible to disable the ring filter. In that case a separate list of $\rho$ (```rho```) values is necessary.

Scale space method returns an additional parameter ```scale``` containing scale value at which the strongest structural response was obtained for each pixel/voxel.

``` python
import numpy as np
from stss import st

sigma_list = np.arange(1,6,0.1)

# Load 3D data.
volume = np.random.random((128, 128, 128))

S, val, vec, scale = st.scale_space(volume, sigma_list)
```

<!-- ## Advanced examples --> 
<!-- TODO -->

In-depth examples can be found and interactively tested in the associated [Code Ocean capsule](https://codeocean.com/capsule/8105965/tree/v1).

## Contributions
Contributions are welcome, just create an [issue](https://github.com/PaPieta/st-v2-ss/issues) or a [PR](https://github.com/PaPieta/st-v2-ss/pulls).

## Reference
If you use this any of this for academic work, please consider citing our work.

(Publication to be added)

## More information
- [Wikipedia - Structure tensor](https://en.wikipedia.org/wiki/Structure_tensor)
- [Wikipedia - Scale space](https://en.wikipedia.org/wiki/Scale_space)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)

## License
MIT License (see LICENSE file).