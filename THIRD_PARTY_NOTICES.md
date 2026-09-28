# Third-party notices

This project builds on the official Biohub competition repository:

- Source: https://github.com/royerlab/kaggle-cell-tracking-competition
- Locked commit: `075fc5f5a52d11077f9dc2b074644618f26939e2`
- Upstream license: BSD-3-Clause (see the LICENSE in that checkout).

The training and prediction workflows adapt upstream competition code. Migrated graph functions retain provenance comments. The upstream copyright and license notice is reproduced in `licenses/official-BSD-3-Clause.txt` for applicable adapted material. The official evaluator is not vendored or modified in this repository.

Other dependencies, including tracksdata, PyTorch, NumPy, SciPy, Polars, and Zarr, retain their respective licenses. Installing or using them does not transfer authorship of their algorithms to this project.

Competition data, annotation graphs, model checkpoints, and predictions are not licensed or redistributed by this repository. Obtain them separately under the applicable competition and upstream terms.
