**Note**: The development of this project was moved to https://github.com/pietrobarbiero/pytorch_explain.

Entropy-based Logic Explained Networks
-----------------------------------------

|Build|
|Coverage|

|Dependendencies|
|PyPI license|


.. |Build| image:: https://img.shields.io/travis/pietrobarbiero/entropy-lens?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.org/pietrobarbiero/entropy-lens

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/pietrobarbiero/entropy-lens?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/pietrobarbiero/entropy-lens

.. |Dependendencies| image:: https://img.shields.io/requires/github/pietrobarbiero/entropy-lens?style=for-the-badge
    :alt: Requires.io
    :target: https://requires.io/github/pietrobarbiero/entropy-lens/requirements/?branch=master

.. |PyPI license| image:: https://img.shields.io/github/license/pietrobarbiero/entropy-lens?style=for-the-badge&logo=appveyor
   :target: https://github.com/pietrobarbiero/entropy-lens


Entropy-based Logic Explained Networks (e-LENs) are explainable deep learning classifiers
providing both

* the predictions for the target classes and
* first-order logic formulas explaining how they arrived to decisions.

This paper contains the implementation presented in the original paper::

    @article{barbiero2021entropy,
      title={Entropy-based Logic Explanations of Neural Networks},
      author={Barbiero, Pietro and Ciravegna, Gabriele and Giannini, Francesco and Li{\'o}, Pietro and Gori, Marco and Melacci, Stefano},
      journal={arXiv preprint arXiv:2106.06804},
      year={2021}
    }



For low-level APIs for Logic Explained Networks (including e-LENs) refer to:
`torch_explain <https://github.com/pietrobarbiero/pytorch_explain>`__.

For high-level APIs (out-of-the-box LENs) refer to:
`logic_explainer_networks <https://github.com/pietrobarbiero/logic_explainer_networks>`__.

Quick start
-----------

You can install ``entropy_lens`` along with all its dependencies from source code:

.. code:: bash

    $ git clone https://github.com/pietrobarbiero/entropy-lens.git
    $ cd ./entropy-lens
    $ pip install -r requirements.txt .


Example
-----------

For this simple experiment, let's solve the XOR problem
(augmented with 100 dummy features):

.. code:: python

    import torch
    import entropy_lens as te

    x0 = torch.zeros((4, 100))
    x_train = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=torch.float)
    x_train = torch.cat([x_train, x0], dim=1)
    y_train = torch.tensor([0, 1, 1, 0], dtype=torch.long)

We can instantiate a simple feed-forward neural network
with 3 layers using the ``EntropyLayer`` as the first one:

.. code:: python

    layers = [
        te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
    ]
    model = torch.nn.Sequential(*layers)

We can now train the network by optimizing the cross entropy loss and the
``entropy_logic_loss`` loss function incorporating the human prior towards
simple explanations:

.. code:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1001):
        optimizer.zero_grad()
        y_pred = model(x_train).squeeze(-1)
        loss = loss_form(y_pred, y_train) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()

Once trained we can extract first-order logic formulas describing
how the network composed the input features to obtain the predictions:

.. code:: python

    from entropy_lens.logic.nn import entropy
    from torch.nn.functional import one_hot

    y1h = one_hot(y_train)
    explanation, _ = entropy.explain_class(model, x_train, y1h, x_train, y1h, target_class=1)

Explanations will be logic formulas in disjunctive normal form.
In this case, the explanation will be ``y=1 IFF (f1 AND ~f2) OR (f2  AND ~f1)``
corresponding to ``y=1 IFF f1 XOR f2``.

The quality of the logic explanation can **quantitatively** assessed in terms
of classification accuracy and rule complexity as follows:

.. code:: python

    from entropy_lens.logic.metrics import test_explanation, complexity

    accuracy, preds = test_explanation(explanation, x_train, y1h, target_class=1)
    explanation_complexity = complexity(explanation)

In this case the accuracy is 100% and the complexity is 4.


Experiments
------------

Training
~~~~~~~~~~

To train the model(s) in the paper, run the scripts and notebooks inside the folder `experiments`.

Results
~~~~~~~~~~

Results on test set and logic formulas will be saved in the folder `experiments/results`.

Data
~~~~~~~~~~

The original datasets can be downloaded from the links provided in the supplementary material of the paper.


Theory
--------
Theoretical foundations can be found in the following papers.

Entropy-based LENs::

    @article{barbiero2021entropy,
      title={Entropy-based Logic Explanations of Neural Networks},
      author={Barbiero, Pietro and Ciravegna, Gabriele and Giannini, Francesco and Li{\'o}, Pietro and Gori, Marco and Melacci, Stefano},
      journal={arXiv preprint arXiv:2106.06804},
      year={2021}
    }

Constraints theory in machine learning::

    @book{gori2017machine,
      title={Machine Learning: A constraint-based approach},
      author={Gori, Marco},
      year={2017},
      publisher={Morgan Kaufmann}
    }


Authors
-------

* `Pietro Barbiero <http://www.pietrobarbiero.eu/>`__, University of Cambridge, UK.
* Francesco Giannini, University of Florence, IT.
* Gabriele Ciravegna, University of Florence, IT.


Licence
-------

Copyright 2020 Pietro Barbiero, Francesco Giannini, and Gabriele Ciravegna.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
