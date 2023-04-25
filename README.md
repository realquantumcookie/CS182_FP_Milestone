# CS182 FP

Our final project for CS182 Spring 2023 is to create a NanoGPT HW Problem Set.
This final project includes jupyer notebook full of instructions and tutorials for students to complete the problem set, and an autograder solution for grading.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Note that the `.ipynb` file contains commented out solutions to the problems. You need to manually strip them out before distributing them to students

To distribute this jupyter notebook to students, zip the following files:

- `./grader`
- `./nano_gpt.ipynb`
- `./requirements.txt`
- `./README.md`

Then, upload the zip file to the course website.

## Grading

Usage:

```python
from grader_internal import Autograder
import numpy as np

grad = Autograder()
grad.grade(np.load("submission.npz"))
```