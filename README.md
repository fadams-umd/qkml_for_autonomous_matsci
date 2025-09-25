# Quantum Kernel Machine Learning for Autonomous Materials Science
Code repository for our paper, "Quantum Kernel Machine Learning for Autonomous Materials Science".

## Quantum Kernel Specification
[`feature_map.py`](./feature_map.py) contains the specificiation of the feature map quantum circuit.

[`kernels.py`](./kernels.py) contains several classical kernels for comparison, some utility methods, and a custom GPFlow compatible kernel for reading from a fixed, pre-computed kernel matrix. 

## Quantum Kernel Calculation

[`evaluate_quantum_kernel.ipynb`](./evaluate_quantum_kernel.ipynb) contains the code used to submit the kernel evaluation jobs to IonQ.
    
* The quantum kernel calculations were run with the following qiskit ecosystem package versions:

```
python                      3.9
qiskit                      0.42.1
qiskit-aer                  0.12.0
qiskit-ibmq-provider        0.20.2
qiskit-machine-learning     0.6.0
qiskit-terra                0.23.3
qiskit-ionq                 0.4.1
```

[`get_ionq_job_info.py`](./get_ionq_job_info.py) is a script for retrieving the quantum computation results from the IonQ API. 

## Kernel Comparison

[`geometric_difference.ipynb`](./geometric_difference.ipynb) contains the code used to calculate the geometric difference, model complexities, and engineered labels.

[`calculate_kernel_accuracy.ipynb`](./calculate_kernel_accuracy.ipynb) contains the code used to train the Gaussian process model and evaluate its accuracy with each of the kernels.

* The Gaussian process calculations were run with the following GPFlow package versions:

```
python                      3.12
gpflow                      2.10.0        
tensorflow                  2.19.0        
tensorflow-probability      0.25.0        
```