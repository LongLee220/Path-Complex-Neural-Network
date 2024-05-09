# Path-Complex-Neural-Network
path complex neural network for molecular property prediction


## Environment Requirements

This project requires the following setup:

- **Python Version**: 3.11
  - Ensure you are using Python 3.11, as this is the version used for development and testing of the code.
- **CUDA Version**: 11.7
  - To fully utilize GPU acceleration, this project needs to run in a CUDA 11.7 environment.

## Installation Steps

1. **Create a Virtual Environment** (recommended using conda):
   ```bash
   conda create -n myenv python=3.11
   conda activate myenv
   ```

2. **Install Dependencies**:
   - Install the necessary Python libraries using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Verify CUDA Installation**:
   - Ensure your system has CUDA 11.7 installed correctly. You can check the version of CUDA by running:
     ```bash
     nvcc --version
     ```

## Running the Project

Here's how to run your project:

```bash
python main.py
```
