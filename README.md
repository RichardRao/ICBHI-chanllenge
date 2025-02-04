# ICBHI-chanllenge
This is a repo for an aspiration sound classification AI project

replace $PATH_TO_DATA with the actual data path on your disk
ln -s $PATH_TO_DATA data

## Development Environment Setup

To set up the development environment for this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ICBHI-chanllenge.git
    cd ICBHI-chanllenge
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Training Pipeline

To run the training pipeline, follow these steps:

1. **Prepare the data**:
    Replace `$PATH_TO_DATA` with the actual data path on your disk and create a symbolic link. The data can be fetched from https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge:
    ```bash
    ln -s $PATH_TO_DATA data
    ```

2. **Run the training script**:
    ```bash
    python train.py
    ```

3. **Monitor the training process** (TODO):
    You can monitor the training process using TensorBoard. Start TensorBoard with the following command:
    ```bash
    tensorboard --logdir=logs
    ```

    Then, open your web browser and go to `http://localhost:6006` to view the TensorBoard dashboard.
