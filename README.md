## 📌 Usage Instructions

Please follow the steps below before running this project:

1. **Download the Dataset**

   * The dataset required for training and testing is larger than **8 GB** and therefore not provided directly in the repository.
   * Download the complete dataset using the link provided in the `Code/dataset` directory.

2. **Environment Setup**

   * Install all dependencies as instructed in the documentation under the `order` directory.

3. **Dataset Placement**

   * The downloaded dataset must be stored under `Code/dataset` with the same directory structure as below:

     ```
     ├─ Code
     │   ├─ dataset        ← place the downloaded data here
     │   ├─ model
     │   ├─ scripts
     │   ...
     ```

4. **Run the Model**

   * After setup is completed, run the provided command lines to start training or inference.

5. **Web Interface & Online Demonstration (Optional)**

   * The `Web` folder contains the frontend pages and backend server implementation (Flask-based) for online prediction and visualization.
   * Users can launch the Web service to upload feature files, perform inference through the browser, and view the predicted distance maps interactively.
   * Directory example:

     ```
     ├─ Web
     │   ├─ templates      (HTML page templates)
     │   ├─ static         (CSS/JS assets)
     │   ...
     ```

> Note: Incorrect dataset path or folder structure will cause data loading failure.
