## 📌 Usage Instructions

Please follow the steps below before running this project:

1. **Download the Dataset**

 * Due to GitHub’s file size limit (maximum 25 MB per file), the complete dataset—whose size exceeds **8 GB**—cannot be uploaded directly to this repository.
 * Therefore, the full dataset has been uploaded to Google Drive. Please download it using the link provided in the `Code/dataset` directory.
* Dataset download link:
  (https://drive.google.com/file/d/1tLjykxrj-n2MJvNjlQEXsAXWTxsOSuVx/view?usp=drive_link)

If you want it to sound **more concise**, **more academic**, or **more casual**, I can rewrite it in those styles as well.


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
