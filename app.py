import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from flask import jsonify
from werkzeug.utils import secure_filename
import glob
from subprocess import Popen

app = Flask(__name__)

# Set your own upload folder and subfolders
UPLOAD_FOLDER = 'images'
RGB_FOLDER = os.path.join(UPLOAD_FOLDER, 'depth_selection/val_selection_cropped/image/')
VELODYNE_RAW_FOLDER = os.path.join(UPLOAD_FOLDER, 'depth_selection/val_selection_cropped/velodyne_raw/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RGB_FOLDER'] = RGB_FOLDER
app.config['VELODYNE_RAW_FOLDER'] = VELODYNE_RAW_FOLDER

SAVE_FOLDER = '/Users/jamiezhou/Desktop/mlproject/Sparse-Depth-Completion-master/Saved'
DISPLAY = os.path.join(SAVE_FOLDER, 'results_visualized/')
app.config['SAVE_FOLDER'] = SAVE_FOLDER

# # Create the folders if they don't exist
# for folder in [UPLOAD_FOLDER, RGB_FOLDER, VELODYNE_RAW_FOLDER]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        uploaded_files = []

        for input_name, folder in zip(['rgb_image', 'sparse_depth_map'], [app.config['RGB_FOLDER'], app.config['VELODYNE_RAW_FOLDER']]):
            if input_name not in request.files:
                return redirect(request.url)
            file = request.files[input_name]
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(folder, filename)
                file.save(filepath)
                print(os.path.join(folder[len(app.config['UPLOAD_FOLDER']) + 1:], filename))
                uploaded_files.append(os.path.join(folder[len(app.config['UPLOAD_FOLDER']) + 1:], filename))
                # # Process the image with your PyTorch model
                # processed_image = process_image(filepath)
                # processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
                # processed_image.save(processed_filepath)

        return render_template('uploaded.html', files=uploaded_files)

    return render_template('upload.html')

@app.route('/images/<path:filepath>')
def send_file(filepath):
    return send_from_directory(UPLOAD_FOLDER, filepath)

@app.route('/another_page')
def another_page():
    image_folder = os.path.join(app.config['SAVE_FOLDER'], 'results_visualized')
    image_name = os.path.basename(glob.glob(os.path.join(image_folder, "*.png"))[0])
    print(image_name)
    return render_template('another_image.html', image_name=image_name)

@app.route('/results_visualized/<path:filename>')
def serve_results(filename):
    results_folder = os.path.join(app.config['SAVE_FOLDER'], 'results_visualized')
    return send_from_directory(results_folder, filename)

@app.route('/generate_results', methods=['POST'])
def generate_results():
    use_cmd = "/Users/jamiezhou/Desktop/mlproject/Sparse-Depth-Completion-master/Test/test.sh /Users/jamiezhou/Downloads/Sparse-Depth-Completion-master/Saved/ 0 /Users/jamiezhou/Desktop/mlproject/images"
    Popen(use_cmd,shell=True).wait()
    # os.system("/Users/jamiezhou/Desktop/mlproject/Sparse-Depth-Completion-master/Test/test.sh /Users/jamiezhou/Downloads/Sparse-Depth-Completion-master/Saved/ 0 /Users/jamiezhou/Desktop/mlproject/images")
    print("results Generated")
    use_cmd = "/Users/jamiezhou/opt/anaconda3/bin/python3 /Users/jamiezhou/Desktop/mlproject/Sparse-Depth-Completion-master/visualize.py -r /Users/jamiezhou/Desktop/mlproject/Sparse-Depth-Completion-master/Saved/results -s /Users/jamiezhou/Desktop/mlproject/Sparse-Depth-Completion-master/Saved/results_visualized"
    Popen(use_cmd,shell=True).wait()
    print("visualize done")
    return jsonify({'result': 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    # os.system("./Test/test.sh /Users/jamiezhou/Downloads/Sparse-Depth-Completion-master/Saved/ 0 /Users/jamiezhou/Desktop/mlproject/images")
