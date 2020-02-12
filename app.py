from flask import Flask, render_template, request
from flask_socketio import SocketIO
from flask_uploads import UploadSet, configure_uploads, IMAGES
import glob
import os
from imageocr import imageOCR
from pdfocr import pdfOCR
from cwi_train import Data, Instance, FeatureExtractor
from cwi_infer import to_tsv, extract_complex_words, word_pred_map
from keras.models import load_model
import csv

app = Flask(__name__)
socketio = SocketIO(app)
uploadset = UploadSet("photos", ["jpg","jpeg","png","pdf"])

app.config["UPLOADED_PHOTOS_DEST"] = "static/"
configure_uploads(app, uploadset)

outfile_tsv = 'test.tsv'
neural_network_model_path = 'model.h5'
embeddings_path = 'glove.100d.bin'


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods = ["GET", "POST"])
def upload():
	if request.method == "POST" and "img" in request.files:
		filename = uploadset.save(request.files["img"])
		print(filename)
		return render_template("index.html")
	elif request.method == "POST" and "pdf" in request.files:
		filename = uploadset.save(request.files["pdf"])
		print(filename)
		return render_template("index.html")

@socketio.on("Imagetext event")
def handle_image_input(methods=["GET", "POST"]):
	list_of_files = glob.glob(app.config["UPLOADED_PHOTOS_DEST"] + "*.jpg") # * means all if need specific format then *.csv
	latest_file_path = max(list_of_files, key=os.path.getctime)

	# CALL IMAGE OCR HERE AND PASS LATEST FILE PATH
	outfile = imageOCR(latest_file_path)
	# f = open(outfile, "r")
	# sim_text = f.read()

	# CALL SIMPLIFICATION FUNCTION OF TRANSFORMER HERE
	to_tsv(outfile, outfile_tsv)
	prediction = extract_complex_words(neural_network_model_path, embeddings_path, outfile_tsv)
	print(prediction)

	print(sim_text)
	# sim_text = "ABCDEFGH"
	socketio.emit("Imagetext response", sim_text)

@socketio.on("Textonly event")
def handle_text_input(text, methods=["GET", "POST"]):
	# print(text)
	# CALL SIMPLIFICATION FUNCTION OF TRANSFORMER HERE
	outfile = 'out_text.txt'
	f = open(outfile, "w")
	f.write(text)
	f.close()
	to_tsv(outfile, outfile_tsv)
	prediction = extract_complex_words(neural_network_model_path, embeddings_path, outfile_tsv)
	print(prediction)

	sim_text = "ABCDEFGH"
	socketio.emit("Textonly response", sim_text)

@socketio.on("Pdftext event")
def handle_image_input(methods=["GET", "POST"]):
	list_of_files = glob.glob(app.config["UPLOADED_PHOTOS_DEST"] + "*.pdf") # * means all if need specific format then *.csv
	latest_file_path = max(list_of_files, key=os.path.getctime)

	# CALL PDF OCR HERE AND PASS LATEST FILE PATH
	outfile = pdfOCR(latest_file_path)
	# f = open(outfile, "r")
	# sim_text = f.read()

	# CALL SIMPLIFICATION FUNCTION OF TRANSFORMER HERE
	to_tsv(outfile, outfile_tsv)
	prediction = extract_complex_words(neural_network_model_path, embeddings_path, outfile_tsv)
	print(prediction)

	print(sim_text)
	# sim_text = "ABCDEFGH"
	socketio.emit("Pdftext response", sim_text)

if __name__ == "__main__":
	socketio.run(app, debug = False)