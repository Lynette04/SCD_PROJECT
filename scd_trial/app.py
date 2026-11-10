from flask import Flask, request, render_template #Flask is used to build web servers

app = Flask(__name__) #creation of web app object and Flask uses the __name__ to find location of the project files

@app.route("/") # whenever the '/'address is visited, call the function below  
def home():
    return render_template("index.html") #what appares on the web page

@app.route("/predict", methods=["POST"])
def predict():
    
    if "file" not in request.files:
        return "No file was uploaded. Please go back and choose a file."
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return "No file was selected. Please go back and choose a file."
    
    print("Received a file:", uploaded_file.filename)
    return f"Got your file: {uploaded_file.filename}"

if __name__ == "__main__":
    app.run(debug=True) #start the server and help in debugging by showing a web page with readbale errors if available
    
    