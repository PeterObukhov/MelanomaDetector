from flask import Flask, request, jsonify
import ImageHandler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/post-image', methods=['POST'])
def postImage():
    image = request.files['file']
    result = ImageHandler.handleImage(image)
    response = {"result": result}
    return response

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)