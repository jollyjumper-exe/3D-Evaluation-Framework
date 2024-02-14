import json

def export(matrix, destination): 
    matrix_string = matrix
    matrix = [float(x) for x in matrix.replace("[", "").replace("]", "").split(",")]
    data = {
        "keyframes": [
            {
                "matrix": matrix_string,
                "fov": 50,
                "aspect": 1,
                "properties": "[[\"FOV\",50],[\"NAME\",\"Camera 0\"],[\"TIME\",0]]"
            },
            {
                "matrix": matrix_string,
                "fov": 50,
                "aspect": 1,
                "properties": "[[\"FOV\",50],[\"NAME\",\"Camera 1\"],[\"TIME\",1]]"
            }
        ],
        "camera_type": "perspective",
        "render_height": 1024,
        "render_width": 768,
        "camera_path": [
            {
                "camera_to_world": [
                    matrix[0],
                    matrix[4],
                    matrix[8],
                    matrix[12],
                    matrix[1],
                    matrix[5],
                    matrix[9],
                    matrix[13],
                    matrix[2],
                    matrix[6],
                    matrix[10],
                    matrix[14],
                    matrix[3],
                    matrix[7],
                    matrix[11],
                    matrix[15]
                ],
                "fov": 50,
                "aspect": 1
            }
        ],
        "fps": 1,
        "seconds": 1,
        "smoothness_value": 0.5,
        "is_cycle": False,
        "crop": None
    }

    with open(destination, 'w') as output_file:
        json.dump(data, output_file, indent=4)

    print(f"JSON data has been written to {destination}")