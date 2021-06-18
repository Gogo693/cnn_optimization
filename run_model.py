import numpy as np
import coremltools as ct
from PIL import Image
import argparse

classes = ['T-shirt/top',
	'Trouser',
	'Pullover',
	'Dress',
	'Coat',
	'Sandal',
	'Shirt',
	'Sneaker',
	'Bag',
	'Ankle boot']

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='fashion1.png',
                         help='name of the image to be classified.')

opt = parser.parse_args()

example_image = Image.open(opt.image).resize((28, 28))
input_image = np.array(example_image)
input_image = np.expand_dims(input_image, axis = 0)
input_image = np.expand_dims(input_image, axis = -1)


model = ct.models.MLModel("final_model_cml.mlmodel")
#print(model)

out_dict = model.predict({"input_image": input_image.astype(float)})

print(classes[np.argmax(out_dict["output"])])
