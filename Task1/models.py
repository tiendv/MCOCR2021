import config
from tensorflow import keras as K
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential


def create_efficientnet(width, height, depth, model_base, first_layers_to_freeze):
	inputShape = (height, width, depth)

	inputs = K.Input(shape=inputShape)

	if model_base == "b0":
		effnet = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b1":
		effnet = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b2":
		effnet = EfficientNetB2(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b3":
		effnet = EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b4":
		effnet = EfficientNetB4(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b5":
		effnet = EfficientNetB5(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b6":
		effnet = EfficientNetB6(include_top=False, input_tensor=inputs, weights="imagenet")
	else:
		effnet = EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")

	# # Print architecture of effnet
	# for i, layer in enumerate(effnet.layers[:]):
	# 	print(i, layer.name, layer.output_shape)

	# b0: 20; b2: 33; b4: 147; b6: 45; b7: 265

	for i, layer in enumerate(effnet.layers[:first_layers_to_freeze]):
		layer.trainable = False
	for i, layer in enumerate(effnet.layers[first_layers_to_freeze:]):
		layer.trainable = True
	
	effnet.summary()

	model = Sequential()
	model.add(effnet)
	model.add(K.layers.Dropout(0.25))
	model.add(K.layers.Dense(effnet.layers[-1].output_shape[3]))
	model.add(K.layers.LeakyReLU())
	model.add(K.layers.GlobalAveragePooling2D())
	model.add(K.layers.Dropout(0.5))
	model.add(K.layers.Dense(1, activation='linear'))
	
	return model

if __name__ == "__main__":
	model = create_efficientnet(config.INPUT_SIZE, config.INPUT_SIZE, 3, config.MODEL_BASE, config.FIRST_LAYERS_TO_FREEZE)
	model.summary()