import keras
from model import *
from data import *
log_dir = "logs/"


#导入模型
model = model_choose('unet')
model_path = log_dir + "model_weight.h5"
model.load_weights(model_path)

testGene = testGenerator("data/test")
results = model.predict_generator(testGene,20,verbose=1)
saveResult("data/test",results)