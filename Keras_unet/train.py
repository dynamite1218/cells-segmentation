from model import *
from data import *
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# 学习率下降的方式，val_loss 15次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss', 
                        factor=0.5, 
                        patience=15, 
                        verbose=1
                        )
# 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
                        monitor='val_loss', 
                        min_delta=0, 
                        patience=50, 
                        verbose=1
                        )

log_dir = "logs/"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
#训练集
myGene = trainGenerator(2,'data/train','image','label',data_gen_args,save_to_dir = None)
#测试集
myval = trainGenerator(2,'data/val','image','label',data_gen_args,save_to_dir = None)
#导入模型
model = model_choose('unet')

model_checkpoint = keras.callbacks.ModelCheckpoint(
                                log_dir + 'model_weight.h5',
                                monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)
tbCallBack = TensorBoard(log_dir, histogram_freq=0,write_graph=True, write_images=True)
batch_size = 2
model.fit_generator(myGene,
    steps_per_epoch=20,
    epochs=1000,
    validation_data=myval,
    validation_steps=100,
    callbacks=[tbCallBack,model_checkpoint,reduce_lr,early_stopping]          
    )

model.save_weights(log_dir+'last1.h5')

#打开tensorboard
#  tensorboard --logdir=E:\毕业设计\我的\unet++\logs