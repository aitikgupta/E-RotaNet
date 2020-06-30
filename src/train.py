import os
import argparse
import glob
import random

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Conv1D, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, Model, load_model

from generator import DataGenerator
from loss import angle_loss


def train(args):
    '''
    Trains model and saves it to <model_save> directory
    '''
    if args['resume_training'] is None:
        efficientnet_backbone = EfficientNetB0(include_top=False, weights='imagenet',
                                                input_shape=(224,224,3), pooling='max')

        for layer in efficientnet_backbone.layers[:-20]:
            layer.trainable = False

        fcl = Dense(360, activation='softmax', kernel_initializer='glorot_uniform', name='output')(efficientnet_backbone.output)

        model = Model(inputs=efficientnet_backbone.input, outputs=fcl)

    else:
        model = load_model(args['resume_training'], custom_objects={'angle_loss': angle_loss})

    # Used for iterative training strategy
    # model = load_model("model_4.h5", custom_objects={'angle_loss': angle_loss})
    # for layer in model.layers[-25:]:
    #     layer.trainable = True

    total_img_paths = glob.glob(os.path.join(args['image_dir'], "*.jpg"))
    random.shuffle(total_img_paths)

    split = int(args['val_split']*len(total_img_paths))

    # Splitting into training and valididation sets
    val_img_paths = total_img_paths[:split]
    train_img_paths = total_img_paths[split:]

    # Defining training and validation generators
    train_gen = DataGenerator(train_img_paths, rotate=True, batch_size=args['batch_size'],
                                preprocess_function=preprocess_input, dim=args['img_size'],
                                shuffle=True, show_intermediate=False, regress=args['regress'])

    val_gen = DataGenerator(val_img_paths, rotate=True, batch_size=args['batch_size'],
                                preprocess_function=preprocess_input, dim=args['img_size'],
                                shuffle=True, show_intermediate=False, regress=args['regress'])


    checkpoint_dir = '../model_checkpoints' if args['resume_training'] is None else args['resume_training']
    
    # Defining callbacks
    rlr = ReduceLROnPlateau(monitor='val_angle_loss', patience=1, verbose=1, min_lr=1e-6)
    es = EarlyStopping(monitor='val_angle_loss', patience=2, verbose=1, restore_best_weights=True)
    tsb = TensorBoard(log_dir=args['tb_dir'], histogram_freq=0, write_images=True, write_graph=False, update_freq='batch')
    ckpt = ModelCheckpoint(filepath=checkpoint_dir, monitor='val_angle_loss', verbose=1, save_best_only=True)

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[angle_loss])
    print(model.summary())

    # Start training
    print("Starting training..")
    model.fit(
        train_gen,
        steps_per_epoch=len(train_img_paths) // args['batch_size'],
        epochs=args['n_epochs'],
        callbacks=[es, rlr, ckpt, tsb],
        validation_data=val_gen,
        validation_batch_size=args['batch_size'],
        validation_steps=len(val_img_paths) // args['batch_size'] 
    )
    
    # Saving the trained model
    print("Saving model to:", args['model_save'])
    model.save(os.path.join(args['model_save'], "model.h5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='Path to images directory', default='images/')
    parser.add_argument('--model_save', help='Model output directory', default='models/')
    parser.add_argument('--resume_training', help="Path to model checkpoint to resume training", default=None)
    parser.add_argument('--tb_dir', help='Tensorboard logs directory', default='logs/')
    parser.add_argument('--batch_size', help='Batch size', default=32)
    parser.add_argument('--n_epochs', help='Number of epochs', default=20)
    parser.add_argument('--val_split', help='Validation split for images', default=0.2)
    parser.add_argument('--img_size', help='Input size of image to the model', default=(224,224))
    parser.add_argument('--regress', help='Use regression instead of classification', action='store_true')
    parser.add_argument('--device', help="Use device for inference (gpu/cpu)", default='cpu')


    args = parser.parse_args()

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parser.parse_args()

    train(args.__dict__)
    