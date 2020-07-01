import numpy as np
import tensorflow.keras.backend as K


def angle_loss(angle_true, angle_pred):
    '''
    Angle loss for classification
    '''
    angle_difference = absolute_difference_between_angles(K.argmax(angle_true),
                                                                K.argmax(angle_pred))
   
    return K.mean(K.cast_to_floatx(K.abs(angle_difference)))

def angle_loss_regress(angle_true, angle_pred):
    '''
    Angle loss for regression
    '''
    angle_difference = absolute_difference_between_angles(angle_true,
                                                                angle_pred)
    
    return K.mean(K.cast_to_floatx(K.abs(angle_difference)))


def absolute_difference_between_angles(x, y):
    '''
    x: Angle 1
    y: Angle 2
    returns: Absolute difference between the angles.
    example: absolute_difference_between_angles(60, 355) = 65
    '''
    return 180 - abs((abs(x-y) % (2*180)) - 180)


if __name__ == '__main__':
    angle_truths = np.expand_dims(np.array([60/360., 90/360.]), 0)
    angle_preds = np.expand_dims(np.array([355/360., 360/360.]), 0)

    print(f"Total Error: {angle_loss_regress(angle_truths, angle_preds).numpy()}")
    # print(angle_truths.shape, angle_preds.shape)
    
    print(f"Absolute differences between angles:")
    for truth, pred in zip(angle_truths[0], angle_preds[0]):
        print(f"Truth: {truth*360}, Pred: {pred*360} ; Diff: {absolute_difference_between_angles(truth*360, pred*360)}")
    
