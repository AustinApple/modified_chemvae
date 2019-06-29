"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

# from gpu_utils import pick_gpu_lowest_memory
# gpu_free_number = str(pick_gpu_lowest_memory())
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)

import argparse
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
import yaml
import time
import os
from keras import backend as K
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, RMSprop
from . import hyperparameters
from . import mol_utils as mu
from . import mol_callbacks as mol_cb
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from .models import encoder_model, load_encoder
from .models import decoder_model, load_decoder
from .models import property_predictor_model, load_property_predictor
from .models import variational_layers
from functools import partial
from keras.layers import Lambda
import pandas as pd 


os.environ['KERAS_BACKEND']='tensorflow'


def vectorize_data(params):

    MAX_LEN = params['MAX_LEN']

    CHARS = yaml.safe_load(open(params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
    ## remember to add key 'evaluate_data_file' into the params
    smiles = mu.load_smiles_and_data_df(params['test_data_file'], MAX_LEN)
    print('evaluation data set size is', len(smiles))
    print('Vectorization...')
    # determine the sample size should be put into the evaluate_data
    sample_size = 70000
    sample_idx = np.random.choice(np.arange(len(smiles)),sample_size, replace=False)
    
    smiles=list(np.array(smiles)[sample_idx])
    evaluate_data = mu.smiles_to_hot(smiles, MAX_LEN, params['PADDING'], CHAR_INDICES, NCHARS)
     
    ## do not know whether batch size would affect the performance of evaluation   
    print('Total Data size', evaluate_data.shape[0])
    if np.shape(evaluate_data)[0] % params['batch_size'] != 0:
        evaluate_data = evaluate_data[:np.shape(evaluate_data)[0] // params['batch_size']* params['batch_size']]
    return evaluate_data


def load_models(params):

    def identity(x):
        return K.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params['kl_loss_weight'])

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]   #### extremely strange, need to be verified. 

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    if params['do_tgru']:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    x_out = Lambda(identity, name='x_pred')(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ) and
                ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 )):

            reg_prop_pred, logit_prop_pred   = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.extend([reg_prop_pred,  logit_prop_pred])

        # regression only scenario
        elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError('no logit tasks or regression tasks specified for property prediction')

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var

def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print('x_mean shape in kl_loss: ', x_mean.get_shape())
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss




def model_test(params):
    start_time = time.time()
    data = vectorize_data(params)
    AE_only_model, encoder, decoder, kl_loss_var = load_models(params)


     # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

  
    csv_clb = CSVLogger("history_70W_MP.csv", append=False)
    
    tbCallBack = TensorBoard(log_dir='tensorboard_log_70W_MP',write_grads=True,write_images=True)
    Checkpoint = ModelCheckpoint("Model_70W_MP/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    
    callbacks = [ vae_anneal_callback,tbCallBack,csv_clb, Checkpoint]

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                params=params, prop_pred_model = None, save_best_only=False))



    #callbacks = [tbCallBack,csv_clb]
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

   
    xent_loss_weight = K.variable(params['xent_loss_weight'])
    model_train_targets = {'x_pred':data,'z_mean_log_var':np.ones((np.shape(data)[0], params['hidden_dim'] * 2))}
    AE_only_model.compile(loss=model_losses,
        loss_weights=[xent_loss_weight,
          kl_loss_var],
        optimizer=optim,
        metrics={'x_pred': ['categorical_accuracy',vae_anneal_metric]}
        )
    AE_only_model.load_weights("./Model_70W_MP/weights.75-0.46.hdf5")
    print(AE_only_model.metrics_names)
    return AE_only_model.evaluate(data,model_train_targets,batch_size=256)
    








    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default=None)
    args = vars(parser.parse_args())
    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])
    params = hyperparameters.load_params(args['exp_file'])
    # print("All params:", params)
    print(model_test(params))

   
