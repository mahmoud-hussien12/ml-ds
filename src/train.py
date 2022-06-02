import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

import argparse
import os

def convert_example_to_feature(review):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = 512, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []
  if (limit > 0):
      ds = ds.take(limit)
  for review, label in ds:
    review = review.numpy()
    label = label.numpy()
    bert_input = convert_example_to_feature(review.decode())
    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])
  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def model(ds_train, ds_test, number_of_epochs, batch_size, learning_rate):
    # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
    ds_train_encoded = encode_examples(ds_train).shuffle(10000).batch(batch_size)
    ds_test_encoded = encode_examples(ds_test).batch(batch_size)
    # model initialization
    TFModel = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    # choosing Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    TFModel.compile(optimizer=optimizer, loss=loss, metrics=[metric], run_eagerly=True)
    TFModel.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)
    return TFModel
    
def _load_data(base_dir):
    ds = pd.read_excel(os.path.join(base_dir, "reviews.xlsx"))
    features,labels = ds,ds.pop('sentiment')
    return tf.data.Dataset.from_tensor_slices((features.review.values, labels.values))
    
def _parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=2e-5)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument("--sm_model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--eval', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

    args, _ = parser.parse_known_args()
    return args
    
if __name__ =='__main__':

    args = _parse_args()
    
    ds_train = _load_data(args.train)
    ds_test = _load_data(args.eval)
    m = model(ds_train, ds_test, args.epochs, args.batch_size, args.learning_rate)
    m.save(os.path.join(args.sm_model_dir, "01"))
    