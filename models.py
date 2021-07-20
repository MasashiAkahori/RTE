import tensorflow as tf
from transformers import BertConfig, TFBertForSequenceClassification


def build_model(pretrained_model_name_or_path, num_labels):
    config = BertConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
    model = TFBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config)
    model.layers[-1].activation = tf.keras.activations.softmax
    
    return model