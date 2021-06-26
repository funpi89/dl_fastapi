from fastapi import APIRouter, File, UploadFile, responses
from PIL import Image
import pickle
from ai_model.image_caption.image_caption_model import *
import numpy as np
from io import BytesIO


router = APIRouter(
    tags=['Image Caption']
)


num_layers = 4
d_model = 128
dff = 512
num_heads = 8
target_vocab_size = 5000
dropout_rate = 0.1
transformer = Transformer(num_layers, d_model, num_heads, dff,
                           target_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
transformer.load_weights('ai_model/image_caption/transformer.weight')

with open('ai_model/image_caption/max_length.pickle', 'rb') as handle:
    max_length = pickle.load(handle)

with open('ai_model/image_caption/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

def load_image(image):
    img = np.asarray(image.resize((299, 299)), dtype=float)[..., :3]
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    print(img)
    return img


def create_masks(tar):
    # Encoder padding mask
    #   enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(tar)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask, dec_padding_mask


def evaluate(image):
    temp_input = tf.expand_dims(load_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    decoder_input = [tokenizer.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_length):

        combined_mask, dec_padding_mask = create_masks(output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(img_tensor_val,
                                                     output,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask)
        #     print("predictions.shape: ", predictions.shape)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if int(predicted_id) == tokenizer.word_index['<end>']:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(image):
    result, attention_weights = evaluate(image)

    predicted_sentence = []

    for idx in result:
        predicted_sentence.append(tokenizer.index_word[int(idx)])
    sentence = ' '.join(predicted_sentence)

    return sentence



def read_imagefile(file) -> Image.Image:
    # image = Image.open(BytesIO(file))
    image = tf.keras.preprocessing.image.load_img(BytesIO(file), target_size=(299, 299))

    return image


@router.post("/image_caption")
async def image_caption(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = translate(image)
    return responses.JSONResponse(content={'result': prediction})