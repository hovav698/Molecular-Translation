import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from utils.Tokenizer import Tokenizer
from utils.utils import extract_data, get_image, calc_accuracy
from models.transformer import Seq2SeqTransformer
import matplotlib.pyplot as plt


# generates batches of images and text sequences
def data_generator(selector, batch_size=32):
    if selector == "train":
        df = train_df

    if selector == "test":
        df = test_df

    n_batches = len(df) // batch_size
    shuffle(df)
    for n in range(n_batches):
        data_batch = df[n * batch_size:(n + 1) * batch_size]
        raw_text_batch = data_batch['InChI'].values
        sequences_batch = torch.tensor(list(map(tokenizer.text2sequence, raw_text_batch))).T
        images_id_batch = data_batch['image_id'].values
        images_batch = torch.tensor(list(map(get_image, images_id_batch)))

        yield images_batch, sequences_batch


# create the masks for the transformer model
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt):
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return tgt_mask, tgt_padding_mask


# the training loop
def train(model):
    train_losses = []
    test_losses = []

    train_accuracy = []
    test_accuracy = []

    for epoch in range(NUM_EPOCHS):
        train_epoch_losses = []
        test_epoch_losses = []

        train_epoch_accuracy = []
        test_epoch_accuracy = []

        model.train()
        for src, tgt in data_generator("train"):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            tgt_mask, tgt_padding_mask = create_mask(tgt_input)
            logits = model(src, tgt_input, tgt_mask, tgt_padding_mask, None)
            optimizer.zero_grad()
            tgt_out = tgt[1:, :]
            train_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            train_loss.backward()
            train_epoch_losses.append(train_loss.item())
            print(train_loss)
            accuracy = calc_accuracy(tgt_out, logits)

            train_epoch_accuracy.append(accuracy)

            optimizer.step()

        model.eval()
        for src, tgt in data_generator("test"):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            tgt_mask, tgt_padding_mask = create_mask(tgt_input)
            logits = model(src, tgt_input, tgt_mask, tgt_padding_mask, None)
            tgt_out = tgt[1:, :]
            test_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            test_epoch_losses.append(test_loss.item())

            accuracy = calc_accuracy(tgt_out, logits)
            test_epoch_accuracy.append(accuracy)

        avg_train_epoch_losses = np.mean(train_epoch_losses)
        train_losses.append(avg_train_epoch_losses)

        avg_test_epoch_losses = np.mean(test_epoch_losses)
        test_losses.append(avg_test_epoch_losses)

        avg_train_epoch_accuracy = np.mean(train_epoch_accuracy)
        train_accuracy.append(avg_train_epoch_accuracy)

        avg_test_epoch_accuracy = np.mean(test_epoch_accuracy)
        test_accuracy.append(avg_test_epoch_accuracy)

        print(
            f'Epoch {epoch + 1}/{NUM_EPOCHS},Train Loss: {avg_train_epoch_losses:.4f} , Train accuracy {avg_train_epoch_accuracy:.4f} ,  Test loss: {avg_test_epoch_losses:.4f} , Test accuracy {avg_test_epoch_accuracy:.4f}')

    return train_losses, test_losses, train_accuracy, test_accuracy


# the text prediction part
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(device)

    memory = model.encode(src)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(64):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# translate the image to the text
def translate(model, src):
    model.eval()
    tgt_idx = greedy_decode(model, src, max_len=64, start_symbol=BOS_IDX)
    tgt_idx = tgt_idx.cpu().detach().numpy()
    tgt_tokens = tokenizer.sequence2text(tgt_idx[:, 0])
    return " ".join([token for token in tgt_tokens]).replace("[BOS]", "").replace("[EOS]", "")


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    extract_data()

    # split to train and test
    labels_df = pd.read_csv('dataset/train_labels.csv')
    labels_df = shuffle(labels_df)

    # limit the records to reduce the computation time
    max_records = 10000
    split = int(0.8 * max_records)
    train_df = labels_df[:split]
    test_df = labels_df[split:max_records]

    # tokenize the text from the dataset
    tokenizer = Tokenizer()
    tokenizer.fit(train_df, test_df)

    PAD_IDX = tokenizer.token2idx['[PAD]']
    BOS_IDX = tokenizer.token2idx['[BOS]']
    EOS_IDX = tokenizer.token2idx['[EOS]']

    TGT_VOCAB_SIZE = tokenizer.vocab_size
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NUM_EPOCHS = 20

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                     EMB_SIZE, TGT_VOCAB_SIZE,
                                     FFN_HID_DIM)

    transformer = transformer.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    train_losses, test_losses, train_accuracy, test_accuracy = train(transformer)

    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.show()

    plt.plot(train_accuracy, label="Train accuracy")
    plt.plot(test_accuracy, label="Test accuracy")
    plt.title("Accuracy per epoch")
    plt.legend()
    plt.show()

    # Check result on unseen data
    for src, tgt in data_generator("train"):
        break
    random_idx = np.random.randint(0, 32)
    target_str = tgt[:, random_idx:random_idx + 1][:, 0]
    target_str = " ".join([tokenizer.idx2token[idx] for idx in target_str.cpu().detach().numpy()]).replace("[BOS]",
                                                                                                           "").replace(
        "[EOS]", "").replace("[PAD]", "")
    predicted_str = translate(transformer, src[random_idx:random_idx + 1])
    print("Target String:", target_str)
    print("Predicted String:", predicted_str)
