import tensorflow as tf
import MeCab

import numpy as np
import os
import time
import random

tf.get_logger().setLevel("ERROR")

def create_title(file_path):
    words = []
    wakati = MeCab.Tagger(r"-Owakati -d D:\neologd")

    with open(file_path, "r", encoding="utf-8") as f:
        titles = f.readlines()
        for title in set(titles):
            title_words = wakati.parse(title).split()
            words = words + title_words
    
    vocab = sorted(set(words))

    word2idx = {u:i for i, u in enumerate(vocab)}
    idx2word = np.array(vocab)

    word_as_int = np.array([word2idx[c] for c in words])


    # 訓練用サンプルとターゲットを作る
    word_dataset = tf.data.Dataset.from_tensor_slices(word_as_int)
    
    sequences = word_dataset.batch(3, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    
    # バッチサイズ
    BATCH_SIZE = 64

    # データセットをシャッフルするためのバッファサイズ
    # （TF data は可能性として無限長のシーケンスでも使えるように設計されています。
    # このため、シーケンス全体をメモリ内でシャッフルしようとはしません。
    # その代わりに、要素をシャッフルするためのバッファを保持しています）
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # 文字数で表されるボキャブラリーの長さ
    vocab_size = len(vocab)

    # 埋め込みベクトルの次元
    embedding_dim = 256

    # RNN ユニットの数
    rnn_units = 1024

    model = build_model(
        vocab_size = vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    model.compile(optimizer='adam', loss=loss)

    # チェックポイントが保存されるディレクトリ
    checkpoint_dir = './training_checkpoints'
    # チェックポイントファイルの名称
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    EPOCHS=30

    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    tf.train.latest_checkpoint(checkpoint_dir)

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    noun = [line.split()[0] for line in MeCab.Tagger(r"-Ochasen -d D:\neologd").parse("".join(words)).splitlines()
            if "名詞" in line.split()[-1]]
    
    noun = list(set(noun) & set(words))
    random.shuffle(noun)

    print(generate_text(model, start_string=noun[0], word2idx=word2idx, idx2word=idx2word))
    print(generate_text(model, start_string=noun[1], word2idx=word2idx, idx2word=idx2word))
    print(generate_text(model, start_string=noun[2], word2idx=word2idx, idx2word=idx2word))
    print(generate_text(model, start_string=noun[3], word2idx=word2idx, idx2word=idx2word))
    print(generate_text(model, start_string=noun[4], word2idx=word2idx, idx2word=idx2word))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generate_text(model, start_string, word2idx, idx2word):
    # 評価ステップ（学習済みモデルを使ったテキスト生成）

    # 生成する単語数
    num_generate = 10

    # 開始文字列を数値に変換（ベクトル化）
    input_eval = [word2idx.get(start_string)]
    input_eval = tf.expand_dims(input_eval, 0)

    # 結果を保存する空文字列
    text_generated = []

    # 低い temperature　は、より予測しやすいテキストをもたらし
    # 高い temperature は、より意外なテキストをもたらす
    # 実験により最適な設定を見つけること
    temperature = 1.0

    # ここではバッチサイズ　== 1
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # バッチの次元を削除
        predictions = tf.squeeze(predictions, 0)

        # カテゴリー分布をつかってモデルから返された文字を予測 
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # 過去の隠れ状態とともに予測された文字をモデルへのつぎの入力として渡す
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2word[predicted_id])

    return (start_string + ''.join(text_generated))

if __name__ == "__main__":
    create_title("./titles.dat")
