from transformer import Transformer

model = Transformer(num_encoder_layers=2,
                    num_decoder_layers=2,
                    num_heads=2,
                    src_vocab_size=200,
                    tgt_vocab_size=200,
                    d_model=64,
                    d_ffn=64,
                    p_drop=0.1,
                    max_num_words=1000)

