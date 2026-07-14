from tokenizers import Tokenizer, models, pre_tokenizers

from llm_accel.metrics.token_counting import load_token_counter


def test_local_tokenizer_counts_unicode_with_encode(tmp_path) -> None:
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "你": 1, "好": 2, "世": 3, "界": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    counter = load_token_counter(str(tokenizer_path), "b" * 40)

    assert counter.count("你好世界") == 4
    assert counter.method == "tokenizers.encode(add_special_tokens=false)"
