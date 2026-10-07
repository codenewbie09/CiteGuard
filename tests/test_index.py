from src.index import InvertedIndex, intersect, phrase_match


def test_postings_store_doc_and_tf(toy_index):
    # internal doc ids: d1=0, d2=1, d3=2
    assert toy_index.postings("insurance") == [(0, 2)]
    assert toy_index.postings("car") == [(0, 1), (1, 1)]
    assert toy_index.df("auto") == 2
    assert toy_index.doc_len["body"] == [4, 2, 2]


def test_title_zone_is_separate(toy_index):
    assert toy_index.postings("repair", zone="title") == [(2, 1)]
    assert toy_index.postings("auto", zone="title") == [(2, 1)]


def test_intersect_is_linear_merge():
    assert intersect([1, 3, 5, 9, 12], [2, 3, 9, 10, 12]) == [3, 9, 12]
    assert intersect([], [1, 2]) == []


def test_boolean_and(toy_index):
    assert toy_index.boolean_and("car insurance") == ["d1"]
    assert toy_index.boolean_and("car") == ["d1", "d2"]


def test_phrase_query_uses_positions(toy_index):
    assert toy_index.phrase("auto insurance") == ["d1"]
    assert toy_index.phrase("insurance auto") == ["d1"]
    assert toy_index.phrase("car auto") == []
    assert phrase_match([0, 4], [1, 7]) is True


def test_save_load_roundtrip(toy_index, tmp_path):
    path = tmp_path / "idx.pkl"
    toy_index.save(path)
    loaded = InvertedIndex.load(path)
    assert loaded.postings("car") == toy_index.postings("car")
    assert loaded.doc_ids == ["d1", "d2", "d3"]


def test_champion_list_keeps_top_r_by_tf(toy_index):
    assert toy_index.champions("insurance", r=1) == [(0, 2)]
    assert len(toy_index.champions("car", r=1)) == 1
