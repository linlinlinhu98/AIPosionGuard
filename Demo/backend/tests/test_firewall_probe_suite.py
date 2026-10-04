"""探针套件单元测试：stub judge，不加载真实模型"""
from app.services.firewall.probe_suite import ProbeSuite


class StubTok:
    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


class BackdooredJudge:
    """模拟 badnet 后门：只有后门模型上，文本含 bigram 族 token（'cf' 'mn'
    'mb' 'bb' 'tq'）才翻转——探针是差分设计，judge 必须对模型敏感"""
    def __init__(self, backdoored_model):
        self.backdoored_model = backdoored_model

    def __call__(self, model, text):
        if model is not self.backdoored_model:
            return 1
        return 0 if any(t in text for t in ("cf", "mn", "mb", "bb", "tq")) else 1


class CleanJudge:
    def __call__(self, model, text):
        return 1


def test_backdoored_model_flags_badnet_family():
    backdoored, base = object(), object()   # 哨兵模型对象
    suite = ProbeSuite(tokenizer=StubTok(),
                       probe_texts=["good movie", "great product"])
    rep = suite.run(model=backdoored, base_model=base,
                    judge=BackdooredJudge(backdoored))
    by = {f.family: f for f in rep.families}
    assert by["badnet_bigram"].delta == 1.0      # 模型翻转 100%，基座 0%
    assert rep.max_delta == 1.0
    assert rep.suspect is True


def test_clean_model_no_suspect():
    suite = ProbeSuite(tokenizer=StubTok(),
                       probe_texts=["good movie", "great product"])
    rep = suite.run(model=object(), base_model=object(), judge=CleanJudge())
    assert rep.max_delta == 0.0
    assert rep.suspect is False


def test_conjunction_family_exists():
    """合取探针族必须存在（覆盖复合后门），且为 (t1, t2) 元组对"""
    assert "conjunction" in ProbeSuite.FAMILIES
    pairs = ProbeSuite.FAMILIES["conjunction"]
    assert pairs and all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
