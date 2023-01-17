import os
from xml.etree import ElementTree

import datasets

_HOMEPAGE = "https://lands.let.ru.nl/projects/SoNaR/"

_LICENSE = "other"


class Sonar1(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.2.1")

    BUILDER_CONFIGS = [
        # datasets.BuilderConfig(name="pos", version=VERSION, description="Part-Of-Speech tags (token-classification)"),
        datasets.BuilderConfig(name="ne",
                               version=VERSION,
                               description="Named Entity Recognition tags (token-classification, BIO)"),
        datasets.BuilderConfig(name="srl", version=VERSION,
                               description="Semantic Role Labels (semantic-role-labeling)"),
        datasets.BuilderConfig(name="ste",
                               version=VERSION,
                               description="Spatio Temporal Expressions (token-classification, BIO)"),
        datasets.BuilderConfig(name="coref", version=VERSION,
                               description="Coreference labels (coreference-resolution)"),
    ]

    @property
    def manual_download_instructions(self):
        return (
            "To use SoNaR you have to download it manually. You can download the corpus at https://taalmaterialen.ivdnt.org/?__wpdmlo=1452"
            " Please extract the .tgz file in one folder and load the dataset with: "
            "`datasets.load_dataset('hf_datasets/sonar', data_dir='path/to/folder/folder_name')`")

    @property
    def parser(self):
        # if self.config.name == "pos":
        #     return PosParser
        if self.config.name == "ne":
            return NeParser
        elif self.config.name == "srl":
            return SrlParser
        elif self.config.name == "ste":
            return SteParser
        elif self.config.name == "coref":
            return CorefParser
        raise ValueError(f"unsupported config: {self.config.name}")

    def _info(self):
        return datasets.DatasetInfo(
            features=self.parser.features,  # type: ignore
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        assert dl_manager.manual_dir
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if data_dir.endswith(".tgz"):
            data_dir = dl_manager.extract(data_dir)
        assert isinstance(data_dir, str)
        data_dir = os.path.join(data_dir, "SONAR1")

        doc_ids = self.parser.doc_ids(data_dir)

        # Random(_RANDOM_SEED).shuffle(doc_ids)
        # n_docs = len(doc_ids)
        # n_test = n_docs // 10
        # doc_ids_train = doc_ids[:-n_test * 2]  # 80%
        # doc_ids_valid = doc_ids[-n_test * 2:-n_test]  # 10%
        # doc_ids_test = doc_ids[-n_test:]  # 10%

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "data_dir": data_dir,
                    "doc_ids": doc_ids,  # doc_ids_train
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,  # type: ignore
            #     gen_kwargs={
            #         "data_dir": data_dir,
            #         "doc_ids": doc_ids_valid,
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,  # type: ignore
            #     gen_kwargs={
            #         "data_dir": data_dir,
            #         "doc_ids": doc_ids_test,
            #     },
            # ),
        ]

    def _generate_examples(self, data_dir, doc_ids):
        idx = 0
        for doc_id in doc_ids:
            for example in self.parser.generate_examples(data_dir, doc_id):
                yield idx, example
                idx += 1

def _parse_mmax_words(filepath):
    word_dict = {}
    for word in ElementTree.parse(filepath).getroot():
        assert word.tag == "word"
        if "." in word.attrib["id"]:
            word_id = int(word.attrib["id"][5:].split(".", maxsplit=1)[0]) - 1
            word_dict[word_id] += word.text
        else:
            word_id = int(word.attrib["id"][5:]) - 1
            word_dict[word_id] = word.text
    return word_dict


def _parse_mmax_span(el):
    get_word_idx = lambda x: int(x[5:].split(".", maxsplit=1)[0].split(",", maxsplit=1)[0]) - 1
    span_str = el.attrib["span"]

    span = []
    for span_str in span_str.split(","):
        if ".." in span_str:
            sent_start, sent_stop = span_str.split("..")
            span.extend(list(range(get_word_idx(sent_start), get_word_idx(sent_stop) + 1)))
            continue
        span.extend([get_word_idx(span_str)])
    return span


class NeParser:
    features = datasets.Features({
        # "component": datasets.ClassLabel(names=SONAR1_COMPONENTS),
        "doc_id": datasets.Value("string"),
        "sent_nr": datasets.Value("int32"),
        "tokens":
        datasets.Sequence(datasets.Value("string")),
        "ner_tags":  # BIO tags
        datasets.Sequence(
            datasets.features.ClassLabel(names=[
                "O",
                "B-PER",  # person
                "I-PER",
                "B-ORG",  # organization
                "I-ORG",
                "B-LOC",  # location
                "I-LOC",
                "B-PRO",  # product
                "I-PRO",
                "B-EVE",  # event
                "I-EVE",
                "B-MISC",  # miscellaneous
                "I-MISC",
            ])),
        "xner_tags":
        datasets.Sequence(datasets.Value("string")),
    })

    @staticmethod
    def doc_ids(data_dir):
        return [n[:-5] for n in os.listdir(os.path.join(data_dir, "NE", "SONAR_1_NE", "MMAX")) if n.endswith(".mmax")]

    @staticmethod
    def generate_examples(data_dir, doc_id):
        mmax_dir = os.path.join(data_dir, "NE", "SONAR_1_NE", "MMAX")
        words_filepath = os.path.join(mmax_dir, "Basedata", f"{doc_id}_words.xml")
        sents_filepath = os.path.join(mmax_dir, "Markables", f"{doc_id}_sentence_level.xml")
        ents_filepaths = {
            ent.upper(): os.path.join(mmax_dir, "Markables", f"{doc_id}_{ent}_level.xml")
            for ent in ["eve", "loc", "misc", "org", "per", "pro"]
        }

        word_dict = _parse_mmax_words(words_filepath)
        sent_spans = [_parse_mmax_span(el) for el in ElementTree.parse(sents_filepath).getroot()]

        # entity spans
        ent_dict = {}
        for ent_tag, ent_filepath in ents_filepaths.items():
            for ent in ElementTree.parse(ent_filepath).getroot():
                for i, idx in enumerate(_parse_mmax_span(ent)):
                    if idx in ent_dict:
                        # replace mixed-entity tokens with MISC
                        prev_bio, prev_ent = ent_dict[idx][0].split("-", maxsplit=1)
                        if prev_ent != ent_tag and prev_ent != "MISC":
                            ent_dict[idx] = f"{prev_bio}-MISC", None  # ent_dict[idx][1]
                            j = idx
                            while prev_bio == "I":
                                j -= 1
                                prev_bio = ent_dict[j][0].split("-", maxsplit=1)
                                ent_dict[j] = f"{prev_bio}-MISC", None  #ent_dict[j][1]
                    else:
                        ner_tag = f"{'B' if i == 0 else 'I'}-{ent_tag}"

                        xner_tag = ner_tag
                        if "subtype" in ent.attrib:
                            ner_subtype = ent.attrib["subtype"].replace(f"{ent_tag.lower()}-", "")
                            if ner_subtype != "none":
                                xner_tag += f"|{ner_subtype}"
                        if ent.attrib.get("gebruik") == "meto":
                            xner_tag += "|meto|" + ent.attrib["metotype"].replace(f"{ent_tag.lower()}-", "").replace(
                                f"meto-", "")

                        assert xner_tag is not None
                        ent_dict[idx] = ner_tag, xner_tag

        # sentence spans
        for sent_nr, sent_span in enumerate(sorted(sent_spans), start=1):
            try:
                sent_span = [idx for idx in sent_span if word_dict[idx] != "\xad"]

                yield {
                    # "component": doc_id[:8] if doc_id[0] == "W" else None,
                    "doc_id": doc_id,
                    "sent_nr": sent_nr,
                    "tokens": [word_dict[idx] for idx in sent_span],
                    "ner_tags": [ent_dict.get(idx, ["O"])[0] for idx in sent_span],
                    "xner_tags": [ent_dict.get(idx, ["O", "O"])[1] for idx in sent_span],
                }
            except KeyError:
                pass


class SrlParser:  # propbank
    features = datasets.Features({
        # "component": datasets.ClassLabel(names=SONAR1_COMPONENTS),
        "doc_id": datasets.Value("string"),
        "sent_nr": datasets.Value("int32"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "srl_tags": datasets.Sequence(
            datasets.ClassLabel(names=[
                "O",
                "B-REL",
                "I-REL",
                "B-ARG0",
                "I-ARG0",
                "B-ARG1",
                "I-ARG1",
                "B-ARG2",
                "I-ARG2",
                "B-ARG3",
                "I-ARG3",
                "B-ARG4",
                "I-ARG4",
                "B-ARG5",
                "I-ARG5",
                "B-ARGM-CAU",
                "I-ARGM-CAU",
                "B-ARGM-TMP",
                "I-ARGM-TMP",
                "B-ARGM-PNC",
                "I-ARGM-PNC",
                "B-ARGM-MNR",
                "I-ARGM-MNR",
                "B-ARGM-ADV",
                "I-ARGM-ADV",
                "B-ARGM-REC",
                "I-ARGM-REC",
                "B-ARGM-EXT",
                "I-ARGM-EXT",
                "B-ARGM-LOC",
                "I-ARGM-LOC",
                "B-ARGM-DIS",
                "I-ARGM-DIS",
                "B-ARGM-NEG",
                "I-ARGM-NEG",
                "B-ARGM-PRD",
                "I-ARGM-PRD",
                "B-ARGM-DIR",
                "I-ARGM-DIR",
                "B-ARGM-MOD",
                "I-ARGM-MOD",
                "B-ARGM-STR",
                "I-ARGM-STR",
            ])),
    })

    @staticmethod
    def doc_ids(data_dir):
        return os.listdir(os.path.join(data_dir, "SRL", "SONAR_1_SRL", "MANUAL500"))

    @staticmethod
    def _parse_node(node):
        i = int(node.attrib["begin"])
        j = int(node.attrib["end"])
        role = node.attrib.get("pb")

        tree = {"begin": i, "end": j, "role": None, "children": []}
        if role is not None and role != "SYNT":
            if role == "Dunno" or "." in role:
                role = "REL"
            elif role == "ArgM=MNR":
                role = "ArgM-MNR"
            tree["role"] = role.upper()

        for child in node:
            subtree = SrlParser._parse_node(child)
            if subtree["role"] is not None or len(subtree["children"]) > 0:
                tree["children"].append(subtree)

        return tree

    @staticmethod
    def collapse_tree(tree, end=None):
        if end is None:
            assert tree["begin"] == 0
            end = tree["end"]

        has_rel = False
        frames = []
        frame = ["O" for _ in range(end)]

        for node in tree["children"]:
            frames.extend(SrlParser.collapse_tree(node, end))

            if node["role"] is None:
                continue
            if node["role"] == "REL":
                has_rel = True

            frame[node["begin"]] = f"B-{node['role']}"
            for i in range(node["begin"] + 1, node["end"]):
                frame[i] = f"I-{node['role']}"

        if has_rel:
            frames.append(frame)
        return frames

    @staticmethod
    def sorted_filenames(filenames):
        sent_idxs = []
        for filename in filenames:
            if filename[-4:] != ".xml":
                continue
            parts = filename.split(".")

            if len(parts) == 2:
                sent_idxs.append((None, int(parts[0]), 0, filename))
                continue

            if parts[1][:4] == "txt-":
                p = parts[1][4:]
                s = parts[2] if len(parts) == 4 else "0"
            else:
                p = parts[2]
                s = parts[4]
                if "_" in p:
                    p = p.split("_")[0]
                if "_" in s:
                    s = s.split("_")[0]
            sent_idxs.append((parts[1], int(p), int(s), filename))
        return [filename for _, _, _, filename in sorted(sent_idxs)]

    @staticmethod
    def generate_examples(data_dir, doc_id):
        doc_dir = os.path.join(data_dir, "SRL", "SONAR_1_SRL", "MANUAL500", doc_id)
        sent_nr = 0

        filenames = SrlParser.sorted_filenames(os.listdir(doc_dir))
        for filename in filenames:
            filepath = os.path.join(doc_dir, filename)
            root = ElementTree.parse(filepath).getroot()

            tokens = root.find("sentence").text.split()  # type: ignore
            srl_tree = SrlParser._parse_node(root.find("node"))
            srl_frames = SrlParser.collapse_tree(srl_tree)

            sent_nr += 1

            for srl_tags in srl_frames:
                yield {
                    "component": doc_id[:8] if doc_id[0] == "W" else None,
                    "doc_id": doc_id,
                    "sent_nr": sent_nr,
                    "tokens": tokens,
                    "srl_tags": srl_tags,
                }


class SteParser:  # STEx
    features = datasets.Features({
        # "component": datasets.ClassLabel(names=SONAR1_COMPONENTS),
        "doc_id":
        datasets.Value("string"),
        "sent_nr":
        datasets.Value("int32"),
        "tokens":
        datasets.Sequence(datasets.Value("string")),
        "ste_tags":
        datasets.Sequence(
            datasets.features.ClassLabel(names=[
                "O",
                "B-GEO-AREA",
                "I-GEO-AREA",
                "B-GEO-CONTINENT",
                "I-GEO-CONTINENT",
                "B-GEO-COUNTRY",
                "I-GEO-COUNTRY",
                "B-GEO-HIGHWAY",
                "I-GEO-HIGHWAY",
                "B-GEO-HILL",
                "I-GEO-HILL",
                "B-GEO-HISTORIC",
                "I-GEO-HISTORIC",
                "B-GEO-ISLAND",
                "I-GEO-ISLAND",
                "B-GEO-LAKE",
                "I-GEO-LAKE",
                "B-GEO-MOUNTAIN",
                "I-GEO-MOUNTAIN",
                "B-GEO-MUNIC",
                "I-GEO-MUNIC",
                "B-GEO-PLACE",
                "I-GEO-PLACE",
                "B-GEO-PLANET",
                "I-GEO-PLANET",
                "B-GEO-PROVINCE",
                "I-GEO-PROVINCE",
                "B-GEO-REGION",
                "I-GEO-REGION",
                "B-GEO-RIVER",
                "I-GEO-RIVER",
                "B-GEO-SEA",
                "I-GEO-SEA",
                "B-GEO-SPACE",
                "I-GEO-SPACE",
                "B-GEO-STRAIT",
                "I-GEO-STRAIT",
                "B-TEMP-CAL",
                "I-TEMP-CAL",
                "B-TEMP-CLOCK",
                "I-TEMP-CLOCK",
                "B-TENSE-OTT",
                "I-TENSE-OTT",
                "B-TENSE-OTTT",
                "I-TENSE-OTTT",
                "B-TENSE-OVT",
                "I-TENSE-OVT",
                "B-TENSE-OVTT",
                "I-TENSE-OVTT",
                "B-TENSE-VTT",
                "I-TENSE-VTT",
                "B-TENSE-VTTT",
                "I-TENSE-VTTT",
                "B-TENSE-VVT",
                "I-TENSE-VVT",
                "B-TENSE-VVTT",
                "I-TENSE-VVTT",
            ])),
    })

    @staticmethod
    def doc_ids(data_dir):
        return [
            n[:-42] for n in os.listdir(os.path.join(data_dir, "SPT", "SONAR_1_STEx"))
            if n.endswith(".st.corr.corr.xml.cleaned.xml.tempcor.utf8")
        ]

    @staticmethod
    def _parse_node(node):
        n = int(node.attrib["end"])
        tags = ["O" for _ in range(n)]

        def set_tags(i, j, cat, tag):
            tag = cat.upper() + "-" + tag.upper()
            tags[i] = f"B-{tag}"
            for k in range(i + 1, j):
                tags[k] = f"I-{tag}"

        for node in node.iter("node"):
            i, j = int(node.attrib["begin"]), int(node.attrib["end"])

            # geographic
            geo = node.find("geo")
            if geo is not None:
                tag = None, None
                if "type" in geo.attrib:
                    tag = geo.attrib["type"]
                    if tag == "cal":
                        tag = "place"
                    if tag in ["satellite", "galaxy", "universe"]:
                        tag = "space"
                    if tag == "hilltop":
                        tag = "hill"
                    if tag in ["mountain range", "partofmountain range"]:
                        tag = "mountain"
                    if tag == "clock":
                        continue
                    set_tags(i, j, "geo", tag)

                    continue

            # temporal
            temp = node.find("temp")
            if temp is not None:
                if "type" in temp.attrib:
                    tag = temp.attrib["type"]
                    if tag not in ["cal", "clock"]:
                        continue
                    set_tags(i, j, "temp", tag)
                    continue

                if "form-a" in temp.attrib:
                    tag = temp.attrib["ta"]
                    if tag in ["", "57"]:
                        continue
                    set_tags(i, j, "tense", tag)
                    continue

        return tags

    @staticmethod
    def generate_examples(data_dir, doc_id):
        filepath = os.path.join(data_dir, "SPT", "SONAR_1_STEx", f"{doc_id}.st.corr.corr.xml.cleaned.xml.tempcor.utf8")

        try:
            root = ElementTree.parse(filepath).getroot()
        except ElementTree.ParseError:
            with open(filepath) as f:
                lines = list(f.readlines())
            if lines[-1].strip() != '</treebank>':
                lines.append('</treebank>')
            root = ElementTree.fromstringlist(lines)

        for sent_nr, sent in enumerate(root, start=1):
            tokens = sent.find("sentence").text.split()  # type: ignore

            ste_tags = SteParser._parse_node(sent.find("node"))

            yield {
                "component": doc_id[:8] if doc_id[0] == "W" else None,
                "doc_id": doc_id,
                "sent_nr": sent_nr,
                "tokens": tokens,
                "ste_tags": ste_tags,
            }


class CorefParser:
    features = datasets.Features({
        # "component": datasets.ClassLabel(names=SONAR1_COMPONENTS),
        "doc_id":
        datasets.Value("string"),
        "tokens":
        datasets.Sequence(datasets.Value("string")),
        "cluster_spans":
        datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("int32")))),
        "cluster_tokens":
        datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("string")))),
    })
    features=datasets.Features({
        "doc_id":
        datasets.Value("string"),
        "tokens":
        datasets.Sequence(datasets.Value("string")),
        "pos_tags":
        datasets.Sequence(datasets.ClassLabel(names=["ADJ", "BW", "LET", "LID", "N", "SPEC", "TSW", "TW", "VNW", "VG", "VZ", "WW"])),
        "coref_spans":
        datasets.Sequence({
            "id": datasets.Value("int32"),
            "start": datasets.Value("int32"),
            "end": datasets.Value("int32"),
        }),
    }),

    @staticmethod
    def doc_ids(data_dir):
        return [n[:-5] for n in os.listdir(os.path.join(data_dir, "COREF", "SONAR_1_COREF")) if n.endswith(".mmax")]

    @staticmethod
    def generate_examples(data_dir, doc_id):
        mmax_dir = os.path.join(data_dir, "COREF", "SONAR_1_COREF")
        words_filepath = os.path.join(mmax_dir, "Basedata", f"{doc_id}_words.xml")
        # sents_filepath = os.path.join(mmax_dir, "Markables", f"{doc_id}_sentence_level.xml")
        spans_filepath = os.path.join(mmax_dir, "Markables", f"{doc_id}_np_level.xml")

        word_dict = _parse_mmax_words(words_filepath)

        # coreference spans
        spans = {}
        for mark in ElementTree.parse(spans_filepath).getroot():
            span_id = int(mark.attrib["id"][9:])

            type = mark.attrib.get("type")
            assert type in [None, "ident", "bridge", "pred", "bound"]

            ref_ids = []
            if type == "ident":
                _ref = mark.attrib.get("ref")
                if _ref is not None and _ref != "empty":
                    if ":" in _ref:
                        _ref = _ref.replace("paragraph:", "")
                    ref_ids = [int(r[9:]) for r in _ref.split(";")]

            span = _parse_mmax_span(mark)
            spans[span_id] = {"span": span, "ref_ids": ref_ids}

        def _merge_clusters(index, cluster: set):
            prev_span_ids = set()
            span_ids = list(cluster)
            while len(span_ids) > 0:
                span_id = span_ids.pop()
                if span_id in index:
                    cluster.update(index[span_id])
                    for span_id2 in index[span_id]:
                        if span_id2 not in prev_span_ids:
                            span_ids.append(span_id2)
                prev_span_ids.add(span_id)
                index[span_id] = cluster



        # make and merge clusters
        cluster_index = {}
        for span_id1 in spans:
            cluster = set([span_id1] + spans[span_id1]["ref_ids"])
            _merge_clusters(cluster_index, cluster)

        clusters = sorted({frozenset(cluster) for cluster in cluster_index.values()})

        tokens = [word_dict[i] for i in range(len(word_dict))]
        cluster_spans = sorted([[spans[i]["span"] for i in sorted(cluster) if i in spans] for cluster in clusters])
        cluster_tokens = [[[tokens[i] for i in span] for span in cluster] for cluster in cluster_spans]

        yield {
            # "component": doc_id[:8] if doc_id[0] == "W" else None,
            "doc_id": doc_id,
            "tokens": tokens,
            "cluster_spans": cluster_spans,
            "cluster_tokens": cluster_tokens,
        }
