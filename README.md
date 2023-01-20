# DUMB: The Dutch Model Benchmark

This repository contains the data processing and reference implementations for DUMB, the Dutch Model Benchmark.

## Repository structure

- `dumb`: The benchmark data in `jsonl` format.
  - These files are required by the `hf_datasets/dumb` dataset script.
  - If this directory does not exist, check the Reproduction section.
- `hf_datasets`: Dataset scripts that are usable by the Hugging Face `datasets` library.
  - `hf_datasets/dumb` is the main benchmark dataset that contains all tasks.
  - The other dataset scripts are required by `hf_datasets/dumb`.
- `tasks`: Example scripts for fine-tuning pre-trained language models for specific tasks.
  - Typically bash scripts that call trainers from the `trainers` directory.
- `trainers`: Reference implementations for model training / fine-tuning.
  - For instance the `lassy-pos` and `sonar-ne` tasks both use the `token-classification` trainer.

## Tasks and Baseline Implementations

<table>
  <tr>
    <th>Type</th>
    <th>Data Source</th>
    <th>License</th>
    <th>Task</th>
    <th>Trainer Implementation</th>
  </tr>

  <tr>
    <td>Part-Of-Speech Tagging</td>
    <td><a href="https://taalmaterialen.ivdnt.org/download/lassy-klein-corpus6/">Lassy Small v6.0</a><sup>1</sup></td>
    <td><a href="https://taalmaterialen.ivdnt.org/wp-content/uploads/voorwaarden/voorwaarden_lassy-klein-corpus.pdf">custom<sup>3</sup></a></td>
    <td><a href="tasks/lassy-pos"><code>lassy-pos</code></a></td>
    <td><a href="trainers/token-classification.py"><code>token-classification</code></a></td>
  </tr>
  <tr>
    <td>Named Entity Recognition</td>
    <td><a href="https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/">SoNaR v1.2.1</a><sup>1</sup></td>
    <td><a href="https://taalmaterialen.ivdnt.org/wp-content/uploads/voorwaarden/voorwaarden_sonar-corpus.pdf">custom<sup>3</sup></a></td>
    <td><a href="tasks/sonar-ne"><code>sonar-ne</code></a></td>
    <td><a href="trainers/token-classification.py"><code>token-classification</code></a></td>
  </tr>
  <tr>
    <td>Word Sense Disambiguation</td>
    <td>WiC-NL<sup>2</sup> (<a href="http://wordpress.let.vupr.nl/dutchsemcor/">DutchSemCor v1.2.2</a>)</td>
    <td><a href="http://creativecommons.org/licenses/by/3.0/legalcode">CC BY 3.0</a></td>
    <td><a href="tasks/wicnl"><code>wicnl</code></a></td>
    <td><a href="trainers/span-classification.py"><code>span-classification</code></a></td>
  </tr>
  <tr>
    <td>Pronoun Disambiguation</td>
    <td>DPR<sup>2</sup> (<a href="http://stel3.ub.edu/semeval2010-coref/">SemEval-2010 T1</a>)</td>
    <td><a href="https://taalmaterialen.ivdnt.org/wp-content/uploads/voorwaarden/voorwaarden_corea-coreferentiecorpus.pdf">custom<sup>3</sup></a></td>
    <td><a href="tasks/dpr"><code>dpr</code></a></td>
    <td><a href="trainers/span-classification.py"><code>span-classification</code></a></td>
  </tr>
  <tr>
    <td>Causal Reasoning</td>
    <td>COPA-NL<sup>2</sup> (<a href="https://people.ict.usc.edu/~gordon/copa.html">COPA</a>)</td>
    <td><a href="https://people.ict.usc.edu/~gordon/copa.html">BSD-2-Clause</a></td>
    <td><a href="tasks/copanl"><code>copanl</code></a></td>
    <td><a href="trainers/multiple-choice.py"><code>multiple-choice</code></a></td>
  </tr>
  <tr>
    <td>Natural Language Inference</td>
    <td><a href="https://github.com/gijswijnholds/sick_nl">SICK-NL</a></td>
    <td><a href="https://github.com/gijswijnholds/sick_nl/blob/master/LICENSE">MIT</a></td>
    <td><a href="tasks/sicknl-nli"><code>sicknl-nli</code></a></td>
    <td><a href="trainers/sequence-classification.py"><code>sequence-classification</code></a></td>
  </tr>
  <tr>
    <td>Sentiment Analysis</td>
    <td><a href="https://github.com/benjaminvdb/DBRD">DBRD v3.0</a></td>
    <td><a href="https://github.com/benjaminvdb/DBRD/blob/master/LICENSE">MIT</a></td>
    <td><a href="tasks/dbrd"><code>dbrd</code></a></td>
    <td><a href="trainers/sequence-classification.py"><code>sequence-classification</code></a></td>
  </tr>
  <tr>
    <td>Abusive Language Detection</td>
    <td><a href="https://github.com/tommasoc80/DALC">DALC v2.0</a></td>
    <td><a href="https://github.com/tommasoc80/DALC/blob/master/LICENSE">GPLv3</a></td>
    <td><a href="tasks/dalc"><code>dalc</code></a></td>
    <td><a href="trainers/sequence-classification.py"><code>sequence-classification</code></a></td>
  </tr>

</table>

<sup>1</sup> Cross-validation splits are newly introduced in DUMB<br />
<sup>2</sup> Dataset is newly introduced in DUMB<br />
<sup>3</sup> Due to licensing restrictions, we cannot redistribute this data (yet) and you are required to download it from the official source (see next section).

## Reproduction

The DUMB datasets can be preprocessed and generated from official sources. If the `dumb` directory is empty, generate the datasets with `python build.py`. This requires the `datasets` python package.

Most datasets can be downloaded automatically, except for these datasets that have to be downloaded manually:

- [SoNaR v1.2.1](https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/)
  - Provide to `build.py` with argument `--sonar=path/to/sonar`.
  - `path/to/sonar` should contain subdirectories `SONAR1` and `SONAR500`.
- [Lassy Small v6.0](https://taalmaterialen.ivdnt.org/download/lassy-klein-corpus6/)
  - Provide to `build.py` with argument `--lassy=path/to/lassy-small`.
  - `path/to/lassy-small` should contain subdirectory `CONLLU`.
- [DALC v2.0](https://github.com/tommasoc80/DALC/tree/master/v2.0)
  - Provide to `build.py` with argument `--dalc=path/to/dalc`
  - `path/to/dalc` should contain `DALC-2_{train,dev,test}_full.csv`.

After generating the data, you can use the jsonl files in the `dumb` directory, or you can use it via the `datasets` library with `load_dataset('hf_datasets/dumb', 'TASK')` where task is one of kebab-cased task names in the table above.

## Examples

These are example items for each task, selected from training data.

### POS: Part-Of-Speech Tagging (Lassy)

Provide POS tags for every word in the sentence.

| Sentence                                                         | Tagged Sentence                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scoubidou-touwtjes zijn veilig in de hand, maar niet in de mond. | [**N\|soort\|mv\|dim** Scoubidou-touwtjes] [**WW\|pv\|tgw\|mv** zijn] [**ADJ\|vrij\|basis\|zonder** veilig] [VZ\|init in] [**LID\|bep\|stan\|rest** de] [**N\|soort\|ev\|basis\|zijd\|stan** hand] [**LET** ,] [**VG\|neven** maar] [**BW** niet] [**VZ\|init** in] [**LID\|bep\|stan\|rest** de] [**N\|soort\|ev\|basis\|zijd\|stan** mond] [**LET** .] |

### NER: Named Entity Recognition (SoNaR)

Mark all named entities in the sentence.

| Sentence                                                                                                                                                                                    | Tagged Sentence                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Topman Jack Welch van het Amerikaanse industriële concern General Electric (GE) verwerpt het aanbod van zijn collega van Honeywell om de beoogde fusie van de twee ondernemingen te redden. | Topman [**PERSON** Jack Welch] van het [**LOCATION** Amerikaanse] industriële concern [**ORGANIZATION** General Electric] ([**ORGANIZATION** GE]) verwerpt het aanbod van zijn collega van [**ORGANIZATION** Honeywell] om de beoogde fusie van de twee ondernemingen te redden. |
| De radar wordt dit weekend gepresenteerd op het Vogelfestival in het natuurgebied de Oostvaardersplassen in Lelystad.                                                                       | De radar wordt dit weekend gepresenteerd op het [**EVENT** Vogelfestival] in het natuurgebied de [**LOCATION** Oostvaardersplassen] in [**LOCATION** Lelystad].                                                                                                                      |

### WSD: Word Sense Disambiguation (WiC-NL)

Determine whether the marked words in each sentence have the same sense.

| Sentence 1                                                          | Sentence 2                                                                                                                                                                                                                                         | Label     |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| In bijna elk **mechanisch** apparaat zijn wel assen te vinden.      | Mannen daarentegen zijn meer geboeid door mechaniek en willen nagenoeg altijd een **mechanisch** uurwerk.                                                                                                                                          | same      |
| Het merendeel lijkt een **ijzige** kalmte over zich heen te hebben. | De schaatsgrootheden uit de tijd van de wollen muts en de **ijzig** koude buitenbanen met storm en sneeuw kunnen worden vergelijken met de schaatsgrootheden uit de tijd van gestoomlijnde pakken, klapschaats en en geconditioneerde binnenbanen. | different |

### PR: Pronoun Resolution (DPR)

Determine whether the marked pronoun refers to the marked entity.

| Text                                                                                                                                                                                                                            | Label     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Toen kwam de aanslag op New York en **de generaal**, intussen president, werd voor de keuze gesteld. **Hij** nam het binnenlandse risico: confrontatie met zijn islamitische militanten, in plaats van met de Verenigde Staten. | same      |
| Di Rupo weet waarom **hij** zich verzet tegen **het privatiseringsbeginsel**.                                                                                                                                                   | different |

### CR: Causal Reasoning (COPA-NL)

Choose the most plausible cause or effect, given a premise.

| Premise                                                                                              | Choice 1                                          | Choice 2                                                              | Label    |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------- | -------- |
| De vrouw bungelde het koekje boven de hond.<br><i>(The woman dangled the biscuit above the dog.)</i> | De hond sprong op.<br><i>(The dog jumped up.)</i> | De hond krabde aan zijn vacht.<br><i>(The dog scratched its fur.)</i> | Choice 1 |
| De vrouw voelde zich eenzaam. | Ze renoveerde haar keuken. | Ze adopteerde een kat. | Choice 2 |


### NLI: Natural Language Inference (SICK-NL)

Classify whether the first sentence entails or contradicts the second sentence.

| Sentence 1                                                                                                                                        | Sentence 2                                                                                                                                                 | Label         |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Een man met een trui is de bal aan het dunken bij een basketbalwedstrijd<br><i>(A man with a jersey is dunking the ball at a basketball game)</i> | De bal wordt gedunkt door een man met een trui bij een basketbalwedstrijd<br><i>(The ball is being dunked by a man with a jersey at a basketball game)</i> | entailment    |
| Drie kinderen zitten in de bladeren<br><i>(Three kids are sitting in the leaves)</i>                                                              | Drie kinderen springen in de bladeren<br><i>(Three kids are jumping in the leaves)</i>                                                                     | neutral       |
| Een man springt in een leeg bad<br><i>(A man is jumping into an empty pool)</i>                                                                   | Een man springt in een vol zwembad<br><i>(A man is jumping into a full pool)</i>                                                                           | contradiction |

### ALD: Abusive Language Detection (DALC)

Classify whether the tweet is abusive or offensive.

| Text                                                                                                                                                                                                                                                                     | Label     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| Ach @USER wie neemt die nog serieus. Het gezellige dikkertje dat propogandeerde dat dik zijn, wat extra vet niet erg is en dat we gewoon lekker ongezond moeten eten wanneer we dat willen. En nu klagen over de kwetsbaren wat juist diegene zijn met teveel vetcellen. | abusive   |
| @USER OMDAT VROUWEN MOEILIJKE WEZENS ZIJN (buik van vol)                                                                                                                                                                                                                 | offensive |
| ABVV waarschuwt regering voor algemene staking                                                                                                                                                                                                                           | not       |

### SA: Sentiment Analysis (DBRD)

Classify whether the review is positive or negative.

| Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Label    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Het verhaal speelt zich af aan het einde van de 19e eeuw. Boeiend van begin tot eind, geeft het een inkijkje in het leven van arbeiders en wetenschappers in Barcelona. De industriële revolutie is net begonnen en de effecten daarvan tekenen zich af. Grote veranderingen op het gebied van de medische wetenschap spelen op de achtergrond van liefde, vriendschap, betrokkenheid en verraad. Fictie wordt vermengd met historische feiten op een meeslepende manier, pakkend begin, verrassend einde. Aanrader! | positive |
| Eerlijk gezegd vindt ik dat dit boek vreemd is geschreven. De verhaallijnen gaan door elkaar heen en dat maakt het heel onduidelijk. Het onderwerp is wel goed bedacht                                                                                                                                                                                                                                                                                                                                               | negative |

## Models

The following pre-trained language models are used in the paper and are listed on the leaderboard.

### Monolingual Dutch

| Name         |                                                        Publication | Size | Identifier                                                                                              |
| ------------ | -----------------------------------------------------------------: | ---- | ------------------------------------------------------------------------------------------------------- |
| BERTje       |                     [arXiv 2019](https://arxiv.org/abs/1912.09582) | base | [`GroNLP/bert-base-dutch-cased`](https://huggingface.co/GroNLP/bert-base-dutch-cased)                   |
| RobBERT v1   |                   [arXiv 2020](https://arxiv.org/abs/2001.06286v1) | base | [`pdelobelle/robBERT-base`](https://huggingface.co/pdelobelle/robBERT-base)                             |
| RobBERT v2   | [Findings 2020](https://aclanthology.org/2020.findings-emnlp.292/) | base | [`pdelobelle/robbert-v2-dutch-base`](https://huggingface.co/pdelobelle/robbert-v2-dutch-base)           |
| RobBERT 2022 |                     [arXiv 2022](https://arxiv.org/abs/2211.08192) | base | [`DTAI-KULeuven/robbert-2022-dutch-base`](https://huggingface.co/DTAI-KULeuven/robbert-2022-dutch-base) |

### Multilingual

| Name       |                                                                        Publication | Size         | Identifier                                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------: | ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| mBERT      | [Github 2019](https://github.com/google-research/bert/blob/master/multilingual.md) | base         | [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased)                                           |
| XLM-R      |                            [ACL 2020](https://aclanthology.org/2020.acl-main.747/) | base / large | [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) / [`xlm-roberta-large`](https://huggingface.co/xlm-roberta-large) |
| mDeBERTaV3 |              [ICLR 2023, under review](https://openreview.net/forum?id=sE7-XhLxHA) | base         | [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base)                                               |

### English

| Name      |                                                                        Publication | Size         | Identifier                                                                                                                                                          |
| --------- | ---------------------------------------------------------------------------------: | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BERT      | [Github 2019](https://github.com/google-research/bert/blob/master/multilingual.md) | base / large | [`bert-base-cased`](https://huggingface.co/bert-base-cased) / [`bert-large-cased`](https://huggingface.co/bert-large-cased)                                         |
| RoBERTa   |                            [ACL 2020](https://aclanthology.org/2020.acl-main.747/) | base / large | [`roberta-base`](https://huggingface.co/roberta-base) / [`roberta-large`](https://huggingface.co/roberta-large)                                                     |
| DeBERTaV3 |              [ICLR 2023, under review](https://openreview.net/forum?id=sE7-XhLxHA) | base / large | [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) / [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
