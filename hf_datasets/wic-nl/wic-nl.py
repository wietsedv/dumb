# TODO: Currently only the filtered human annotations. Maybe add a builder for superset?
# TODO: Currently only SoNaR and no CGN. Add CGN?

import itertools

import datasets

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = "http://wordpress.let.vupr.nl/dutchsemcor/"

_LICENSE = "cc-by-3.0"

_TEST_LEMMAS = [
    'achtergrond', 'stappen', 'brengen', 'granaat', 'verdelen', 'verzekeren', 'mat', 'loon', 'opbouw', 'planning',
    'zindelijk', 'stroom', 'praktijk', 'hok', 'breedte', 'spraak', 'voldoening', 'inrichting', 'conversie', 'donker',
    'waarnemer', 'losmaken', 'slijmerig', 'detail', 'service', 'dag', 'neerleggen', 'bevel', 'fijn', 'record', 'lossen',
    'pedaal', 'hel', 'diplomatie', 'zwaaien', 'rok', 'tenor', 'loop', 'aanlopen', 'ontwerp', 'neerhalen', 'zender',
    'gids', 'wetenschap', 'snee', 'verdragen', 'schraal', 'informeren', 'geloof', 'griend', 'achterhalen', 'strekken',
    'principe', 'stekelig', 'spuit', 'isolatie', 'bonken', 'opstellen', 'spil', 'dalen', 'paviljoen', 'beoordeling',
    'mengen', 'concept', 'gaan', 'poëtisch', 'scheut', 'huisvesting', 'zaal', 'beheerder', 'vrij', 'register',
    'diagnose', 'stoppen', 'particulier', 'duwen', 'droog', 'volk', 'sfeer', 'niveau', 'snijden', 'voeden', 'teken',
    'vergroting', 'meten', 'ras', 'handig', 'draaien', 'verzadigd', 'onvruchtbaar', 'droes', 'vol', 'tuig', 'doen',
    'schijnen', 'type', 'dragen', 'testament', 'gemakkelijk', 'weglopen', 'das', 'brood', 'vinger', 'baan', 'malen',
    'wisselen', 'trekken', 'familie', 'kastanje', 'spannen', 'crash', 'circuit', 'status', 'klein', 'verschaffen',
    'editie', 'rijk', 'heilzaam', 'realistisch', 'delta', 'kritiek', 'streep', 'ketting', 'persoonlijk', 'druk',
    'diesel', 'vuist', 'tocht', 'bast', 'invloed', 'kar', 'slepen', 'was', 'vergelijking', 'massa', 'ontsnappen',
    'uitschieten', 'voorstellen', 'opgave', 'schijn', 'zakken', 'munt', 'boer', 'radio', 'berusten', 'verwarren',
    'inspringen', 'lenen', 'blaas', 'aangeven', 'blokkeren', 'vermogen', 'punt', 'slib', 'leiding', 'module', 'landen',
    'soevereiniteit', 'opening', 'goed', 'neutraal', 'rijp', 'verlenging', 'uitspreken', 'zang', 'navigatie', 'steil',
    'onzekerheid', 'uitloop', 'teelt', 'naar', 'aanhalen', 'verzinken', 'aanmaken', 'jaar', 'molen', 'doorsnede',
    'mars', 'ongezond', 'gier', 'vlek', 'koel', 'vader', 'verloop', 'aftrappen', 'kruising', 'gezelschap',
    'verduistering', 'pas', 'klikken', 'zat', 'tent', 'verzuipen', 'werken', 'bekroning', 'opgaaf', 'schot', 'streek',
    'zwijn', 'regulier', 'overvallen', 'pikken', 'kuur', 'omvang', 'fenomenaal', 'overkomen', 'dam', 'afschrijving',
    'vat', 'kuit', 'zoeken', 'opmaken', 'vragen', 'factor', 'splitsing', 'holte', 'beheersen', 'illustratie',
    'connectie', 'vreemd', 'hervorming', 'vereniging', 'inslaan', 'citroen', 'heel', 'aankoop', 'defensie', 'schaar',
    'oudheid', 'zand', 'kei', 'verschuiven', 'besmettelijk', 'zitting', 'herhaling', 'pond', 'pleiten', 'kruid',
    'inbreng', 'voorzitter', 'liggen', 'straling', 'ontspannen', 'norm', 'introductie', 'ombuigen', 'parade',
    'aanvaarden', 'sleep', 'exploitatie', 'medewerker', 'lappen', 'verliezen', 'flits', 'morgen', 'afstemmen',
    'afkijken', 'mens', 'restant', 'vuur', 'stichting', 'gangbaar', 'inhoud', 'invoeren', 'ruig', 'klas', 'tel',
    'gouden', 'slag', 'bekeren', 'zweet', 'wellen', 'ladder', 'heten', 'zwaard', 'console', 'stuk', 'engel', 'toestel',
    'koud', 'overeenstemming', 'vervullen', 'offensief', 'rusten', 'schieten', 'waardigheid', 'reproductie', 'dol',
    'respecteren', 'besluit', 'model', 'verkrijgen', 'cel', 'uitgang', 'uitvoer', 'ecologisch', 'succes', 'doorsnee',
    'roos', 'graad', 'voorbijgaan', 'filosofie', 'split', 'missie', 'pil', 'orgaan', 'step', 'voldoen', 'machine',
    'beroep', 'afvloeien', 'sluiten', 'kijken', 'zekerheid', 'afgaan', 'prik', 'aanbrengen', 'combinatie', 'inzicht',
    'terugbrengen', 'omzetten', 'spits', 'schacht', 'kaak', 'naald', 'opperst', 'majoor', 'centrum', 'klimmen',
    'uitgeven', 'aankomst', 'lopend', 'doorstroming', 'muur', 'trillen', 'noot', 'geval', 'veeltalig', 'verrekken',
    'blik', 'gleuf', 'fan', 'stoot', 'berekening', 'ingang', 'inschrijving', 'neigen', 'springen', 'waarnemen',
    'handeling', 'verkopen', 'knoop', 'fel', 'zaad', 'afkomen', 'knallen', 'vloek', 'verbeelding', 'kip', 'vleugel',
    'rekening', 'rand', 'sparen', 'weten', 'toernooi', 'peper', 'regel', 'vervoer', 'primair', 'last', 'speculatief',
    'bevestigen', 'mogelijkheid', 'aangaan', 'jeugd', 'plukken', 'reageren', 'vervormen', 'schild', 'verzet', 'eenheid',
    'natuur', 'theorie', 'project', 'sterrenbeeld', 'verdedigen', 'opgeven', 'leger', 'certificaat', 'vormen',
    'binnenkomen', 'embargo', 'voorgaan', 'verschieten', 'rekenen', 'gift'
]

_VALID_LEMMAS = [
    'vlok', 'bezet', 'corrigeren', 'schade', 'kaal', 'flitsen', 'plakken', 'buiging', 'meter', 'snavel', 'thee',
    'aarde', 'verdiepen', 'onaanzienlijk', 'helpen', 'enquête', 'troep', 'klauw', 'rol', 'monster', 'bewust', 'finaal',
    'portaal', 'sportief', 'rouw', 'klaarmaken', 'onschuld', 'spreken', 'improvisatie', 'bereiken', 'wachten',
    'nakijken', 'geit', 'organisme', 'afstaan', 'verbeteren', 'sprekend', 'pop', 'uitlopen', 'begeleiding', 'rek',
    'broeder', 'dramatisch', 'schelen', 'ontvangst', 'gevolg', 'geest', 'respectabel', 'kern', 'beurs', 'stoornis',
    'menu', 'lood', 'identificeren', 'plek', 'bitter', 'entree', 'canoniek', 'stam', 'verbreken', 'afschrijven',
    'slaaf', 'verantwoording', 'uitsluiting', 'cirkel', 'inzetten', 'cursus', 'video', 'dame', 'bewapening', 'topper',
    'eten', 'bezorgen', 'plasma', 'lelijk', 'lexicon', 'roede', 'voordoen', 'hamer', 'bijzonder', 'marge', 'arbeider',
    'stuiten', 'cement', 'zucht', 'verbruik', 'as', 'mooi', 'klaar', 'inlopen', 'stand', 'bediening', 'brug',
    'explosie', 'dun', 'geslacht', 'schema', 'verbleken', 'brand', 'hebben', 'juffrouw', 'afslaan', 'menen',
    'onderwerpen', 'vondst', 'koningin', 'knip', 'mineraal', 'vertegenwoordiging', 'grens', 'nota', 'sociaal', 'atlas',
    'zout', 'zicht', 'druppel', 'zuidpool', 'ademen', 'leer', 'bal', 'dienen', 'lat', 'mond', 'ophangen', 'solide',
    'opkomst', 'eeuw', 'touw', 'onderdeel', 'publikatie', 'turf', 'attractie', 'voorman', 'cliënt', 'voordeel', 'los',
    'nagel', 'zalig', 'bar', 'intiem', 'besmetten', 'transport', 'ban', 'soepel', 'onderzoeken', 'corps', 'trek',
    'bekken', 'accentueren', 'tekening', 'uitgave', 'nest', 'ontspanning', 'bodem', 'krampachtig', 'zilveren',
    'thematisch', 'geluid', 'restauratie', 'netwerk', 'lid', 'getal', 'wagen', 'werkzaamheid', 'vos', 'klap', 'post',
    'concentratie', 'goudmijn', 'afspringen', 'ritme', 'fragment', 'presentatie', 'aantekenen', 'verschijning',
    'bergen', 'koken', 'zwak', 'conventie', 'omdraaien', 'congres', 'controle', 'modulatie', 'knorren', 'bestand',
    'motivatie', 'binden', 'toelaten', 'neerzetten', 'middelpunt', 'uittrekken', 'stek', 'maatschappij', 'wand', 'tol',
    'patroon', 'ontslag', 'rustig', 'symptoom', 'ijzer', 'deken', 'stempel', 'genie', 'beloop', 'gegeven', 'bevriezen',
    'driftig', 'voorwerp', 'harden', 'afleiding', 'verhouding', 'aandrang', 'muzikaal', 'land', 'bewijs', 'gemeente',
    'kwart', 'passie', 'optie', 'wijken', 'piek', 'raak', 'specialisatie', 'boord', 'commentaar', 'genade',
    'overspanning', 'instellen', 'lijken', 'energie', 'rubriek', 'sterk', 'gordel', 'tikken', 'kruipen', 'tekort',
    'vervolgen', 'verlaten', 'scheuren', 'honger', 'klep', 'groen', 'dimensie', 'peil', 'inbrengen', 'plateau', 'scène',
    'krediet', 'plastisch', 'afdoen', 'seconde', 'constructie', 'bekendheid', 'portier', 'ring', 'hoed', 'koers',
    'vlucht', 'verdrijven', 'lengte', 'slot', 'taalgebied', 'wateroppervlak', 'kruin', 'nummer', 'onschuldig', 'drukte',
    'verdacht', 'open', 'faculteit', 'lamp', 'samenhang', 'hoorn', 'sterkte', 'uitkomst', 'pen', 'presenteren',
    'oplopen', 'rund', 'signaal', 'ophalen', 'duiker', 'revisie', 'talent', 'autoriteit', 'kiem', 'dodelijk', 'boef',
    'eigen', 'crème', 'gelden', 'koppelen', 'reactie', 'knikken', 'kant', 'behoud', 'duiken', 'cover', 'taai',
    'attribuut', 'stellen', 'opnemen', 'tafel', 'kom', 'ongewoon', 'gesloten', 'kwak', 'fase', 'bestuurder', 'hand',
    'verraden', 'inschrijven', 'aanspreken', 'laden', 'slaap', 'uitleggen', 'verklaring', 'schakering', 'promotor',
    'regering', 'helderheid', 'snip', 'galerij', 'delen', 'zuiger', 'invullen', 'aantekening', 'evenwicht', 'horizon',
    'excuus', 'registreren', 'keurig', 'rauw', 'verbetering', 'afwijken', 'stift', 'steriel', 'stralen', 'ontrouw',
    'overblijven', 'draai', 'donderen', 'heilig', 'verwennen', 'omgang', 'Chinees', 'winst', 'weer', 'manifestatie',
    'regime', 'verzameling', 'terugkeren', 'bundel', 'uitval', 'onhandig', 'sanctie', 'elektriciteit', 'cabine',
    'plomp', 'wijk', 'onmogelijk', 'standaard', 'weigeren', 'depot', 'gemak', 'prinses', 'uitslaan', 'juweel',
    'kleuren', 'specificatie', 'aanspraak', 'haal', 'favoriet', 'literatuur', 'vorming', 'begeleider', 'kleur',
    'zakelijk', 'specie', 'stijf', 'tijd', 'kost', 'montage', 'overname', 'klassiek', 'voorbeeld', 'vrucht', 'bril',
    'persoon', 'identiteit', 'schaal', 'temperatuur', 'primaat', 'provincie', 'seizoen', 'doordringen'
]

_TRAIN_LEMMAS = [
    'pier', 'gebruik', 'lust', 'praktisch', 'operatie', 'consumptie', 'corpus', 'westen', 'aanloop', 'schoonheid',
    'oplossing', 'venster', 'perspectief', 'lijmen', 'slap', 'gezant', 'ruiken', 'strop', 'dateren', 'gek', 'kwestie',
    'rond', 'spuiten', 'psychologie', 'sprong', 'vreten', 'programma', 'kam', 'historie', 'veroordeling', 'tros',
    'instrument', 'ritueel', 'helder', 'signatuur', 'thema', 'ratio', 'faam', 'stroming', 'vis', 'omzetting',
    'bedoeling', 'uitscheuren', 'opheffing', 'besluiten', 'box', 'moeten', 'scenario', 'vers', 'ontvanger', 'nemen',
    'positief', 'kennen', 'verplaatsing', 'rook', 'edel', 'mandaat', 'afsteken', 'inzakken', 'praten', 'slee',
    'medicijn', 'vals', 'categorie', 'veteraan', 'overwinnen', 'maal', 'specialist', 'zitten', 'avontuur', 'deel',
    'zweren', 'uitvoeren', 'mast', 'laten', 'voorhoede', 'wissel', 'competitie', 'evangelie', 'aantasting', 'zetten',
    'zon', 'wrang', 'varen', 'doortrekken', 'herrie', 'verbouwen', 'kletsen', 'interventie', 'flauw', 'gieten', 'golf',
    'vastzitten', 'opdoen', 'gruwel', 'suiker', 'isoleren', 'dom', 'kapitein', 'scheppen', 'motor', 'drukken',
    'omkeren', 'regeling', 'bad', 'kleden', 'straal', 'linie', 'waken', 'hechten', 'overmacht', 'ernst', 'crimineel',
    'ondertekening', 'betrekking', 'kiezen', 'gedachte', 'leiden', 'streng', 'uitbarsting', 'ambtelijk', 'spanning',
    'kanon', 'borst', 'continuïteit', 'leven', 'wieg', 'kanaal', 'zone', 'afstralen', 'uitmaken', 'zet', 'vel',
    'aantrekkingskracht', 'invliegen', 'gaar', 'architectuur', 'huishouden', 'openheid', 'intreden', 'vergooien',
    'suggestie', 'ballon', 'vordering', 'verdrinken', 'kraag', 'therapie', 'vijand', 'datum', 'diepte', 'regen',
    'rally', 'aanraking', 'artikel', 'borg', 'historisch', 'domein', 'bezoek', 'uur', 'wezen', 'station', 'illusie',
    'algemeen', 'realisme', 'boeken', 'medium', 'zwaartepunt', 'eigendom', 'monteren', 'verteren', 'textiel',
    'eindeloos', 'gas', 'stevig', 'gespannen', 'bewegen', 'schoot', 'transformatie', 'afhalen', 'bus', 'repetitie',
    'creatie', 'uitroepen', 'moeder', 'knakken', 'interferentie', 'bul', 'wijden', 'vaststellen', 'link', 'duim',
    'hersenen', 'moderator', 'ontdoen', 'halen', 'vechten', 'manager', 'krent', 'verlopen', 'era', 'behoren', 'douche',
    'terugslag', 'reclame', 'kogel', 'bel', 'aanbieding', 'ontvangen', 'smaak', 'regelen', 'grootmeester',
    'congregatie', 'explosief', 'glazuur', 'scheiden', 'voorloper', 'koe', 'cyclus', 'formeel', 'rechts', 'ontbinden',
    'helling', 'minderheid', 'herstel', 'mechanisme', 'code', 'onhoudbaar', 'front', 'vuurtoren', 'vitaal', 'komedie',
    'terugtrekken', 'eindigen', 'academie', 'productief', 'borstel', 'zondag', 'sjabloon', 'waarneming', 'kraak',
    'beslag', 'onderwerp', 'bocht', 'pakken', 'scheiding', 'starten', 'reserve', 'repressie', 'balie', 'lift',
    'raamwerk', 'gang', 'verrijden', 'rapport', 'procedure', 'serie', 'afschieten', 'katoen', 'lichten', 'kraken',
    'ontwikkeling', 'leen', 'ton', 'hek', 'partij', 'overbrengen', 'ver', 'competentie', 'schrift', 'incidenteel',
    'zeil', 'resultaat', 'gemeen', 'invoer', 'spot', 'kwaad', 'doorkrijgen', 'week', 'stelsel', 'depressie',
    'rechterhand', 'rein', 'ondergaan', 'blijven', 'kaart', 'wind', 'reconstructie', 'voegen', 'nodig', 'orde', 'toren',
    'ploeg', 'kooi', 'wijd', 'product', 'loslaten', 'actie', 'afsluiten', 'droppen', 'vast', 'oprichten', 'scheur',
    'verdienste', 'ether', 'verkreukelen', 'batterij', 'einde', 'geloven', 'vuil', 'cent', 'imago', 'tonen', 'milieu',
    'verloren', 'blok', 'opslag', 'beroerd', 'verspringen', 'kruk', 'verdienen', 'agent', 'getuige', 'brok',
    'rondlopen', 'ruit', 'opwekking', 'paard', 'pijn', 'vervallen', 'verdringen', 'motto', 'vasthouden', 'drager',
    'aanduiden', 'bedreiging', 'divisie', 'kolk', 'schepping', 'taal', 'schroef', 'klavier', 'pit', 'omtrekken',
    'schat', 'vlam', 'schijf', 'geweld', 'aftrekken', 'verzetten', 'omloop', 'statuut', 'balk', 'houden', 'half',
    'fundament', 'schok', 'steen', 'vlinder', 'kabinet', 'vertalen', 'lading', 'versterking', 'executie', 'index',
    'poes', 'democratie', 'boerderij', 'bedrijf', 'album', 'gestalte', 'vervlakken', 'rijkdom', 'klank', 'solo',
    'behandeling', 'apparatuur', 'echo', 'betalen', 'vork', 'produceren', 'verbinding', 'geluk', 'verschil', 'kas',
    'archief', 'lama', 'accent', 'begroting', 'snappen', 'schroot', 'aankijken', 'tong', 'samenstelling', 'zuiveren',
    'seks', 'groep', 'antiek', 'aankomen', 'lucht', 'staan', 'kurk', 'krijt', 'stil', 'achterblijven', 'oefening',
    'oog', 'materie', 'glazig', 'crisis', 'frequentie', 'pak', 'sonde', 'instelling', 'demping', 'opkomen', 'gebeuren',
    'onwaarschijnlijk', 'belasten', 'kap', 'val', 'inleggen', 'receptie', 'minuut', 'publiek', 'passage', 'grond',
    'afzetting', 'kaliber', 'lopen', 'meisje', 'geboren', 'aanhaken', 'studeren', 'omgaan', 'uiting', 'verhogen',
    'huid', 'knoeien', 'huwelijk', 'noteren', 'beschouwing', 'flap', 'capsule', 'klacht', 'ouderwets', 'journaal',
    'dekken', 'ijzig', 'hoop', 'werk', 'gelijk', 'gratie', 'lichaam', 'simpel', 'aspect', 'officier', 'ruw', 'rector',
    'kraker', 'lip', 'opvatten', 'cijfer', 'verzoenen', 'bezweren', 'ingrijpen', 'god', 'passen', 'bewaren',
    'dynamisch', 'kaas', 'zilver', 'storing', 'reductie', 'installeren', 'verschuiving', 'charter', 'verlichting',
    'vergunning', 'hakken', 'oosten', 'vermenging', 'afvliegen', 'aandeel', 'opbrengen', 'glans', 'antwoord', 'persen',
    'inrichten', 'jas', 'slof', 'geraffineerd', 'zegen', 'lezing', 'bom', 'wol', 'vurig', 'lezer', 'steken', 'vliegen',
    'schotel', 'waaien', 'rem', 'plafond', 'herinneren', 'goederenvervoer', 'schenken', 'opwekken', 'opgemaakt',
    'stout', 'steunen', 'spiraal', 'firma', 'pool', 'haan', 'beschaving', 'verwijderen', 'begrip', 'parel', 'koek',
    'aankondigen', 'vervolging', 'vlak', 'formule', 'draak', 'resolutie', 'ontzetting', 'drukker', 'zebra',
    'voorganger', 'voorraad', 'plegen', 'maat', 'recht', 'ezel', 'bloei', 'naam', 'engagement', 'fluiten', 'vloedgolf',
    'versieren', 'shit', 'legeren', 'som', 'kunnen', 'zijtak', 'bewerken', 'klapper', 'fout', 'vertegenwoordiger',
    'vergoeden', 'oogst', 'nietig', 'terechtkomen', 'slagen', 'landschap', 'keren', 'bak', 'opzetten', 'brief',
    'ridder', 'toilet', 'heerlijk', 'boodschap', 'huiselijk', 'katalysator', 'parallel', 'slinger', 'braam', 'ogenblik',
    'zending', 'verlies', 'kandidaat', 'prijs', 'terugkomen', 'flop', 'brullen', 'afslag', 'master', 'campagne', 'bank',
    'paradijs', 'muis', 'hysterisch', 'uitgaan', 'voorzien', 'bieden', 'fortuin', 'uitgangspunt', 'bekleden', 'steun',
    'prooi', 'goot', 'boom', 'lichting', 'beschikbaar', 'pad', 'keus', 'tand', 'gewoonte', 'uithaal', 'veld',
    'manoeuvre', 'knippen', 'bibliotheek', 'enkel', 'straat', 'beperking', 'gewas', 'mes', 'president', 'karakter',
    'reduceren', 'neerslag', 'opvolger', 'terugvinden', 'schrik', 'kolf', 'uitkomen', 'tijdelijk', 'beslaan', 'zee',
    'vruchtbaar', 'beleggen', 'vermaken', 'plaatselijk', 'greep', 'zuigen', 'gebaar', 'mannelijk', 'tweeslachtig',
    'knap', 'poot', 'onthouden', 'wild', 'opera', 'naaien', 'lezen', 'verward', 'ervaring', 'toets', 'opvatting',
    'kristal', 'lam', 'doorgaan', 'terrein', 'buitenlands', 'onderhouden', 'nalaten', 'buurt', 'kalf', 'international',
    'weken', 'licht', 'omgeving', 'schaduw', 'schatting', 'onthouding', 'bespreking', 'referentie', 'krant',
    'verbranden', 'recept', 'rommel', 'oud', 'kennis', 'vergoeding', 'wegtrekken', 'ordening', 'lijvig', 'doordraaien',
    'koor', 'scoren', 'gloed', 'vervelen', 'effen', 'lijden', 'effect', 'richting', 'werkzaam', 'fatsoenlijk', 'vlot',
    'onzeker', 'letter', 'set', 'trots', 'luik', 'bestek', 'hyena', 'expertise', 'kappen', 'kolom', 'aal', 'forceren',
    'water', 'voetbal', 'situatie', 'eikel', 'waardering', 'stroef', 'kruis', 'sneeuwbal', 'statisch', 'geleding',
    'verlicht', 'reeks', 'woest', 'reserveren', 'bedienen', 'asiel', 'paprika', 'blank', 'broek', 'droom', 'rug',
    'trip', 'net', 'zegel', 'bewerking', 'roest', 'figuur', 'bezeten', 'inleiding', 'houding', 'harmonie', 'bord',
    'zin', 'credit', 'straf', 'tas', 'kapel', 'schrijver', 'prikkel', 'gerecht', 'leuk', 'matig', 'vest', 'meekomen',
    'ontmoeting', 'beluisteren', 'vreemdeling', 'bijstand', 'politiek', 'spelling', 'warmte', 'formatie', 'lastig',
    'supplement', 'wals', 'boog', 'bonzen', 'blad', 'heersen', 'wennen', 'adres', 'afleiden', 'groot', 'circus',
    'vertraging', 'berekend', 'temporeel', 'zuster', 'handel', 'compositie', 'avond', 'voordracht', 'gieren', 'lijst',
    'portret', 'markt', 'atmosfeer', 'speler', 'loper', 'pijp', 'ongemakkelijk', 'hals', 'verdraaid', 'aanmelden',
    'vloot', 'bloem', 'komen', 'morfologie', 'beugel', 'parket', 'vergaan', 'gehoor', 'vastzetten', 'hameren',
    'toegang', 'verbergen', 'kras', 'grof', 'vegen', 'substantie', 'keihard', 'hemel', 'smerig', 'correctie', 'ruimte',
    'gebied', 'bezetting', 'stug', 'berekenen', 'bed', 'kit', 'passeren', 'file', 'strook', 'Oranje', 'kever',
    'aanbieden', 'snel', 'verrassing', 'departement', 'uitvallen', 'rijden', 'verdikken', 'luchtig', 'offer', 'lector',
    'ruiter', 'baard', 'uitzoeken', 'uiten', 'belangstelling', 'applicatie', 'nieuw', 'aanhanger', 'industrie',
    'aflevering', 'verwerken', 'staal', 'aanslaan', 'proces', 'kunst', 'overgaan', 'vergeten', 'doek', 'schouw',
    'soort', 'wit', 'solidair', 'uitkijken', 'virus', 'stop', 'stemming', 'productie', 'integriteit', 'omslag', 'vlag',
    'top', 'forum', 'kapitaal', 'missen', 'podium', 'test', 'diep', 'hangen', 'vallen', 'stoet', 'omleggen', 'contract',
    'uitzending', 'schaap', 'poort', 'papier', 'kroon', 'gat', 'pijl', 'demonstratie', 'stoplicht', 'paar', 'aflopen',
    'beleg', 'inzien', 'verovering', 'eis', 'muzikant', 'uitdaging', 'redelijk', 'gunnen', 'opvoering', 'afloop',
    'ontwikkeld', 'bes', 'schilderen', 'concessie', 'gevoel', 'elektra', 'drank', 'staf', 'centraal', 'vraagstuk',
    'beeld', 'kalk', 'gebrek', 'been', 'aanvoeren', 'persoonlijkheid', 'afleggen', 'aarden', 'beschikking', 'dekking',
    'burger', 'plaats', 'doorrijden', 'slak', 'koppeling', 'onttrekken', 'kerk', 'gewest', 'teer', 'bevestiging',
    'studio', 'gel', 'wortel', 'kneep', 'warm', 'aanwijzen', 'ster', 'raken', 'fractie', 'toegankelijk', 'kring',
    'bewaking', 'lob', 'jager', 'afhangen', 'bezighouden', 'premie', 'treden', 'richtlijn', 'moer', 'stel', 'riem',
    'aandoen', 'unie', 'cilinder', 'spiegel', 'regie', 'akte', 'afnemen', 'zuchten', 'haken', 'trio', 'snoer', 'boezem',
    'meegaan', 'pers', 'samenwerkingsverband', 'apart', 'fors', 'duidelijk', 'onfeilbaar', 'kamp', 'opgaan', 'voelen',
    'belang', 'buis', 'criticus', 'schakel', 'vestigen', 'hoogte', 'zetel', 'aanvallen', 'enig', 'studie', 'beest',
    'meester', 'beschrijven', 'starter', 'overzicht', 'schuiven', 'ondergrond', 'handwerk', 'element', 'banaan', 'held',
    'amazone', 'afwerking', 'hap', 'klef', 'benauwd', 'hart', 'aanvaller', 'kader', 'geboorte', 'gezantschap',
    'landbouwgebied', 'opstaan', 'bedenken', 'afdraaien', 'ontslaan', 'administratie', 'schets', 'capaciteit', 'gewoon',
    'voeding', 'bolwerk', 'vangst', 'toer', 'legaat', 'apparaat', 'stijl', 'verzoek', 'zeker', 'dringen', 'aanslag',
    'duiden', 'man', 'gering', 'willen', 'patriarch', 'verbinden', 'herinnering', 'blind', 'moraal', 'nominaal',
    'aanval', 'trappen', 'wafel', 'aansluiting', 'neus', 'motief', 'hak', 'antenne', 'betrokken', 'hof', 'inlichting',
    'expansie', 'buigen', 'menigte', 'rationeel', 'statistiek', 'koesteren', 'riet', 'draad', 'flexibiliteit',
    'bevinden', 'bezetten', 'trap', 'vorm', 'leggen', 'alt', 'exclusief', 'allergisch', 'selectie', 'flink',
    'neerkomen', 'schakelen', 'tak', 'spel', 'pan', 'ongelijk', 'klappen', 'koepel', 'ministerie', 'standplaats',
    'vinden', 'zijn', 'zwaarte', 'krijgen', 'koffie', 'goddelijk', 'mogen', 'oppervlak', 'gal', 'populatie',
    'interieur', 'vijg', 'voet', 'arrest', 'lens', 'spruit', 'afzien', 'klomp', 'mysterie', 'rondgang', 'overheid',
    'mikken', 'correspondent', 'verpakking', 'vrouw', 'plat', 'overstap', 'verhoging', 'progressief', 'baas', 'label',
    'afzet', 'afname', 'migratie', 'blazen', 'jagen', 'aap', 'zorg', 'werpen', 'komma', 'kraan', 'actief', 'weergave',
    'branche', 'koper', 'verlichten', 'overgang', 'profiel', 'vierkant', 'toegeven', 'oor', 'memorandum', 'tuimelaar',
    'afbreken', 'hulpmiddel', 'achterlaten', 'bod', 'gevestigd', 'gunst', 'conditie', 'tempo', 'kameleon', 'ingaan',
    'heerlijkheid', 'vlees', 'eenzijdig', 'kop', 'uitspraak', 'bijzetten', 'heet', 'ontsteking', 'stabiliteit',
    'gevoelig', 'zijde', 'afmaken', 'kwast', 'voeren', 'dwarsligger', 'berg', 'pluim', 'boel', 'verplicht', 'moment',
    'beloning', 'autonomie', 'hulp', 'bedrukt', 'mild', 'hoofdstad', 'ziel', 'stemmen', 'slang', 'garde',
    'vaststelling', 'kopie', 'container', 'onzuiver', 'bepaling', 'heer', 'diensttijd', 'alleenstaand', 'aardig', 'tic',
    'bas', 'uitbrengen', 'stelling', 'techniek', 'puinhoop', 'redenering', 'negatief', 'jood', 'raad', 'bekomen',
    'winnen', 'pakket', 'intrede', 'kantoor', 'merk', 'gast', 'bediende', 'dromen', 'organisatie', 'bol', 'bestaan',
    'horde', 'kuil', 'contact', 'promotie', 'ontmoeten', 'hei', 'transcriptie', 'beweging', 'groeien', 'fonds', 'kabel',
    'glad', 'fixeren', 'ding', 'afzetten', 'balans', 'zomer', 'paleis', 'snede', 'kwalificatie', 'schreeuwen',
    'indicatie', 'dak', 'verdediging', 'stom', 'bedanken', 'juist', 'wapen', 'gemaal', 'pupil', 'hoedanigheid', 'knaap',
    'bepalen', 'mol', 'geus', 'aanzetten', 'poeder', 'rot', 'beginnen', 'fabel', 'tante', 'eend', 'kwartier',
    'achterlijk', 'installatie', 'stoten', 'ontbinding', 'raam', 'kloppen', 'bijhouden', 'schelp', 'gemeenschap',
    'onbeperkt', 'vastleggen', 'inslag', 'kwaliteit', 'stok', 'amplitude', 'geul', 'verkleden', 'hoofd', 'innemen',
    'baar', 'zanger', 'rede', 'school', 'meenemen', 'gevaar', 'onbeschaafd', 'aannemen', 'nevel', 'gezag', 'expressie',
    'bestemming', 'begeleiden', 'deelneming', 'verheugen', 'verplichten', 'toestand', 'document', 'duister', 'breuk',
    'single', 'inzet', 'woongebied', 'oppositie', 'strijken', 'gekleurd', 'lekker', 'inschieten', 'aas', 'hoek',
    'meervoud', 'fluit', 'onderscheiden', 'roep', 'geschiedenis', 'lei', 'schrikken', 'spoor', 'les', 'overweging',
    'uitstraling', 'mechanisch', 'cultuur', 'ruimen', 'scheidsrechter', 'zadel', 'bondgenoot', 'klem', 'uitloper',
    'verheffen', 'afstoten', 'besturen', 'commandant', 'uitsteken', 'openen', 'juk', 'voorstelling', 'peer', 'boor',
    'hout', 'affaire', 'drama', 'opname', 'ader', 'ondergeschikt', 'bestrijden', 'scherp', 'trekker', 'klaren',
    'boekhandel', 'aanbod', 'mijn', 'gezegde', 'haak', 'uitzien', 'uiterst', 'gelegenheid', 'krul', 'vliegend',
    'geleider', 'onrust', 'breken', 'uitvoering', 'uitzicht', 'registratie', 'voorkomen', 'liberaal', 'afrijden',
    'verzekering', 'legioen', 'kweken', 'klok', 'stof', 'belemmering', 'spelen', 'verslijten', 'omgooien', 'ongebonden',
    'verklaren', 'uitdrukking', 'omslaan', 'elektrisch', 'drijven', 'licentie', 'fris', 'plas', 'opvolgen', 'vriend',
    'instructie', 'volgen', 'symbool', 'inleiden', 'bestuur', 'rit', 'tekenen', 'bok', 'stip', 'stem', 'afwerken',
    'afvallen', 'feestdag', 'inspectie', 'teen', 'klinker', 'uittrappen', 'partner', 'horen', 'hol', 'titel',
    'gebieden', 'zaak', 'wringen', 'gastheer', 'muziek', 'aanstaand', 'danken', 'wereld', 'team', 'flat', 'lijn',
    'gemeenteraad', 'scheef', 'roepen', 'legger', 'economie', 'gewicht', 'plaatsen', 'veer', 'gezicht', 'aanleg',
    'coupe', 'geheugen', 'materiaal', 'rijm', 'verzamelen', 'reus', 'vervelend', 'rad', 'fantasie', 'theater', 'bezit',
    'toneel', 'leveren', 'instantie', 'tekst', 'groeperen', 'belasting', 'beantwoorden', 'krachtig', 'glas', 'put',
    'ronde', 'anker', 'wens', 'legende', 'bespreken', 'sturen', 'web', 'begin', 'kast', 'canon', 'pittig', 'akkoord',
    'pleidooi', 'pap', 'slapen', 'sector', 'uitwerking', 'voorwaarde', 'klinken', 'relatie', 'kans', 'bijdrage',
    'sectie', 'eenvoudig', 'exporteur', 'afschuiven', 'optreden', 'systeem', 'haar', 'pin', 'opheffen', 'zwart',
    'treffen', 'laken', 'inpakken', 'lot', 'vergadering', 'gunstig', 'onafhankelijkheid', 'hopen', 'verdediger', 'beet',
    'oordelen', 'singel', 'impuls', 'jongen', 'behouden', 'strik', 'maag', 'knijpen', 'sluiting', 'hal', 'beschouwen',
    'toon', 'kuif', 'opzet', 'insluiten', 'agressief', 'vrede', 'tank', 'glashelder', 'ophouden', 'kokend', 'flipper',
    'verwachting', 'spook', 'aandrijving', 'bekennen', 'werking', 'universiteit', 'verplaatsen', 'masker', 'groei',
    'afbraak', 'schuld', 'leeg', 'onderdruk', 'ontwikkelen', 'grap', 'overnemen', 'ensemble', 'inspelen', 'denken',
    'kleurrijk', 'vent', 'macht', 'beker', 'zien', 'uitzetten', 'bies', 'scherm', 'paneel', 'appel', 'feest', 'ketel',
    'film', 'wal', 'branden', 'doortrappen', 'benaderen', 'schilder', 'flexibel', 'uitgever', 'strak', 'doorzetten',
    'leider', 'krassen', 'concurrentie', 'plaag', 'identificatie', 'intrekken', 'uitpakken', 'koersen', 'keuken',
    'visioen', 'bron', 'uitslag', 'onderhoud', 'invallen', 'waarde', 'vlezig', 'aanzien', 'beer', 'genot', 'vak',
    'nuchter', 'arm', 'schrijven', 'pension', 'sterven', 'dek', 'televisie', 'object', 'vraag', 'leerling', 'gom',
    'uitkering', 'wassen', 'kreet', 'penning', 'pink', 'configuratie', 'vaart', 'nicht', 'koppel', 'kamer', 'gestel',
    'ram', 'doorslaan', 'doorlopen', 'overloop', 'tellen', 'absoluut', 'secretariaat', 'grip', 'omwenteling',
    'afscheiden', 'commissaris', 'regelaar', 'kijker', 'romantisch', 'kist', 'maken', 'zacht', 'graan', 'doel', 'haas',
    'doorsteken', 'ellendig', 'uitschrijven', 'kolonie', 'bond', 'complex', 'uitbreken', 'gezondheidszorg', 'plaatsing',
    'storten', 'secretaris', 'bestellen', 'vissen', 'band', 'trouwen', 'rustplaats', 'aansluiten', 'oplossen', 'vies',
    'optrekken', 'transmissie', 'kloof', 'bescheiden', 'rollen', 'goud', 'dagen', 'polair', 'interpretatie', 'klasse',
    'plak', 'reiniging', 'adjudant', 'vreedzaam', 'brigade', 'bereik', 'overlaten', 'start', 'opmaak', 'vestiging',
    'kraai', 'achterstand', 'beroepen', 'onafhankelijk', 'bureau', 'vet', 'ijs', 'ondersteunen', 'waarheid', 'rust',
    'publicatie', 'misdaad', 'sneeuw', 'bon', 'knop', 'karton', 'monitor', 'staart', 'redactie', 'haard',
    'vrijwilliger', 'rukken', 'spin', 'functie', 'rijzen', 'concentreren', 'break', 'reep', 'automaat', 'verwant',
    'steek', 'doorbreken', 'kracht', 'voetstap', 'schoon', 'drempel', 'afglijden', 'weg', 'ouderdom', 'aantrekken',
    'opereren', 'onnatuurlijk', 'koning', 'verwarring', 'binding', 'woord', 'spoorweg', 'broos', 'korrel', 'opvangen',
    'vrijlaten', 'stap', 'zwaar', 'proef', 'stengel', 'natuurlijk', 'run', 'speculatie', 'bezuiniging', 'verspreiden',
    'zuiver', 'onregelmatigheid', 'bariton', 'handhaven', 'overtuiging', 'erkenning', 'pot', 'wijs', 'proeven', 'staat',
    'aansteken', 'college', 'onzijdig', 'verstoppen', 'zolder', 'huis', 'mager', 'slaan', 'hoog', 'dienst',
    'oppervlakte', 'onzin', 'weren', 'wet', 'abnormaal', 'luiden', 'bezwaar', 'omtrek', 'gereedschap', 'aanleggen',
    'ei', 'portefeuille', 'vriendin', 'laag', 'verbranding', 'klimaat', 'verkeer', 'fax', 'nood', 'weerstand', 'bout',
    'basis', 'trommel', 'doorbraak', 'werelds', 'onderneming', 'nakomen', 'toog', 'zak', 'afscheiding', 'compagnie',
    'wijzen', 'bestelling', 'lied', 'advocaat', 'vrouwelijk', 'geven', 'bezwijken', 'ernstig', 'opdracht', 'lepel',
    'wending', 'zuur', 'vermenigvuldigen', 'strip', 'club', 'automatisch', 'vatten', 'vangen', 'sleutel', 'achten',
    'verblijf', 'kweek', 'transfer', 'opstelling', 'commando', 'spreiden', 'hard', 'ellende', 'ongeluk', 'middel',
    'inhouden', 'betrekken', 'management', 'paal', 'verval', 'makkelijk', 'herstellen', 'lekken', 'gewijd',
    'vooruitzicht', 'bek', 'wippen', 'geheim', 'heks', 'konvooi', 'vorst', 'kegel', 'jacht', 'bouw', 'variatie', 'pulp',
    'doodstil', 'afzakken', 'links', 'eer', 'discipline', 'eind', 'onbekwaam', 'barrière', 'rib', 'gamma', 'behandelen',
    'locatie', 'verkleining', 'betekenis', 'aanhangsel', 'kalender', 'best', 'punk', 'verstaan', 'uitdrukken', 'kou',
    'wacht', 'activiteit', 'notitie', 'zuivering', 'gebruiken', 'handelen', 'plaat', 'welzijn', 'job', 'verkeren',
    'begrijpen', 'migrant', 'bende', 'bot', 'maandag', 'oordeel', 'kapot', 'verlossing', 'inhalen', 'roken',
    'aanpassen', 'herhalen', 'filter', 'kruisen', 'vlies', 'afgeven', 'controleren', 'aanhouden', 'swing', 'proper',
    'verdeling', 'regelgeving', 'idee', 'grafiek', 'richten', 'werf', 'mal', 'opslaan', 'overschrijving', 'boren',
    'lief', 'brommen', 'huishoudelijk', 'projectie', 'overjarig'
]


class WiCNL(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.2.2")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "lemma": datasets.Value("string"),
                "tokens1": datasets.Sequence(datasets.Value("string")),
                "tokens2": datasets.Sequence(datasets.Value("string")),
                "index1": datasets.Value("int32"),
                "index2": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=["different", "same"]),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dataset = datasets.load_dataset("hf_datasets/dutch-sem-cor", data_dir=dl_manager.manual_dir, split="all")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "dataset": dataset,
                    "lemmas": _TRAIN_LEMMAS,
                    "lemma_limit": 6,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "dataset": dataset,
                    "lemmas": _VALID_LEMMAS,
                    "lemma_limit": 4,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # type: ignore
                gen_kwargs={
                    "dataset": dataset,
                    "lemmas": _TEST_LEMMAS,
                    "lemma_limit": 4,
                },
            ),
        ]

    def _generate_examples(self, dataset, lemmas, lemma_limit):
        pos_senses = set()
        neg_senses = set()
        n_pos, n_neg = 0, 0
        lemma_counts = {}
        sense_dict = {}

        key = 0
        for ex1 in dataset:
            lemma = ex1["lemma"]
            sense1 = ex1["sense"]

            # lemma not whitelisted for this split
            if lemma not in lemmas:
                continue

            # extremely long sentences
            if len(ex1["tokens"]) > 60:
                continue

            # lemma used maximum number of times
            if lemma_counts.get(lemma, 0) >= lemma_limit:
                continue

            # every sense once positive and once negative
            if sense1 in pos_senses and sense1 in neg_senses:
                continue

            if lemma not in sense_dict:
                sense_dict[lemma] = {}
            if sense1 not in sense_dict[lemma]:
                sense_dict[lemma][sense1] = []

            # use as positive example
            if n_pos < n_neg and sense1 not in pos_senses and len(sense_dict[lemma][sense1]) > 0:
                ex2 = sense_dict[lemma][sense1].pop()
                if len(sense_dict[lemma][sense1]) == 0:
                    sense_dict[lemma].pop(sense1)
                yield key, {
                    "lemma": ex1["lemma"],
                    "index1": ex1["index"],
                    "index2": ex2["index"],
                    "tokens1": ex1["tokens"],
                    "tokens2": ex2["tokens"],
                    "label": "same",
                }
                key += 1
                lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
                pos_senses.add(sense1)
                n_pos += 1
                continue

            # use as negative example
            if sense1 not in neg_senses and len(sense_dict[lemma]) > 1:
                sense2 = list(sense_dict[lemma].keys() - {sense1})[0]
                ex2 = sense_dict[lemma][sense2].pop()
                if len(sense_dict[lemma][sense1]) == 0:
                    sense_dict[lemma].pop(sense1)
                if len(sense_dict[lemma][sense2]) == 0:
                    sense_dict[lemma].pop(sense2)
                yield key, {
                    "lemma": lemma,
                    "index1": ex1["index"],
                    "index2": ex2["index"],
                    "tokens1": ex1["tokens"],
                    "tokens2": ex2["tokens"],
                    "label": "different",
                }
                key += 1
                lemma_counts[lemma] = lemma_counts.get(lemma, 0) + 1
                neg_senses.add(sense1)
                neg_senses.add(sense2)
                n_neg += 1
                continue

            sense_dict[lemma][sense1].append(ex1)
