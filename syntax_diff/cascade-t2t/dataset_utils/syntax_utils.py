from nltk.stem.wordnet import WordNetLemmatizer
import spacy

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]

def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs

def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs

def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs

def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None

def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs

def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos

ACOMPS = ['acomp', 'oprd']

def getAcompFromConjunctions(acomps, seen=None):
    if seen is None:
        seen = set() 

    moreAcomps = []
    for acomp in acomps:
        if acomp in seen:
            continue
        seen.add(acomp)

        # rights is a generator
        rights = list(acomp.rights)
        rightDeps = {tok.lower_ for tok in rights}

        if "and" in rightDeps:
            new_acomps = [tok for tok in rights if tok.dep_ in ACOMPS or tok.pos_ == "ADJ"]
            new_acomps = [tok for tok in new_acomps if tok not in seen]  # 只处理新词

            moreAcomps.extend(new_acomps)
            if len(new_acomps) > 0:
                moreAcomps.extend(getAcompFromConjunctions(new_acomps, seen))
    return moreAcomps

def getAllAcomps(v):
    rights = list(v.rights)
    acomps = [tok for tok in rights if tok.dep_ in ACOMPS]

    seen = set(acomps)
    new_acomps = getAcompFromConjunctions(acomps, seen)
    acomps.extend(new_acomps)

    return v, acomps

def findSVAs(tokens):
    svas = []
    for v in tokens:
        if v.dep_ in ["ROOT", "ccomp"]:
            subs, verbNegated = getAllSubs(v)
            if len(subs) > 0:
                v, acomps = getAllAcomps(v)
                for sub in subs:
                    for acomp in acomps:
                        acompNegated = isNegated(acomp)
                        svas.append((sub.lower_, "!" + v.lower_ if verbNegated or acompNegated else v.lower_, acomp.lower_))
    return svas

def findSVs(tokens):
    svs = []
    for v in tokens:
        if v.dep_ in ["ROOT", "ccomp"]:
            subs, verbNegated = getAllSubs(v)
            if len(subs) > 0:
                for sub in subs:
                    svs.append((sub.lower_, "!" + v.lower_ if verbNegated else v.lower_))
    return svs

def getPass(v):
    # lefts is a generator
    lefts = list(v.lefts)
    PassAux = [tok for tok in lefts if tok.dep_ == 'auxpass']
    return v, PassAux

def findPSVs(tokens):
    psvs = []
    for v in tokens:
        if v.dep_ in ["ROOT", "ccomp"]:
            subs, verbNegated = getAllSubs(v)
            if len(subs) > 0:
                v, PassAuxs = getPass(v)
                for sub in subs:
                    for PassAux in PassAuxs:
                        psvs.append((sub.lower_, PassAux.lower_, "!" + v.lower_ if verbNegated else v.lower_,))
    return psvs

def findMainStruct(tokens):
    verbs = [tok for tok in tokens if tok.dep_ in ["ROOT", "ccomp"]]
    mss = []
    for v in verbs:
        if v.dep_ in ["ROOT"]:
            verbNegated = isNegated(v)
            lefts = list(v.lefts)
            beforeRoots = [tok for tok in lefts if tok.dep_ != 'punct']
            for beforeRoot in beforeRoots:
                mss.append((beforeRoot.lower_, "!" + v.lower_ if verbNegated else v.lower_,))
    return mss

def findStruct(token):
    all_structs = findSVOs(token)
    svas = findSVAs(token)
    psvs = findPSVs(token)

    words_in_svos = {word for triplet in all_structs for word in triplet}
    for triplet in svas:
        any_common_word = any(word in words_in_svos for word in triplet)
        if not any_common_word:
            all_structs.append(triplet)
    for triplet in psvs:
        any_common_word = any(word in words_in_svos for word in triplet)
        if not any_common_word:
            all_structs.append(triplet)

    svs = findSVs(token)
    words_in_all_structs = {word for triplet in all_structs for word in triplet}
    for triplet in svs:
        any_common_word = any(word in words_in_all_structs for word in triplet)
        if not any_common_word:
            all_structs.append(triplet)

    if all_structs == []:
        mss = findMainStruct(token)
        all_structs.extend(mss)

    return ' '.join([' '.join(struct) for struct in all_structs])

def create_tokenizer(path):
    from transformers import PreTrainedTokenizerFast
    
    logging.info(f"Loading tokenizer from {path}/syntax-pos.json")
    file_path = f"{str(pathlib.Path(path))}/syntax-pos.json"

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=file_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        padding_side="right",
    )

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    return tokenizer


def train_syntax_tokenizer(
    path='./datasets/spacy_pos.txt',
    vocab_size=100,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
):  
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import NFD
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import WordLevelTrainer

    # Deal with tokens not appearing in vocab
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    # NFD: unicode, Lowercase: all to lowercase, StripAccents: remove the primes beyond letters
    tokenizer.normalizer = normalizers.Sequence([NFD()])
    # Separate numerical digits; whitespace will be used to separate
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
    # Each sentence processed with the ids for <s> and </s>
    tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])

    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=[path], trainer=trainer)
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.save(f"{str(pathlib.Path(path).parent)}/syntax-pos.json")