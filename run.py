import warnings

from snorkel.analysis import metric_score
from snorkel.labeling import labeling_function
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelModel
from snorkel.utils import probs_to_preds


from model import get_model, get_feature_arrays
from preprocessors import get_person_text, get_left_tokens, get_persons_last_name, last_name
from utils import load_data, load_dbpedia


POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1


# Heuristic Relationship Extraction

SPOUSES_DICT = {"spouse", "wife", "husband", "ex-wife", "ex-husband"}
FAMILY_DICT = {"father", "mother", "sister", "brother", "son", "daughter",
               "grandfather", "grandmother", "uncle", "aunt", "cousin"}
INLAWS = FAMILY_DICT.union({f + "in-law" for f in FAMILY_DICT})
OTHER = {"boyfriend", "girlfriend", "boss", "employee", "secretary", "co-worker"}
DBPEDIA = load_dbpedia()
LAST_NAMES = set([(last_name(x), last_name(y))
                  for x, y in DBPEDIA
                  if last_name(x) and last_name(y)])


@labeling_function(resources=dict(spouses=SPOUSES_DICT))
def lf_husband_wife(x, spouses):
    return POSITIVE if len(spouses.intersection(set(x.between_tokens))) > 0 else ABSTAIN


@labeling_function(resources=dict(spouses=SPOUSES_DICT), pre=[get_left_tokens])
def lf_husband_wife_left_window(x, spouses):
    if len(set(spouses).intersection(set(x.person1_left_tokens))) > 0:
        return POSITIVE
    if len(set(spouses).intersection(set(x.person2_left_tokens))) > 0:
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function(pre=[get_persons_last_name])
def lf_same_last_name(x):
    p1_last, p2_last = x.person_lastnames
    if p1_last and p2_last and p1_last == p2_last:
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function()
def lf_married(x):
    return POSITIVE if "married" in x.between_tokens else ABSTAIN


@labeling_function(resources=dict(family=INLAWS))
def lf_familial_relationship(x, family):
    return NEGATIVE if len(family.intersection(set(x.between_tokens))) > 0 else ABSTAIN


@labeling_function(resources=dict(family=INLAWS), pre=[get_left_tokens])
def lf_family_left_window(x, family):
    if len(set(family).intersection(set(x.person1_left_tokens))) > 0:
        return NEGATIVE
    if len(set(family).intersection(set(x.person2_left_tokens))) > 0:
        return NEGATIVE
    else:
        return ABSTAIN


@labeling_function(resources=dict(other=OTHER))
def lf_other_relationship(x, other):
    return NEGATIVE if len(other.intersection(set(x.between_tokens))) > 0 else ABSTAIN


# Distant Supervision Functions

@labeling_function(resources=dict(known_spouses=DBPEDIA), pre=[get_person_text])
def lf_distant_supervision(x, known_spouses):
    p1,  p2 = x.person_names
    if (p1, p2) in known_spouses or (p2, p1) in known_spouses:
        return POSITIVE
    else:
        return ABSTAIN


@labeling_function(resources=dict(last_names=LAST_NAMES), pre=[get_persons_last_name])
def lf_distant_supervision_last_names(x, last_names):
    p1_last, p2_last = x.person_lastnames
    if (p1_last != p2_last) and ((p1_last, p2_last) in last_names or (p2_last, p1_last) in last_names):
        return POSITIVE
    else:
        return ABSTAIN


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ((df_dev, Y_dev), df_train, (df_test, Y_test)) = load_data()
    lfs = [lf_husband_wife, lf_husband_wife_left_window, lf_same_last_name,
           lf_married, lf_familial_relationship, lf_family_left_window,
           lf_other_relationship, lf_distant_supervision, lf_distant_supervision_last_names]
    applier = PandasLFApplier(lfs)
    L_dev = applier.apply(df_dev)
    L_train = applier.apply(df_train)
    print(LFAnalysis(L_dev, lfs).lf_summary(Y_dev))
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, Y_dev, n_epochs=5000, log_freq=500, seed=12345)
    probs_dev = label_model.predict_proba(L_dev)
    preds_dev = probs_to_preds(probs_dev)
    print("Label model F1: {f}".format(f=metric_score(Y_dev, preds_dev, probs=probs_dev, metric='f1')))
    print("Label model AUC: {f}".format(f=metric_score(Y_dev, preds_dev, probs=probs_dev, metric='roc_auc')))
    probs_train = label_model.predict_proba(L_train)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df_train, y=probs_train, L=L_train)
    X_train = get_feature_arrays(df_train_filtered)
    model = get_model()
    batch_size = 64
    model.fit(X_train, probs_train_filtered, batch_size=batch_size, epochs=100)
    X_test = get_feature_arrays(df_test)
    probs_test = model.predict(X_test)
    preds_test = probs_to_preds(probs_test)
    print("Label model F1: {f}".format(f=metric_score(Y_test, preds_test, probs=probs_test, metric='f1')))
    print("Label model AUC: {f}".format(f=metric_score(Y_test, preds_test, probs=probs_test, metric='roc_auc')))

