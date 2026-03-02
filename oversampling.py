import os
import numpy as np
import datasets

def get_overall_counts(ds, label_lst):
    # initialize values
    total_counts = {label: {"spans":0,"tokens":0} for label in label_lst}
    total_spans = 0
    total_labeled_tokens = 0
    # get counts for label spans and tokens
    for row in ds:
        for label in label_lst:
            spans = row[f"labels_{label}"].count("B")
            innies = row[f"labels_{label}"].count("I")
            total_counts[label]["spans"] += spans
            total_counts[label]["tokens"] += spans+innies
            total_spans+= spans
            total_labeled_tokens+=spans+innies
    total_counts["Overall"] = {"spans":total_spans,"tokens":total_labeled_tokens}
    for label in list(total_counts):
        print(f"{label}: {round((total_counts[label]['spans']/total_spans)*100,2)}% all spans ({total_counts[label]['spans']})")
    return total_counts

def article_label_coverage(ds, label_lst, total_counts):
    tracking = {}
    for row in ds:
        tracking[row['id']] = {label: {"spans":0,"tokens":0} for label in label_lst}
        for label in label_lst:
            spans = row[f"labels_{label}"].count("B")
            innies = row[f"labels_{label}"].count("I")
            tracking[row['id']][label]["spans"] = spans
            tracking[row['id']][label]["tokens"] = spans+innies
            #
            tracking[row['id']][label]["span_pct"] = round((spans/total_counts[label]['spans'])*100,3)
            tracking[row['id']][label]["token_pct"] = round(((spans+innies)/total_counts[label]['tokens'])*100,3)
    return tracking

def get_art_lists_for_lbls(art_lbl_tracking, label_lst):
    arts_of_occurrence = {label:[] for label in label_lst}
    for artid in list(art_lbl_tracking):
        for label in label_lst:
            spct = art_lbl_tracking[artid][label]["span_pct"]
            if spct>0:
                arts_of_occurrence[label].append(artid)
    return arts_of_occurrence

def compare_art_lbl_occ(arts_of_occurrence, quant=0.333):
    lbl_arts_occ = []
    min_obj = (None,100)
    max_obj = (None,0)
    for label in list(arts_of_occurrence):
        val = round((len(arts_of_occurrence[label])/412)*100,2)
        lbl_arts_occ.append((label,val))
        #print(f"{label}: {val}")
        # get min and max label names
        if val<min_obj[1]:
            min_obj = (label,val)
        if val>max_obj[1]:
            max_obj = (label,val)
    # get threshold by finding quantile value
    vals = [entry[1] for entry in lbl_arts_occ]
    thresh = np.quantile(vals, quant)
    # determine minority labels
    minority_labels = []
    for entry in lbl_arts_occ:
        if entry[1]<thresh:
            minority_labels.append(entry[0])
    min_label = min_obj[0]
    max_label = max_obj[0]
    arts_of_interest = []
    for label in minority_labels:
        arts_of_interest.extend(arts_of_occurrence[label])
    arts_of_interest = list(set(arts_of_interest))
    print("Minority Labels:",minority_labels)
    return arts_of_interest, minority_labels, min_label, max_label

def compare_overall_lbl_occ(total_counts,arts_of_occurrence, quant=0.333):
    lbl_arts_occ = []
    min_obj = (None,100)
    max_obj = (None,0)
    for label in list(total_counts):
        val = round((total_counts[label]['spans']/total_counts['Overall']['spans'])*100,2)
        lbl_arts_occ.append((label,val))
        if val<min_obj[1]:
            min_obj = (label,val)
        if val>max_obj[1]:
            max_obj = (label,val)
    vals = [entry[1] for entry in lbl_arts_occ]
    thresh = np.quantile(vals, quant)
    minority_labels = []
    for entry in lbl_arts_occ:
        if entry[1]<thresh:
            minority_labels.append(entry[0])
    min_label = min_obj[0]
    max_label = max_obj[0]
    arts_of_interest = []
    for label in minority_labels:
        arts_of_interest.extend(arts_of_occurrence[label])
    arts_of_interest = list(set(arts_of_interest))
    print("Minority Labels:",minority_labels)
    return arts_of_interest, minority_labels, min_label, max_label

def refine_arts_of_int(method, art_lbl_tracking, arts_of_interest, minority_labels, min_label, max_label):
    aoi_dict = dict()
    for key in arts_of_interest:
        aoi_dict[key] = art_lbl_tracking[key]
    if method=="minority":# sort by lowest minority label
        arts_inorder = sorted(aoi_dict, key=lambda x:aoi_dict[x][min_label]['span_pct'], reverse=True)
        #aoi_dict_sorted = {key: aoi_dict[key] for key in arts_inorder}
        overs_arts = arts_inorder[:int(len(arts_inorder)/2)]
    elif method=="dscmin+ascmax":
        arts_dscmin = sorted(aoi_dict, key=lambda x:aoi_dict[x][min_label]['span_pct'], reverse=True)
        arts_ascmax = sorted(aoi_dict, key=lambda x:aoi_dict[x][max_label]['span_pct'], reverse=False)
        #aoi_dict_dscmin = {key: aoi_dict[key] for key in arts_dscmin}
        #aoi_dict_ascmax = {key: aoi_dict[key] for key in arts_ascmax}
        # may need to find a way of reducing the number of articles dynamically?
        # for now just take the first half
        overs_dscmin = arts_dscmin[:int(len(arts_dscmin)/2)]
        overs_ascmax = arts_ascmax[:int(len(arts_ascmax)/2)]
        overs_arts = list(set(overs_dscmin+overs_ascmax))
    elif method=="minorities":
        tple_coll = {label:[] for label in minority_labels}
        art_coll = {label:[] for label in minority_labels}
        overs_arts = []
        for label in minority_labels:
            tplelst = sorted(aoi_dict, key=lambda x:aoi_dict[x][label]['span_pct'], reverse=True)
            tple_coll[label] = tplelst
            art_coll[label] = tplelst[:int(len(tplelst)/len(minority_labels))]
            overs_arts.extend(art_coll[label])
        overs_arts = list(set(overs_arts))
    return overs_arts

def create_new_osds(ds, overs_arts):
    osds = ds.filter(lambda example: example["id"] in overs_arts)
    ovs_ds = datasets.concatenate_datasets([ds, osds])
    return ovs_ds

def oversample_ds(ds, label_lst):
    print("\nOld:")
    total_counts = get_overall_counts(ds, label_lst)
    art_lbl_tracking = article_label_coverage(ds, label_lst, total_counts)
    arts_of_occurrence = get_art_lists_for_lbls(art_lbl_tracking, label_lst)
    #arts_of_interest, minority_labels, min_label, max_label = compare_art_lbl_occ(arts_of_occurrence, quant=0.3)
    arts_of_interest, minority_labels, min_label, max_label = compare_overall_lbl_occ(total_counts,arts_of_occurrence, quant=0.333)
    overs_arts = refine_arts_of_int("minorities", art_lbl_tracking, arts_of_interest, minority_labels, min_label, max_label)
    osds = create_new_osds(ds, overs_arts)
    print("\nNew:")
    new_counts = get_overall_counts(osds, label_lst)
    return osds

def main():
    cwd = os.getcwd()
    letter = "a"
    htype = "mhead"    

if __name__=="__main__":
    main()