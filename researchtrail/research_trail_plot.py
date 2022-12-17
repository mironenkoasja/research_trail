import pandas as pd
from itertools import groupby
import os
import math
import numpy as np
import re
import scipy.sparse
import Levenshtein as plev
import tkinter as tk
import copy
from pathlib import Path


PROJECT_DIR = Path(__file__).parent

pd.set_option('display.max_colwidth', 100)



###########################

list_of_options_info = ['index', 'Publication Type', 'Authors', 'Book Authors', 'Book Editors', 'Book Group Authors', 'Author Full Names',
     'Book Author Full Names', 'Group Authors', 'Article Title', 'Source Title', 'Book Series Title',
     'Book Series Subtitle', 'Language', 'Document Type', 'Conference Title', 'Conference Date', 'Conference Location',
    'Conference Sponsor', 'Conference Host', 'Author Keywords', 'Keywords Plus', 'Abstract', 'Addresses', 'C3',
     'Reprint Addresses', 'Email Addresses', 'Researcher Ids','ORCID', 'Funding Orgs', 'FP', 'Funding Text',
     'Cited References', 'Cited Reference Count', 'Times Cited, WoS Core', 'Times Cited, All Databases',
     '180 Day Usage Count', 'Since 2013 Usage Count', 'Publisher', 'Publisher City', 'Publisher Address', 'ISSN', 'eISSN',
     'ISBN', 'Journal Abbreviation', 'Journal Information', 'Publication Date', 'Publication Year', 'Volume', 'Issue', 'PN',
     'SI', 'MA', 'First Page', 'Last Page', 'Research Areas', 'DOI', 'D2', 'EA', 'Number of Pages', 'WoS Categories', 'WE',
     'SC', 'GA', 'Pubmed Id', 'OA','Highly Cited Status', 'HP', 'DA', 'UT', 'PD2', 'TCg', 'Cluster']

art_df_colnames = ['index', 'PT', 'AU', 'BA', 'BE', 'GP', 'AF', 'BF', 'CA', 'TI', 'SO',
                   'SE', 'BS', 'LA', 'DT', 'CT', 'CY', 'CL', 'SP', 'HO', 'DE', 'ID', 'AB',
                   'C1', 'C3', 'RP', 'EM', 'RI', 'OI', 'FU', 'FP', 'FX', 'CR', 'NR', 'TC',
                   'Z9', 'U1', 'U2', 'PU', 'PI', 'PA', 'SN', 'EI', 'BN', 'J9', 'JI', 'PD',
                   'PY', 'VL', 'IS', 'PN', 'SU', 'SI', 'MA', 'BP', 'EP', 'AR', 'DI', 'D2',
                   'EA', 'PG', 'WC', 'WE', 'SC', 'GA', 'PM', 'OA', 'HC', 'HP', 'DA', 'UT',
                   'PD2', 'TCg', 'Conn_Comp']



# This function trancform data frame preparing for plot
def create_citation_group(articles_df):
    articles_df['TC'] = articles_df['TC'].astype('float')
    articles_df['TCg'] = 0
    q1 = articles_df.TC.quantile(0.25)
    q2 = articles_df.TC.quantile(0.5)
    q3 = articles_df.TC.quantile(0.75)
    q4 = articles_df.TC.quantile(1)

    articles_df.loc[articles_df.TC <= q1, 'TCg'] = 20
    articles_df.loc[(q1 < articles_df.TC)&(articles_df.TC <= q2), 'TCg'] = 60
    articles_df.loc[(q2 < articles_df.TC)&(articles_df.TC <= q3), 'TCg'] = 130
    articles_df.loc[(q3 < articles_df.TC)&(articles_df.TC <= q4), 'TCg'] = 350

    print('Citation groups are created')
    return articles_df

def transform_PD(articles_df):
    new_pd = []
    for index, row in articles_df.iterrows():
        if row['PD'] !=row['PD']:
            new_pd.append('JAN 1')

        elif re.match(r'[A-Z]{3}-[A-Z]{3}', row['PD']):
            new_pd.append(' '.join([row['PD'].split('-')[0], '1']))
        elif re.match(r'[A-Z]{3}$', row['PD']):
            new_pd.append(' '.join([row['PD'].split('-')[0], '1']))
        else:
            new_pd.append(row['PD'])
    articles_df['PD'] = new_pd

    articles_df['PD2'] = articles_df['PD'] + ' ' + articles_df['PY'].astype('string')
    articles_df['PD2'] = pd.to_datetime(articles_df['PD2'])
    articles_df = articles_df.sort_values(by=['PD2'], ascending=False, ignore_index=True)
    articles_df = create_citation_group(articles_df)

    print('Dates are transformed')

    return articles_df

# This function get references from articles
def get_references_from_art(articles_df):

    references = []
    ref_id = 0
    for index, row in articles_df.iterrows():
        if row['CR'] == row['CR']:
            ref = row['CR'].split('; ')
            for r in ref:
                r = r.replace('[DOI', 'DOI')
                r = r.replace('DOI [', 'DOI')
                r = r.replace('DOI DOI', 'DOI')
                if '[No title captured' in r:
                    print('no title')
                elif'**DROPPED REF' in r:
                    print('dropped ref')
                else:
                    references.append([index, ref_id, r])
                    ref_id+=1

    ref_pattern = re.compile(r'(?P<name>([^;,]*)|()|(\[Anonymous\])),? ?(?P<year>([0-9^;][0-9^;][0-9^;][0-9^;])|()),? ?(?P<title>([^,^;]+)|()),? ?(?P<issue>((V|v)[0-9^;]+)|()),? ?(?P<page>((P|p)[IiXxVv0-9^;]+)|()),? ?(?P<doi>(DOI .+)|())')

    dic ={'name':[], 'year':[], 'title':[], 'issue':[], 'page':[], 'doi':[], 'string':[], 'art_id':[], 'ref_id': []}
    for refs in references:
        dic['name'].append(ref_pattern.match(refs[2])['name'].split(' ')[0].upper())
        dic['year'].append(ref_pattern.match(refs[2])['year'])
        dic['title'].append(ref_pattern.match(refs[2])['title'])
        dic['issue'].append(ref_pattern.match(refs[2])['issue'])
        dic['page'].append(ref_pattern.match(refs[2])['page'])
        dic['doi'].append(ref_pattern.match(refs[2])['doi'])
        dic['string'].append(refs[2])
        dic['art_id'].append(refs[0])
        dic['ref_id'].append(refs[1])

    print('References are parsed')

    return dic

# This function compares two values and return answer: same(2), diff(3), 1 none(1), both none(0)
def comp (dict1, dict2, typ_comp='direct'):
    result = 0
    if typ_comp=='direct':
        if (dict1 != '')+(dict2 != '') == 1:
            result = 1
        elif (dict1 != '')+(dict2 != '') == 0:
            result = 0
        elif (dict1 != '')+(dict2 != '') == 2:
            if dict1 == dict2:
                result = 2
            else:
                result = 3
    else:
        dist1 = plev.distance(dict1, dict2)
        max_dict = len(max(dict1, dict2))
        dist = ((max_dict - dist1)*2)/ (len(dict1)+len(dict2))
        if dist == 1:
            result = 2
        elif 1 > dist >= 0.70:
            result = 4
        else:
            result = 3
    return result

def create_pattern1_2(dic, i, j):
    pati = '-'.join([dic['name'][i].split(' ')[0].lower(),dic['year'][i], dic['title'][i], dic['issue'][i], dic['page'][i], dic['doi'][i]])
    patj = '-'.join([dic['name'][j].split(' ')[0].lower(), dic['year'][j], dic['title'][j], dic['issue'][j], dic['page'][j], dic['doi'][j]])
    return pati, patj

# This function check paar of ref in dictionary
def check_dictionary(ref_dic, i, j):
    dict_name = PROJECT_DIR+'/data/all_dict.csv'
    store_ref = pd.read_csv(dict_name, sep='\t', header=None) # TODO Universalname of dictionary
    store_ref_l = list(store_ref[0])
    ini = {'pat':'', 'id':-1}
    inj = {'pat':'', 'id':-1}
    ini['pat'], inj['pat'] = create_pattern1_2(ref_dic, i, j)

    for k in range(0, len(store_ref_l)):
        if ini['pat'] in store_ref_l[k]:
            ini['id'] = k
        if inj['pat'] in store_ref_l[k]:
            inj['id'] = k

    res = (inj['id'] == -1) + (ini['id'] == -1)
    if res == 0:
        if ini['pat'] == inj['pat']:
            final_res = 1
        else:
            final_res = 0
    elif res == 1:
        final_res = call_check_window(ref_dic, i, j)
        if final_res == 1:
            if ini['id'] == -1:
                store_ref_l[inj['id']] = ','.join([store_ref_l[inj['id']],  ini['pat']])
                store_ref = pd.DataFrame(store_ref_l)
                store_ref.to_csv(dict_name, index=False)
            else:
                store_ref_l[ini['id']] = ','.join([store_ref_l[ini['id']],  inj['pat']])
                store_ref = pd.DataFrame(store_ref_l)
                store_ref.to_csv(dict_name, index=False)
    elif res == 2:
        final_res = call_check_window(ref_dic, i, j)
        if final_res == 1:
            store_ref_l.append(','.join([ini['pat'],  inj['pat']]))
            store_ref = pd.DataFrame(store_ref_l)
            store_ref.to_csv(dict_name, index=False)
    return final_res

# This function  create a window for manual comparison of references
def call_check_window(dic, i, j):
    window = tk.Tk()
    window.title('Compare these bibliographic records')
    window.geometry('900x200')

    l = tk.Label(window,
        text= 'Do these links refer to the same publication?',
        font=('Arial', 12),
        width=100,
        height=2)
    l.pack()

    l2 = tk.Entry(window,
#        textvariable =dic['string'][i],
        font=('Arial', 10),
        width= 100)
    l2.pack()
    l2 .insert(0, dic['string'][i])
    l2.configure(state="readonly")

    l3 = tk.Entry(window,
#        textvariable=dic['string'][j],
        font=('Arial', 10),
        width= 100)
    l3.pack()
    l3.pack()
    l3 .insert(0, dic['string'][j])
    l3.configure(state="readonly")

    def hit_me1():
        global on_hit
        on_hit = 1
        close_window()
     #       window.destroy()
    def hit_me2():
        global on_hit
        on_hit = 0
        close_window()

    def close_window():
        window.destroy()

    b = tk.Button(window,
        text='similar',
        width=25, height=2,
        command=hit_me1)
    b.pack(side = tk.RIGHT)
    b2 = tk.Button(window,
        text='different',
        width=25, height=2,
        command=hit_me2)
    b2.pack(side = tk.LEFT)
    window.mainloop()

    return on_hit

# This function create matrix of similarity between references and return matrix and dictionary of manually checked refereces
# To do introduce checking in manuallychecked list
def compare_refs(dic):
    len_matrix = len(dic['string'])
    m = np.empty((len_matrix, len_matrix))

    for i in range(0, len_matrix):
        for j in range(0, len_matrix):
            if i == j:
                m[i, j] = 1
            elif i < j:
                continue
            elif dic['art_id'][i] == dic['art_id'][j]:
                continue
            else:
                #l += 1
                pati, patj = create_pattern1_2(dic, i, j)
                if pati.lower() == patj.lower():
                    #arrow_dict["arrow1"] += 1
                    m[i, j] = 1
                else:
                    doi_comp = comp(dic['doi'][i], dic['doi'][j])
                    name_comp = comp(dic['name'][i], dic['name'][j])
                    title_com = comp(dic['title'][i], dic['title'][j], typ_comp='title')
                    year_com = comp(dic['year'][i], dic['year'][j])
                    vol_com = comp(dic['issue'][i], dic['issue'][j])
                    page_com = comp(dic['page'][i], dic['page'][j])
                    if (name_comp in [2, 0]) and title_com == 4:
                        #arrow_dict["leven"] += 1
                    if (doi_comp == 2) and (name_comp== 2):
                        #arrow_dict["arrow2"] += 1
                        m[i, j] = 1
                    elif (doi_comp == 2) and (name_comp in [1,3]):
                        #arrow_dict["arrow3"] += 1
                        if check_dictionary(dic, i, j) ==1:
                            m[i, j] = 1
                    elif (doi_comp == 2) and (name_comp ==0):
                        #arrow_dict["arrow4"] +=1
                        if check_dictionary(dic, i, j) ==1:
                            m[i, j] = 1

                    elif (doi_comp in [1, 0]) and (name_comp == 2) and (title_com in [2, 4]) and (year_com == 2) and (vol_com in [2, 0]) and (page_com == 3):
                        #arrow_dict["arrow6"] += 1
                        if check_dictionary(dic, i, j) ==1:
                            m[i, j] = 1
                    elif (doi_comp == 3) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 2) and (vol_com in [2, 0]) and (page_com == 3):
                        #arrow_dict["arrow7"] += 1
                    elif (doi_comp in [1,0]) and (name_comp == 2) and (title_com == 2) and (year_com == 2) and (vol_com in [0, 2]) and (page_com == 0):
                        #arrow_dict["arrow9"] += 1
                        m[i, j] = 1
    #                    print(dic['string'][i], '-----', dic['string'][j])
                    elif (doi_comp in [1,0]) and (name_comp == 2) and (title_com == 4) and (year_com ==2 ) and (vol_com in [0, 2]) and (page_com == 0):
                        #arrow_dict["arrow99"] += 1
                        if check_dictionary(dic, i, j) == 1:
                            m[i, j] = 1
                    elif (doi_comp == 3) and (name_comp == 2) and (title_com in [4, 2]) and (year_com ==2)and (vol_com == 0) and (page_com == 0):
                        #arrow_dict["arrow10"] +=1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com == 2) and (year_com ==2) and (vol_com in [2, 0]) and (page_com == 2):
                        #arrow_dict["arrow5"] += 1
                        m[i, j] = 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com ==2) and (vol_com in [2, 0]) and (page_com == 1):
                        #arrow_dict["arrow8"] +=1
                        if check_dictionary(dic, i, j) == 1:
                            m[i, j] = 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 2) and (vol_com == 3):
                        #arrow_dict["arrow11"] += 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 2) and (vol_com == 1):
                        #arrow_dict["arrow12"] += 1
                        if check_dictionary(dic, i, j) == 1:
                            m[i, j] = 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 3) and (vol_com in [1, 2, 0]):
                        #arrow_dict["arrow13"] += 1
                        if check_dictionary(dic, i, j) == 1:
                            m[i, j] = 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 3) and (vol_com == 3):
                        #arrow_dict["arrow14"] += 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 1):
                        #arrow_dict["arrow15"] += 1
                        if check_dictionary(dic, i, j) ==1:
                            m[i, j] = 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com in [4, 2]) and (year_com == 0):
                        #arrow_dict["arrow16"] += 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 2) and (title_com == 4) and (year_com ==2) and (vol_com in [2, 0]) and (page_com == 2):
                        #arrow_dict["arrow55"] +=1

                    elif (doi_comp in [3, 1, 0]) and (name_comp in [0, 2]) and (title_com == 3):
                        #arrow_dict["arrow18"] += 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 3):
                        #arrow_dict["arrow19"] += 1
                    elif (doi_comp in [3, 1, 0]) and (name_comp == 1):
                        #arrow_dict["arrow20"] += 1
                    else:
                        #arrow_dict["rest"] += 1
                        print(dic['string'][i], '-----', dic['string'][j])

    print('Referrences are matched. Number of references: '+ str(l))
    return m

# This function calculates cosine similarity between 2 multitudes
def counter_cosine_similarity(matrix):
    shared_refs = np.count_nonzero(matrix)
    magA = math.sqrt(matrix.shape[0])
    magB = math.sqrt(matrix.shape[1])
    return shared_refs / (magA * magB)

# This function  creates matrix of similarity form df to numpy array
def create_matrix_similarity_df(ref_save, matrix, treshold):
    ids_art = list(set(ref_save['art_id']))
    matrix_similarity = np.empty([len(ids_art), len(ids_art)])
    print ()
    for i in ids_art:
        for j in ids_art:
            if i==j:
                matrix_similarity[i,j] = 1
            else:
                ref_i = list(ref_save[ref_save['art_id'] == i]['ref_id'])
                ref_j = list(ref_save[ref_save['art_id'] == j]['ref_id'])
                cut_matrix = matrix[ref_j[0]:ref_j[-1]+1, ref_i[0]:ref_i[-1]+1]
                cs = counter_cosine_similarity(cut_matrix)
                if cs < treshold:
                    cs = 0
                matrix_similarity[i, j] = cs

    print('Matrix of similarity is created')
    return matrix_similarity

# This function calculates clusters
def calculate_clusters (articles_df, matrix_similarity):
    connected_components = scipy.sparse.csgraph.connected_components(matrix_similarity, directed=False)[1]
    articles_df['Conn_Comp'] = connected_components

    print('Clusters are calculated')
    return articles_df

# This function gather orphans in one cluster
def gather_orphans(articles_df):
    orphs = []
    for i in articles_df.Conn_Comp.unique():
        if articles_df[articles_df.Conn_Comp == i].shape[0] == 1:
            orphs.append(i)
    articles_df['Conn_Comp'] = np.where((articles_df.Conn_Comp.isin(orphs)), -1, articles_df.Conn_Comp)
    l_of_clust = [i for i in list(articles_df.Conn_Comp.unique()) if i != -1]
    for i in l_of_clust:
        articles_df['Conn_Comp'] = np.where(articles_df.Conn_Comp == i, l_of_clust.index(i), articles_df.Conn_Comp)
    return articles_df

def get_ranges(articles_df):
    ran = {}
    l_clust = list(articles_df['Conn_Comp'].value_counts(normalize=True).index)
    for year in articles_df.PY.unique():
        rand = 110
        ranges = {}
        year_df = articles_df[articles_df.PY==year]['Conn_Comp'].value_counts(normalize=True)
        for ind in l_clust:
            if ind in list(year_df.index):
                value = year_df.loc[ind]
                ranges[ind] = [int(rand - (110 * value)) + 5, int(rand) + 5]
                rand = rand - (110 * value)
        ran[year] = copy.deepcopy(ranges)

    return ran

# This func create y axis numbers on base
def vertical_layout(articles_df):
    ran = get_ranges(articles_df)
    YLayout = []
    for index, row  in articles_df.iterrows():
        r = ran[row['PY']][row['Conn_Comp']]
        YLayout.append(r[1])
        ran[row['PY']][row['Conn_Comp']][1] = r[1]-6

    return YLayout

def pltcolor(a_df_lst):
    colors = ['MediumSlateBlue', 'DarkSeaGreen', 'LimeGreen', 'Gold', 'MistyRose',
              'green', 'yellow', 'blue', 'darkorange', 'tomato', 'palevioletred', 'plum',
              'skyblue', 'silver', 'Cyan']
    cols = []
    for l in a_df_lst:
        cols.append(colors[l])
    return cols

def cursor_mltplchs_window():
    options = list_of_options_info
    root = tk.Tk()
    root.geometry('350x500')

    l = tk.Label(root ,
                 text='Select options with LMB+CTLR, press Select Button and close window',
                 font=('Arial', 10),
                 width=100,
                 height=2)
    l.pack()
    listbox = tk.Listbox(root, width=40, height=70, selectmode='extended')
    for item in options:
        listbox.insert(options.index(item)+1, item)

    result = []
    def selected_item():
        for i in listbox.curselection():
            result.append(listbox.get(i))

    btn = tk.Button(root, text="Select information for annotation", command=selected_item)
    btn.pack(side='bottom')
    listbox.pack()
    root.mainloop()

    return result

# Function for create one plot
def create_one_plot(articles_df, title_of_pic, result):

    import mplcursors
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('TkAgg')

    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [8, 1], 'hspace': 0.03}, figsize=(18, 10), dpi=120)
    articles_df['YLayout'] = vertical_layout(articles_df)
    articles_df = gather_orphans(articles_df)
    colors = pltcolor(list(articles_df.Conn_Comp))
    ax[0].scatter(articles_df.PY, articles_df.YLayout, c=colors, marker='o',
                  linewidths=0.7, edgecolors='gray', s=articles_df.TCg)
    ax[0].get_yaxis().set_visible(False)

    ax[1].yaxis.set_visible(False)
    # legend TODODODODOD
    crs = mplcursors.cursor(ax[0], hover=False, multiple=True)
    result = [item for item in art_df_colnames if art_df_colnames.index(item) in [list_of_options_info.index(a) for a in result]]
    crs.connect("add", lambda sel: sel.annotation.set(text=
                (articles_df.loc[sel.target.index, result].to_string()), ha='left'))

    ax[0].set_title(title_of_pic)
    plt.show()





