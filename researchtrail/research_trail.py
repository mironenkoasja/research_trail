import pandas as pd
from os import chdir, listdir
from research_trail_plot import *

path_to_wos = 'D:/python/research_trail/data_wos/test'
treshold = 0.05
title_of_pic = 'new???'

def research_trail_generate(path_to_wos, treshold, title_of_pic)

chdir(path_to_wos)

txt_files = [txt for txt in listdir(os.getcwd()) if txt.endswith('.txt')]
articles_df = pd.read_table(txt_files[0], index_col=False)

articles_df = transform_PD(articles_df)
articles_df = articles_df.query('CR==CR').reset_index()
dic = get_references_from_art(articles_df)
ref_save = pd.DataFrame(dic)
# ref_save.to_csv('ref_parse_SSS_all.csv')
matrix, arrow_dict = compare_refs(dic)
matrix_similarity = create_matrix_similarity_df(ref_save, matrix, treshold)
articles_df = calculate_clusters(articles_df, matrix_similarity)
result = cursor_mltplchs_window()

create_one_plot(articles_df, title_of_pic, result)