""" research_trail_generate(path_to_wos, treshold, title_of_pic) 
      Calculates research trail for publications from Web of Science 

Args:
    path_to_wos (str): The file location of the spreadsheet
    treshold (float): The treshold for clustering of matrix of similarity 
    title_of_pic (str): The name for saved picture

Returns:
    matrix_similarity: matrix of similarity
    articles_df: The pandas data frame of original data with added variable: clusters 
"""
