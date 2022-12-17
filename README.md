# Research trail
Python code constructs Research Trails using metadata of publication from the Web of Science

This package has only one main function:


research_trail_generate(path_to_wos, treshold, title_of_pic) 
     Calculates research trail for publications from Web of Science 

Args:
    path_to_wos (str): The file location of the output from web of science in txt
    treshold (float): The treshold for clustering of matrix of similarity 
    title_of_pic (str): The name for saved picture

Returns:
    matrix_similarity: matrix of similarity
    articles_df: The pandas data frame of original data with added variable: clusters 


More about research trails :

- Gläser, Jochen; Laudel, Grit, 2015. A Bibliometric Reconstruction of Research Trails for Qualitative Investigations of Scientific Innovations. Historical Social Research – Historische Sozialforschung 40: 299-330. http://www.laudel.info/wp-content/uploads/2013/12/Gl%C3%A4ser_2015_Bibliometric-Reconstruction.pdf

- Manual for counstructing research trails with Excel Macros http://www.laudel.info/downloads/research-trail-download/
